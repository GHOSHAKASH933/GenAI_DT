"""
Interventional data generation and action-support gating.

Two findings drive this module:

1. Twins trained on OBSERVATIONAL care data learn essentially no dose-response.
   Correction boluses are rare and confounded with meals (insulin is given when
   glucose is about to rise), so the twin attributes little glucose-lowering to
   insulin. Model-predictive control over such a twin selects grossly excessive
   doses, because the twin does not believe the dose will act.

2. Conformal calibration of OUTCOME uncertainty does not detect this. The twin
   is confidently wrong about the *effect of an action*, not noisy about the
   outcome of a passive forecast. Calibration and causal validity are different
   failures and need different instruments.

So we add:
  - `simulate_interventional`: a randomised-dose exploration policy, giving the
    twin positivity/overlap over the action space it will later be queried on.
  - `ActionSupportGate`: a k-NN support test in (context, dose) space that
    refuses to answer queries outside the action support of the training data.
"""
from __future__ import annotations

import numpy as np
import torch

from . import data as D
from .simulator import simulate_closed_loop, CGM_PERIOD


def randomised_dose_policy(params, p_act=0.6, max_dose=2.5, seed=0,
                           state_independent=True):
    """
    Safety-constrained exploration policy. At each decision epoch, with
    probability `p_act`, a correction dose is drawn and delivered, subject to a
    hard low-glucose and insulin-on-board lockout.

    `state_independent` controls how the dose MAGNITUDE is drawn, and it is the
    single most consequential design choice in this study:

      True  : magnitude ~ U(0, max_dose), independent of the patient's state
              given that the lockout passed. Within the admissible region the
              dose is unconfounded, so the dose-response is identified.
      False : magnitude scaled by current glucose, as a cautious clinician would
              titrate. This is safer per-decision but re-couples dose to state,
              degrades overlap, and leaves the dose-response unidentified — the
              twin then learns a severely attenuated treatment effect even
              though the data is nominally "interventional".

    We report both, because the second is what a well-intentioned data-collection
    protocol would naturally produce.
    """
    rng = np.random.default_rng(seed)

    def f(hg, hi, hc, s):
        g = hg[-1]
        trend = (hg[-1] - hg[-4]) / 3.0 if len(hg) >= 4 else 0.0
        iob = hi[-36:].sum() * 0.55 if len(hi) else 0.0
        if g < 100 or g + trend * 12 < 90 or iob > 2.5:
            return 0.0, False
        if rng.random() > p_act:
            return 0.0, False
        if state_independent:
            return float(rng.uniform(0.0, max_dose)), False
        ceiling = min(max_dose, max(0.0, (g - 90.0) / 35.0))
        return float(rng.uniform(0.0, ceiling)), False

    return f


def build_interventional_cohort(cohort, n_days, seed, p_act=0.6,
                                state_independent=True, max_dose=2.5):
    """
    Re-record each patient under the exploration policy. Same physiology and
    same patients; only the action distribution changes.
    """
    out = []
    for k, rec in enumerate(cohort):
        rng = np.random.default_rng((seed * 7919 + k) % (2 ** 32))
        ctrl = randomised_dose_policy(rec["params"], p_act=p_act, max_dose=max_dose,
                                      seed=(seed * 104729 + k) % (2 ** 32),
                                      state_independent=state_independent)
        tr = simulate_closed_loop(rec["params"], n_days, rng, ctrl,
                                  exercise=False, decision_period=6)
        out.append(dict(glucose=tr["glucose"], truth=tr["truth"],
                        insulin=tr["insulin"], carbs=tr["carbs"],
                        site=rec["site"], params=rec["params"], pid=rec["pid"]))
    return out


# ------------------------------------------------------------------ gating

class ActionSupportGate:
    """
    Refuses twin queries whose (context, action) pair lies outside the training
    action support.

    Mechanism: project the context to a low-dimensional summary (current glucose,
    trend, insulin on board, recent carbs), then for a query dose d ask what
    fraction of the k nearest training contexts received a dose at least as
    large. If that fraction is below `min_support`, the twin is extrapolating
    about the action and the query is refused.

    This is a positivity/overlap check — the assumption that makes a
    counterfactual query identifiable — enforced at inference time.
    """

    def __init__(self, k=50, min_support=0.05):
        self.k = k
        self.min_support = min_support
        self.Z = None
        self.doses = None
        self.scale = None

    @staticmethod
    def summarise(X):
        """(N, CTX_DIM) -> (N, 4) context summary in natural units."""
        g = X[:, :D.CONTEXT] * D.G_SCALE + D.G_MEAN
        i = X[:, D.CONTEXT:2 * D.CONTEXT] * D.I_SCALE
        c = X[:, 2 * D.CONTEXT:3 * D.CONTEXT] * D.C_SCALE
        cur = g[:, -1]
        trend = (g[:, -1] - g[:, -4]) / 3.0
        iob = i[:, -36:].sum(axis=1) if i.shape[1] >= 36 else i.sum(axis=1)
        carb = c[:, -12:].sum(axis=1)
        return np.stack([cur, trend, iob, carb], axis=1)

    @staticmethod
    def query_dose(X):
        lo = D.CONTEXT * 3
        return X[:, lo] * D.I_SCALE

    def fit(self, Xtr):
        self.Z = self.summarise(Xtr)
        self.doses = self.query_dose(Xtr)
        self.scale = self.Z.std(axis=0) + 1e-6
        self.Zn = self.Z / self.scale
        return self

    def support(self, X):
        """-> fraction of local training neighbours that received >= the query dose."""
        Zq = self.summarise(X) / self.scale
        dq = self.query_dose(X)
        out = np.empty(len(X))
        for j in range(len(X)):
            d2 = ((self.Zn - Zq[j]) ** 2).sum(axis=1)
            idx = np.argpartition(d2, min(self.k, len(d2) - 1))[:self.k]
            out[j] = float((self.doses[idx] >= dq[j] - 1e-6).mean())
        return out

    def admissible(self, X):
        return self.support(X) >= self.min_support


def dose_response_slope(model, X, doses=(0.0, 0.5, 1.0, 1.5, 2.0), n_samples=32):
    """
    Estimate the twin's believed glucose-lowering effect, in mg/dL per unit,
    by linear regression of predicted end-of-horizon glucose on dose.

    Compared against each patient's true correction factor, this measures
    whether the twin has learned a usable causal dose-response at all.
    """
    lo = D.CONTEXT * 3
    ends = []
    for d in doses:
        Xd = X.copy()
        Xd[:, lo] = d / D.I_SCALE
        with torch.no_grad():
            s = model.sample(torch.as_tensor(Xd), n_samples=n_samples,
                             include_obs_noise=False)
        ends.append(D.denorm(s.mean(0).numpy())[:, -1])
    E = np.stack(ends)                       # (nD, N)
    dv = np.asarray(doses, dtype=np.float64)
    dc = dv - dv.mean()
    slope = (dc[:, None] * (E - E.mean(axis=0))).sum(axis=0) / (dc ** 2).sum()
    return -slope                            # positive = glucose falls per unit
