"""
Split-conformal calibration of the generative twin.

The twin emits samples over future trajectories. We conformalise the derived
quantity that actually governs the therapeutic decision: the MINIMUM glucose over
the 60-min horizon, i.e. the hypoglycaemia risk of acting now.

Split conformal gives finite-sample marginal coverage >= 1 - alpha under
exchangeability, with no distributional assumption on the twin. The calibration
patients are disjoint from both training and test patients.
"""
from __future__ import annotations

import numpy as np
import torch

from . import data as D


def twin_quantiles(model, X, alpha=0.10, n_samples=96, batch=2048):
    """
    -> (q_lo, q_med, q_hi) for min-over-horizon glucose, in mg/dL.
    """
    los, meds, his = [], [], []
    Xt = torch.as_tensor(X)
    with torch.no_grad():
        for s in range(0, len(Xt), batch):
            xb = Xt[s:s + batch]
            samp = model.sample(xb, n_samples=n_samples)          # (S, B, H)
            gmin = samp.min(dim=-1).values.numpy()                # (S, B)
            gmin = D.denorm(gmin)
            los.append(np.quantile(gmin, alpha / 2, axis=0))
            meds.append(np.quantile(gmin, 0.5, axis=0))
            his.append(np.quantile(gmin, 1 - alpha / 2, axis=0))
    return (np.concatenate(los), np.concatenate(meds), np.concatenate(his))


def horizon_mean_pred(model, X, n_samples=96, batch=2048):
    """Point prediction of end-of-horizon glucose [mg/dL] (posterior mean)."""
    out = []
    Xt = torch.as_tensor(X)
    with torch.no_grad():
        for s in range(0, len(Xt), batch):
            samp = model.sample(Xt[s:s + batch], n_samples=n_samples)
            out.append(D.denorm(samp.mean(0).numpy()))
    return np.concatenate(out)


class ConformalGate:
    """
    Calibrated abstention for a generative twin.

    fit()     : computes the conformal correction Q from held-out calibration patients
    interval(): returns a coverage-guaranteed interval on min-horizon glucose
    decide()  : maps that interval to {act, attenuate, abstain}
    """

    def __init__(self, alpha=0.10, hypo_mgdl=70.0, width_tau=None):
        self.alpha = alpha
        self.hypo = hypo_mgdl
        self.Q = None
        self.width_tau = width_tau

    @staticmethod
    def _no_future_carbs(X):
        """Calibration windows whose action plan contains no announced meal.

        The controller queries the twin with zero future carbohydrate, so
        calibrating on windows that *do* contain a meal would violate
        exchangeability between calibration and deployment. Restricting the
        calibration set to matched action support restores it approximately.
        Residual shift (dose magnitude) is reported as a limitation.
        """
        lo = D.CONTEXT * 3 + D.HORIZON
        return np.isclose(X[:, lo:lo + D.HORIZON].sum(axis=1), 0.0)

    def fit(self, model, Xc, Yc, width_quantile=0.90, match_action_support=True):
        if match_action_support:
            m = self._no_future_carbs(Xc)
            if m.sum() >= 200:
                Xc, Yc = Xc[m], Yc[m]
            self.n_calib_matched = int(m.sum())
        self.n_calib = int(len(Xc))
        q_lo, _, q_hi = twin_quantiles(model, Xc, alpha=self.alpha)
        y_true_min = D.denorm(Yc).min(axis=1)
        scores = np.maximum(q_lo - y_true_min, y_true_min - q_hi)
        n = len(scores)
        k = int(np.ceil((n + 1) * (1 - self.alpha)))
        k = min(max(k, 1), n)
        self.Q = float(np.sort(scores)[k - 1])
        # abstention threshold: the width above which the twin is in its own tail
        widths = (q_hi + self.Q) - (q_lo - self.Q)
        if self.width_tau is None:
            self.width_tau = float(np.quantile(widths, width_quantile))
        self.calib_width_median = float(np.median(widths))
        return self

    def interval(self, model, X):
        q_lo, q_med, q_hi = twin_quantiles(model, X, alpha=self.alpha)
        return q_lo - self.Q, q_med, q_hi + self.Q

    def coverage(self, model, X, Y):
        lo, _, hi = self.interval(model, X)
        y = D.denorm(Y).min(axis=1)
        return float(((y >= lo) & (y <= hi)).mean()), float(np.mean(hi - lo))

    def decide(self, lo, hi):
        """
        Returns (action_scale, abstain).
          abstain          -> twin too uncertain; caller uses the fallback controller
          action_scale=0   -> predicted hypo risk; withhold correction
          action_scale=1   -> act on the twin
        """
        if (hi - lo) > self.width_tau:
            return 0.0, True
        if lo < self.hypo:
            return 0.0, False
        return 1.0, False
