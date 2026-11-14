"""
Correction-bolus controllers evaluated in closed loop.

The twin controllers are model-predictive: a candidate dose is written into the
twin's action channel and the resulting counterfactual trajectory distribution is
scored. The two twin arms differ ONLY in how that distribution is used.

  twin_ungated : acts on the posterior MEAN. This is what the digital-twin
                 literature describes when it says a twin "recommends treatment".
  twin_gated   : acts on the conformal LOWER BOUND of the horizon minimum, and
                 abstains to the fallback controller when the calibrated interval
                 is too wide to support any decision.
"""
from __future__ import annotations

import numpy as np
import torch

from . import data as D
from .conformal import ConformalGate

TARGET = 110.0
HYPO = 70.0
DOSE_GRID = np.arange(0.0, 3.01, 0.5, dtype=np.float32)


def _build_x(hist_g, hist_i, hist_c, dose):
    """Context + action plan, with `dose` delivered at the first horizon step."""
    g = hist_g[-D.CONTEXT:]
    i = hist_i[-D.CONTEXT:]
    c = hist_c[-D.CONTEXT:]
    fut_i = np.zeros(D.HORIZON, np.float32)
    fut_i[0] = dose
    fut_c = np.zeros(D.HORIZON, np.float32)      # upcoming meals unannounced
    return D.encode_context(g, i, c, fut_i, fut_c).astype(np.float32)


def _batch_x(hist_g, hist_i, hist_c, doses):
    return np.stack([_build_x(hist_g, hist_i, hist_c, d) for d in doses])


def basal_only(params):
    def f(hg, hi, hc, s):
        return 0.0, False
    return f


def pid_controller(params, kp=0.0045, kd=0.09, iob_window=36):
    """Conventional correction with insulin-on-board suppression and a
    predictive low-glucose cut-off. Also the gate's fallback."""
    def f(hg, hi, hc, s):
        g = hg[-1]
        trend = (hg[-1] - hg[-4]) / 3.0 if len(hg) >= 4 else 0.0
        iob = hi[-iob_window:].sum() * 0.55 if len(hi) else 0.0
        if g + trend * 12 < HYPO + 15 or g < 90:
            return 0.0, False
        dose = max(0.0, kp * (g - TARGET) + kd * trend - 0.35 * iob)
        return float(np.clip(dose, 0.0, 3.0)), False
    return f


def twin_controller(model, params, gate: ConformalGate | None, support=None,
                    n_samples=32):
    """
    Model-predictive correction dosing over the twin.

      gate=None, support=None : act on the posterior mean at every epoch. This
            is the policy implied by the digital-twin literature's claim that a
            twin can "recommend personalised treatment".
      gate only               : conformal outcome gate. Abstains when the
            calibrated interval on the horizon minimum is too wide.
      gate + support          : adds an action-support (positivity) test. A dose
            whose local training support is too thin is not a question the twin
            can answer, whatever its predictive interval looks like.

    On abstention the conventional controller acts instead.
    """
    fallback = pid_controller(params)

    def f(hg, hi, hc, s):
        X = torch.as_tensor(_batch_x(hg, hi, hc, DOSE_GRID))
        with torch.no_grad():
            samp = model.sample(X, n_samples=n_samples)          # (S, nD, H)
        traj = D.denorm(samp.numpy())
        mean_end = traj[:, :, -1].mean(axis=0)                   # per dose
        gmin = traj.min(axis=2)                                  # (S, nD)

        if gate is None:
            # act on the posterior mean: smallest dose bringing predicted
            # end-of-horizon glucose to target
            ok = np.where(mean_end <= TARGET)[0]
            return float(DOSE_GRID[ok[0]] if len(ok) else DOSE_GRID[-1]), False

        lo = np.quantile(gmin, gate.alpha / 2, axis=0) - gate.Q
        hi_ = np.quantile(gmin, 1 - gate.alpha / 2, axis=0) + gate.Q

        # abstain if the twin cannot resolve even the zero-dose outcome
        if (hi_ - lo)[0] > gate.width_tau:
            d, _ = fallback(hg, hi, hc, s)
            return d, True

        admissible = np.ones(len(DOSE_GRID), bool)
        if support is not None:
            admissible = support.admissible(X.numpy())
            if not admissible.any():
                d, _ = fallback(hg, hi, hc, s)
                return d, True

        # largest dose whose calibrated worst case stays above the hypo bound
        safe = np.where((lo >= HYPO) & admissible)[0]
        if len(safe) == 0:
            return 0.0, False
        cand = [j for j in safe if mean_end[j] <= TARGET]
        j = cand[0] if cand else safe[-1]
        return float(DOSE_GRID[j]), False

    return f


def glycaemic_metrics(trace):
    g = trace["truth"]
    below = g < HYPO
    events, run = 0, 0
    for b in below:
        run = run + 1 if b else 0
        if run == 3:          # >=15 min continuous => one event
            events += 1
    return dict(
        tir=float(((g >= 70) & (g <= 180)).mean() * 100),
        tbr=float(below.mean() * 100),
        tbr54=float((g < 54).mean() * 100),
        tar=float((g > 180).mean() * 100),
        hypo_events=int(events),
        mean_g=float(g.mean()),
        cv=float(g.std() / g.mean() * 100),
        total_correction=float(trace["correction"].sum()),
        abstain_rate=float(trace["abstained"].mean() * 100),
    )
