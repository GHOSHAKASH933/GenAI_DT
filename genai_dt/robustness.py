"""
Robustness check: is the ungated twin's failure an artefact of an unfair baseline?

A reviewer will reasonably object that no deployed system doses without an
insulin-on-board (IOB) constraint. We therefore re-run the ungated twin with two
standard clinical safeguards added, and ask whether they rescue it:

  ungated          : MPC on the posterior mean (as in the main experiments)
  ungated + IOB    : subtract active insulin from the computed dose
  ungated + IOB+PLGS : additionally suspend dosing on predicted low glucose

If the failure survives both, it is a property of the twin's treatment-effect
error rather than of a naive controller.
"""
from __future__ import annotations
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

import argparse, json, hashlib
import numpy as np
import torch

from . import data as D
from .simulator import build_cohort, simulate_closed_loop
from .models import CVAETwin, train_cvae
from .control import glycaemic_metrics, DOSE_GRID, TARGET, HYPO


def _stable_seed(*parts):
    payload = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=4).digest(), "big")


def guarded_twin_controller(model, params, use_iob=True, use_plgs=True, n_samples=24):
    from .control import _batch_x

    def f(hg, hi, hc, s):
        g = hg[-1]
        trend = (hg[-1] - hg[-4]) / 3.0 if len(hg) >= 4 else 0.0
        iob = hi[-36:].sum() * 0.55 if len(hi) else 0.0
        if use_plgs and (g + trend * 12 < HYPO + 15 or g < 90):
            return 0.0, False
        X = torch.as_tensor(_batch_x(hg, hi, hc, DOSE_GRID))
        with torch.no_grad():
            samp = model.sample(X, n_samples=n_samples)
        mean_end = D.denorm(samp.numpy())[:, :, -1].mean(axis=0)
        ok = np.where(mean_end <= TARGET)[0]
        dose = float(DOSE_GRID[ok[0]] if len(ok) else DOSE_GRID[-1])
        if use_iob:
            dose = max(0.0, dose - 0.35 * iob)
        return float(np.clip(dose, 0.0, 3.0)), False

    return f


def run_seed(seed, cfg):
    coh = build_cohort(seed, patients_per_site=cfg["patients_per_site"],
                       n_days=cfg["n_days"])
    tr, _, te = D.split_by_patient(coh, seed)
    Xo, Yo, _ = D.pack(tr)
    torch.manual_seed(seed)
    twin = train_cvae(CVAETwin(), Xo, Yo, epochs=cfg["epochs"], seed=seed)

    arms = {"ungated": dict(use_iob=False, use_plgs=False),
            "ungated_iob": dict(use_iob=True, use_plgs=False),
            "ungated_iob_plgs": dict(use_iob=True, use_plgs=True)}
    out = {}
    for name, kw in arms.items():
        per = []
        for rec in te[:cfg["loop_patients"]]:
            rng = np.random.default_rng(_stable_seed(seed, rec["pid"], "rb"))
            ctrl = guarded_twin_controller(twin, rec["params"],
                                           n_samples=cfg["loop_samples"], **kw)
            trc = simulate_closed_loop(rec["params"], cfg["loop_days"], rng, ctrl,
                                       decision_period=6)
            per.append(glycaemic_metrics(trc))
        out[name] = per
    return dict(seed=seed, arms=out)


CFG = dict(patients_per_site=12, n_days=10, epochs=30,
           loop_patients=10, loop_days=5, loop_samples=24)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--out", default=str(ROOT / "results" / "robustness.json"))
    a = ap.parse_args()
    runs = []
    for s in range(a.seeds):
        runs.append(run_seed(s, CFG))
        print(f"  robustness seed {s} done", flush=True)
        with open(a.out, "w") as f:
            json.dump(dict(config=CFG, runs=runs), f, indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
