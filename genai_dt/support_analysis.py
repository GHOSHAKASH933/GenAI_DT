"""
Action-support profiling (no model training required).

Reports, for each training-data regime, the fraction of test contexts at which a
given correction dose has sufficient local support to be a well-posed query.
This isolates the positivity/overlap condition from everything else in the study.
"""
from __future__ import annotations
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

import argparse, json, os
import numpy as np

from . import data as D
from .simulator import build_cohort
from .intervention import ActionSupportGate, build_interventional_cohort
from .control import DOSE_GRID


def run_seed(seed, patients_per_site=12, n_days=10, k=50, min_support=0.05):
    coh = build_cohort(seed, patients_per_site=patients_per_site, n_days=n_days)
    tr, _, te = D.split_by_patient(coh, seed)
    Xte, _, _ = D.pack(te)
    lo = D.CONTEXT * 3

    regimes = {"observational": D.pack(tr)[0]}
    regimes["interventional_randomised"] = D.pack(
        build_interventional_cohort(tr, n_days, seed, state_independent=True))[0]
    regimes["interventional_titrated"] = D.pack(
        build_interventional_cohort(tr, n_days, seed, state_independent=False))[0]

    out = {}
    for name, Xtr in regimes.items():
        gate = ActionSupportGate(k=k, min_support=min_support).fit(Xtr)
        prof = {}
        for d in DOSE_GRID:
            X = Xte.copy()
            X[:, lo] = d / D.I_SCALE
            prof[f"{d:.1f}"] = float(gate.admissible(X).mean())
        doses = Xtr[:, lo] * D.I_SCALE
        out[name] = dict(profile=prof,
                         frac_treated=float((doses > 0).mean()),
                         mean_dose_treated=float(doses[doses > 0].mean())
                         if (doses > 0).any() else 0.0,
                         max_dose=float(doses.max()))
    return dict(seed=seed, regimes=out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--patients-per-site", type=int, default=12)
    ap.add_argument("--days", type=int, default=10)
    ap.add_argument("--out", default=str(ROOT / "results" / "support.json"))
    a = ap.parse_args()
    runs = []
    for s in range(a.seeds):
        runs.append(run_seed(s, a.patients_per_site, a.days))
        print(f"  support seed {s} done", flush=True)
        with open(a.out, "w") as f:
            json.dump(dict(doses=[float(d) for d in DOSE_GRID], runs=runs), f, indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
