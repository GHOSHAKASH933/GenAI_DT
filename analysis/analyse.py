"""
Aggregate results across seeds with paired tests, bootstrap CIs and Holm
correction. No number enters the manuscript except through this script.
"""
from __future__ import annotations
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

import json, sys, os
import numpy as np
from scipy import stats

RES = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "results" / "results.json")


def load(path=RES):
    with open(path) as f:
        d = json.load(f)
    return d["config"], d["runs"]


def boot_ci(x, n=10000, seed=0, agg=np.mean):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    bs = agg(rng.choice(x, (n, len(x)), replace=True), axis=1)
    return (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))


def fmt(x, lo, hi, p=2):
    return f"{x:.{p}f} [{lo:.{p}f}, {hi:.{p}f}]"


def holm(pvals, labels):
    order = np.argsort(pvals)
    m = len(pvals)
    out = {}
    running = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (m - rank) * pvals[i])
        running = max(running, adj)
        out[labels[i]] = running
    return out


def min_attainable_p(n):
    """
    Smallest two-sided Wilcoxon signed-rank p-value achievable with n pairs.

    With few pairs the test cannot reach significance however large the effect.
    Reporting such a comparison as a significance test is misleading, so every
    caller checks this and reports descriptively instead when the floor exceeds
    the threshold.
    """
    return 2.0 / (2 ** n) if n > 0 else 1.0


def powered(n, alpha=0.05):
    return min_attainable_p(n) <= alpha


def paired(a, b):
    """Wilcoxon signed-rank (primary, no normality assumption) + paired t."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if len(a) < 3 or np.allclose(a, b):
        return dict(n=int(len(a)), wilcoxon_p=float("nan"), t_p=float("nan"),
                    median_diff=float(np.median(a - b)) if len(a) else float("nan"),
                    min_p=min_attainable_p(len(a)), powered=powered(len(a)),
                    cohens_dz=float("nan"))
    try:
        w = float(stats.wilcoxon(a, b).pvalue)
    except Exception:
        w = float("nan")
    t = float(stats.ttest_rel(a, b).pvalue)
    d = a - b
    return dict(n=int(len(a)), wilcoxon_p=w, t_p=t,
                median_diff=float(np.median(d)),
                min_p=min_attainable_p(len(a)), powered=powered(len(a)),
                cohens_dz=float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else float("nan"))


# ------------------------------------------------------------------ E1

def table_e1(runs):
    keys = [k for k in runs[0]["e1"] if not k.startswith("_")]
    rows = []
    for k in sorted(keys):
        ratios = [r["e1"][k]["ratio"] for r in runs]
        slopes = [r["e1"][k]["slope_mean"] for r in runs]
        lo, hi = boot_ci(ratios)
        rows.append((k, np.mean(slopes), np.mean(ratios), lo, hi))
    return rows


def print_e1(runs):
    print("\n=== E1  treatment-effect recovery (ratio 1.0 = correct) ===")
    true_s = [r["true_sensitivity"] for r in runs]
    print(f"true sensitivity across seeds: {np.mean(true_s):.2f} "
          f"(sd {np.std(true_s):.2f}) mg/dL per U")
    print(f"{'data | model':46s} {'slope':>8s}  {'ratio [95% CI]':>26s}")
    for k, sl, r, lo, hi in table_e1(runs):
        print(f"{k:46s} {sl:8.2f}  {fmt(r, lo, hi):>26s}")

    # key comparisons
    print("\n  paired comparisons on recovery ratio:")
    comps = [
        ("observational|cvae", "interventional_randomised|cvae"),
        ("interventional_titrated|cvae", "interventional_titrated|ar_twin"),
        ("observational|cvae", "observational|point_mlp"),
        ("interventional_randomised|cvae", "interventional_randomised|point_mlp"),
    ]
    ps, labs, stash = [], [], {}
    for a, b in comps:
        ra = [r["e1"][a]["ratio"] for r in runs]
        rb = [r["e1"][b]["ratio"] for r in runs]
        st = paired(rb, ra)
        lab = f"{b} vs {a}"
        stash[lab] = st
        if np.isfinite(st["wilcoxon_p"]):
            ps.append(st["wilcoxon_p"]); labs.append(lab)
    adj = holm(ps, labs) if ps else {}
    for lab, st in stash.items():
        pa = adj.get(lab, float("nan"))
        print(f"    {lab:70s} Δmed={st['median_diff']:+6.3f} "
              f"p={st['wilcoxon_p']:.4f} p_holm={pa:.4f} dz={st['cohens_dz']:+.2f}")


# ------------------------------------------------------------------ E2

def print_e2(runs):
    print("\n=== E2  60/120-min forecasting RMSE (mg/dL) ===")
    keys = list(runs[0]["e2"].keys())
    print(f"{'model':32s} {'RMSE [95% CI]':>24s} {'MAE':>8s}")
    for k in keys:
        r = [x["e2"][k][0] for x in runs]
        m = [x["e2"][k][1] for x in runs]
        lo, hi = boot_ci(r)
        print(f"{k:32s} {fmt(np.mean(r), lo, hi):>24s} {np.mean(m):8.2f}")
    print("\n  federation comparisons (RMSE, lower better):")
    base = [x["e2"]["twin_local_only"][0] for x in runs]
    for k in ("twin_fedavg", "twin_fedavg_personalised", "twin_centralised"):
        st = paired(base, [x["e2"][k][0] for x in runs])
        print(f"    local_only vs {k:28s} Δmed={st['median_diff']:+7.3f} "
              f"p={st['wilcoxon_p']:.4f} dz={st['cohens_dz']:+.2f}")


# ------------------------------------------------------------------ E3

def print_e3(runs):
    print("\n=== E3  calibration of the horizon-minimum interval ===")
    for lab in ("obs", "int"):
        rc = [r["calibration"][lab]["raw_coverage"] for r in runs]
        cc = [r["calibration"][lab]["conformal_coverage"] for r in runs]
        rw = [r["calibration"][lab]["raw_width"] for r in runs]
        cw = [r["calibration"][lab]["conformal_width"] for r in runs]
        nm = runs[0]["calibration"][lab]["nominal"]
        print(f"  twin={lab}  nominal={nm:.2f}")
        print(f"    raw        coverage {fmt(np.mean(rc), *boot_ci(rc), 3)}  "
              f"width {np.mean(rw):6.2f}")
        print(f"    conformal  coverage {fmt(np.mean(cc), *boot_ci(cc), 3)}  "
              f"width {np.mean(cw):6.2f}")
        st = paired(np.abs(np.array(cc) - nm), np.abs(np.array(rc) - nm))
        print(f"    |coverage-nominal| improved: Δmed={st['median_diff']:+.4f} "
              f"p={st['wilcoxon_p']:.4f}")


# ------------------------------------------------------------------ E4

ARMS = ["basal_only", "pid", "twin_obs_ungated", "twin_obs_conformal",
        "twin_obs_dual", "twin_int_dual"]
METRICS = [("tir", "TIR%", 1), ("tbr", "TBR<70%", 2), ("tbr54", "TBR<54%", 2),
           ("tar", "TAR>180%", 1), ("hypo_events", "hypo ev", 2),
           ("total_correction", "insulin U", 1), ("abstain_rate", "abstain%", 1)]


def per_seed(runs, regime, arm, metric):
    """Mean over patients within a seed -> one value per seed (paired unit)."""
    return [float(np.mean([p[metric] for p in r["loop"][regime][arm]])) for r in runs]


def print_e4(runs):
    for regime in ("matched", "shifted"):
        print(f"\n=== E4  closed loop, regime = {regime} ===")
        hdr = f"{'arm':22s}" + "".join(f"{lbl:>14s}" for _, lbl, _ in METRICS)
        print(hdr)
        for arm in ARMS:
            line = f"{arm:22s}"
            for key, _, p in METRICS:
                v = per_seed(runs, regime, arm, key)
                line += f"{np.mean(v):14.{p}f}"
            print(line)

        print("  safety comparisons vs twin_obs_ungated (TBR<70, lower better):")
        ref = per_seed(runs, regime, "twin_obs_ungated", "tbr")
        ps, labs, stash = [], [], {}
        for arm in ("twin_obs_conformal", "twin_obs_dual", "twin_int_dual", "pid"):
            st = paired(ref, per_seed(runs, regime, arm, "tbr"))
            stash[arm] = st
            if np.isfinite(st["wilcoxon_p"]):
                ps.append(st["wilcoxon_p"]); labs.append(arm)
        adj = holm(ps, labs) if ps else {}
        for arm, st in stash.items():
            print(f"    {arm:22s} Δmed={st['median_diff']:+7.3f} pp  "
                  f"p={st['wilcoxon_p']:.4f} p_holm={adj.get(arm, float('nan')):.4f} "
                  f"dz={st['cohens_dz']:+.2f}")

        print("  TIR non-inferiority of twin_int_dual vs pid:")
        a = per_seed(runs, regime, "twin_int_dual", "tir")
        b = per_seed(runs, regime, "pid", "tir")
        d = np.array(a) - np.array(b)
        lo, hi = boot_ci(d)
        print(f"    ΔTIR = {fmt(np.mean(d), lo, hi)} pp "
              f"(non-inferior if lower bound > -5 pp)")


# ------------------------------------------------------------------ E5

def print_e5(runs):
    print("\n=== E5  edge distillation ===")
    for k in ("teacher_rmse", "student_rmse", "teacher_latency_ms",
              "student_latency_ms", "teacher_params", "student_params"):
        v = [r["e5"][k] for r in runs]
        lo, hi = boot_ci(v)
        print(f"  {k:22s} {fmt(np.mean(v), lo, hi, 3)}")
    sp = [r["e5"]["teacher_latency_ms"] / r["e5"]["student_latency_ms"] for r in runs]
    cr = [r["e5"]["teacher_params"] / r["e5"]["student_params"] for r in runs]
    dr = [r["e5"]["student_rmse"] - r["e5"]["teacher_rmse"] for r in runs]
    print(f"  speedup      {fmt(np.mean(sp), *boot_ci(sp))}x")
    print(f"  compression  {fmt(np.mean(cr), *boot_ci(cr))}x")
    print(f"  ΔRMSE        {fmt(np.mean(dr), *boot_ci(dr))} mg/dL")


def main():
    cfg, runs = load()
    print(f"seeds = {len(runs)}   config = {cfg}")
    print(f"patients: train={runs[0]['n_train']} calib={runs[0]['n_calib']} "
          f"test={runs[0]['n_test']}  train windows={runs[0]['n_windows_train']}")
    print_e1(runs); print_e2(runs); print_e3(runs); print_e4(runs); print_e5(runs)


if __name__ == "__main__":
    main()
