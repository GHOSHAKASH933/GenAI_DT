"""Publication figures. Every panel is derived from results.json."""
from __future__ import annotations
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyse import load, boot_ci, per_seed, ARMS

FIG = str(ROOT / "results" / "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 8,
    "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "legend.frameon": False, "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

INK = "#1b1b1b"
C = {"obs": "#b3242b", "int_rand": "#1f6f8b", "int_tit": "#c98a1a",
     "ref": "#5a5a5a", "ok": "#2d6a4f", "warn": "#9b4dca"}
ARM_LABEL = {
    "basal_only": "Basal only", "pid": "Conventional (PID)",
    "twin_obs_ungated": "Twin, ungated", "twin_obs_conformal": "Twin + conformal",
    "twin_obs_dual": "Twin + dual gate", "twin_int_dual": "AR-Twin$^{\\mathrm{int}}$ + dual gate",
}


def _bars(ax, labels, means, los, his, colors, ylabel, hline=None, rot=25):
    x = np.arange(len(labels))
    err = np.vstack([np.array(means) - np.array(los), np.array(his) - np.array(means)])
    err = np.clip(err, 0, None)
    ax.bar(x, means, 0.66, color=colors, edgecolor=INK, linewidth=0.4,
           yerr=err, error_kw=dict(ecolor=INK, elinewidth=0.6, capsize=2))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=rot, ha="right" if rot else "center")
    ax.set_ylabel(ylabel)
    if hline is not None:
        ax.axhline(hline, color=INK, lw=0.7, ls=(0, (4, 2)), zorder=0)


# ---------------------------------------------------------------- Fig 1

def fig1_treatment_effect(runs):
    """The central result: recovery of the true dose-response."""
    combos = [
        ("observational|cvae", "Observational\nCVAE", C["obs"]),
        ("observational|ar_twin", "Observational\nAR-Twin", C["obs"]),
        ("observational|point_mlp", "Observational\nMLP (point)", C["ref"]),
        ("interventional_titrated|cvae", "Titrated\nCVAE", C["int_tit"]),
        ("interventional_titrated|ar_twin", "Titrated\nAR-Twin", C["int_tit"]),
        ("interventional_randomised|cvae", "Randomised\nCVAE", C["int_rand"]),
        ("interventional_randomised|ar_twin", "Randomised\nAR-Twin", C["int_rand"]),
        ("interventional_randomised|point_mlp", "Randomised\nMLP (point)", C["ref"]),
    ]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.1, 2.7),
                                   gridspec_kw=dict(width_ratios=[1.45, 1]))
    labels, means, los, his, cols = [], [], [], [], []
    for key, lab, col in combos:
        if key not in runs[0]["e1"]:
            continue
        v = [r["e1"][key]["ratio"] for r in runs]
        lo, hi = boot_ci(v)
        labels.append(lab); means.append(np.mean(v)); los.append(lo); his.append(hi); cols.append(col)
    _bars(ax1, labels, means, los, his, cols,
          "Recovered / true dose–response", hline=1.0, rot=35)
    ax1.set_ylim(0, max(1.35, max(his) * 1.15))
    ax1.text(0.015, 0.95, "correct", transform=ax1.transAxes, va="top", fontsize=6.5,
             color=INK)
    ax1.set_title("(a) Treatment-effect recovery", loc="left", fontsize=8.5)

    # absolute slopes vs truth
    true_s = np.mean([r["true_sensitivity"] for r in runs])
    keys = ["observational|cvae", "interventional_randomised|cvae",
            "interventional_randomised|ar_twin", "interventional_randomised|point_mlp"]
    lbl = ["Obs\nCVAE", "Rand\nCVAE", "Rand\nAR-Twin", "Rand\nMLP"]
    ms, ls, hs = [], [], []
    for k in keys:
        v = [r["e1"][k]["slope_mean"] for r in runs]
        lo, hi = boot_ci(v)
        ms.append(np.mean(v)); ls.append(lo); hs.append(hi)
    _bars(ax2, lbl, ms, ls, hs, [C["obs"], C["int_rand"], C["int_rand"], C["ref"]],
          "mg dL$^{-1}$ per unit", rot=0)
    ax2.axhline(true_s, color=C["ok"], lw=1.0)
    ax2.text(0.98, true_s, " true", color=C["ok"], va="bottom", ha="right",
             fontsize=6.5, transform=ax2.get_yaxis_transform())
    ax2.set_title("(b) Absolute sensitivity", loc="left", fontsize=8.5)
    fig.savefig(f"{FIG}/fig1_treatment_effect.pdf")
    fig.savefig(f"{FIG}/fig1_treatment_effect.png")
    plt.close(fig)


# ---------------------------------------------------------------- Fig 2

def fig2_calibration(runs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.1, 2.5))
    nominal = runs[0]["calibration"]["obs"]["nominal"]
    labs, ms, ls, hs, cols = [], [], [], [], []
    for lab, key, col in (("CVAE$^{\\mathrm{obs}}$", "obs", C["obs"]),
                          ("AR-Twin$^{\\mathrm{int}}$", "int", C["int_rand"])):
        for kind, hatch in (("raw_coverage", None), ("conformal_coverage", None)):
            v = [r["calibration"][key][kind] for r in runs]
            lo, hi = boot_ci(v)
            labs.append(f"{lab}\n{'uncalibrated' if 'raw' in kind else 'conformal'}")
            ms.append(np.mean(v)); ls.append(lo); hs.append(hi)
            cols.append(col if "raw" in kind else C["ok"])
    _bars(ax1, labs, ms, ls, hs, cols, "Empirical coverage", hline=nominal, rot=30)
    ax1.set_ylim(0, 1.02)
    ax1.set_title("(a) Interval coverage", loc="left", fontsize=8.5)

    labs, ms, ls, hs, cols = [], [], [], [], []
    for lab, key, col in (("CVAE$^{\\mathrm{obs}}$", "obs", C["obs"]),
                          ("AR-Twin$^{\\mathrm{int}}$", "int", C["int_rand"])):
        for kind in ("raw_width", "conformal_width"):
            v = [r["calibration"][key][kind] for r in runs]
            lo, hi = boot_ci(v)
            labs.append(f"{lab}\n{'uncal.' if 'raw' in kind else 'conformal'}")
            ms.append(np.mean(v)); ls.append(lo); hs.append(hi)
            cols.append(col if "raw" in kind else C["ok"])
    _bars(ax2, labs, ms, ls, hs, cols, "Interval width (mg dL$^{-1}$)", rot=30)
    ax2.set_title("(b) Interval width", loc="left", fontsize=8.5)
    fig.savefig(f"{FIG}/fig2_calibration.pdf"); fig.savefig(f"{FIG}/fig2_calibration.png")
    plt.close(fig)


# ---------------------------------------------------------------- Fig 3

def fig3_closed_loop(runs):
    fig, axes = plt.subplots(2, 3, figsize=(7.1, 4.4))
    panels = [("tir", "Time in range 70–180 (%)", False),
              ("tbr", "Time below 70 mg dL$^{-1}$ (%)", True),
              ("total_correction", "Correction insulin (U)", True)]
    for row, regime in enumerate(("matched", "shifted")):
        for col, (metric, ylabel, lower_better) in enumerate(panels):
            ax = axes[row, col]
            labs, ms, ls, hs, cols = [], [], [], [], []
            for arm in ARMS:
                v = per_seed(runs, regime, arm, metric)
                lo, hi = boot_ci(v)
                labs.append(ARM_LABEL[arm]); ms.append(np.mean(v))
                ls.append(lo); hs.append(hi)
                cols.append({"basal_only": C["ref"], "pid": "#7a7a7a",
                             "twin_obs_ungated": C["obs"],
                             "twin_obs_conformal": C["warn"],
                             "twin_obs_dual": C["int_tit"],
                             "twin_int_dual": C["ok"]}[arm])
            _bars(ax, labs, ms, ls, hs, cols, ylabel if row == 1 else "", rot=38)
            if row == 0:
                ax.set_title(f"({chr(97+col)}) {ylabel.split('(')[0].strip()}",
                             loc="left", fontsize=8.5)
                ax.set_xticklabels([])
            if col == 2:
                ax.text(1.04, 0.5, regime.upper(), transform=ax.transAxes,
                        rotation=270, va="center", ha="center", fontsize=7.5,
                        color=INK, fontweight="bold")
    fig.tight_layout(w_pad=1.4, h_pad=1.0)
    fig.savefig(f"{FIG}/fig3_closed_loop.pdf"); fig.savefig(f"{FIG}/fig3_closed_loop.png")
    plt.close(fig)


# ---------------------------------------------------------------- Fig 4

def fig4_prediction_and_edge(runs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.1, 2.6),
                                   gridspec_kw=dict(width_ratios=[1.3, 1]))
    order = ["persistence", "ridge_ar", "point_mlp_central", "twin_local_only",
             "twin_fedavg", "twin_fedavg_personalised", "twin_centralised"]
    nice = ["Persistence", "Ridge AR", "MLP (point)", "Local only",
            "FedAvg", "FedAvg + pers.", "Centralised"]
    ms, ls, hs = [], [], []
    for k in order:
        v = [r["e2"][k][0] for r in runs]
        lo, hi = boot_ci(v)
        ms.append(np.mean(v)); ls.append(lo); hs.append(hi)
    cols = [C["ref"], C["ref"], C["ref"], C["obs"], C["int_tit"], C["ok"], "#3a3a3a"]
    _bars(ax1, nice, ms, ls, hs, cols, "RMSE (mg dL$^{-1}$)", rot=35)
    ax1.set_title("(a) 2-hour forecasting error", loc="left", fontsize=8.5)

    sp = [r["e5"]["teacher_latency_ms"] / r["e5"]["student_latency_ms"] for r in runs]
    cr = [r["e5"]["teacher_params"] / r["e5"]["student_params"] for r in runs]
    dr = [r["e5"]["student_rmse"] - r["e5"]["teacher_rmse"] for r in runs]
    vals = [np.mean(sp), np.mean(cr), np.mean(dr)]
    cis = [boot_ci(sp), boot_ci(cr), boot_ci(dr)]
    _bars(ax2, ["Latency\nspeed-up (x)", "Parameter\nreduction (x)", "ΔRMSE\n(mg/dL)"],
          vals, [c[0] for c in cis], [c[1] for c in cis],
          [C["ok"], C["ok"], C["int_tit"]], "", rot=0)
    ax2.set_title("(b) Edge distillation", loc="left", fontsize=8.5)
    fig.savefig(f"{FIG}/fig4_prediction_edge.pdf")
    fig.savefig(f"{FIG}/fig4_prediction_edge.png")
    plt.close(fig)


def fig5_support(path=str(ROOT / "results" / "support.json")):
    """Admissibility of each candidate dose under each training regime."""
    if not os.path.exists(path):
        return False
    d = json.load(open(path))
    doses = d["doses"]; runs = d["runs"]
    series = [("observational", "Observational", C["obs"], "o-"),
              ("interventional_titrated", "Interventional, titrated", C["int_tit"], "s--"),
              ("interventional_randomised", "Interventional, randomised", C["int_rand"], "^-")]
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    for key, lab, col, style in series:
        ys, los, his = [], [], []
        for x in doses:
            v = [100 * r["regimes"][key]["profile"][f"{x:.1f}"] for r in runs]
            lo, hi = boot_ci(v)
            ys.append(np.mean(v)); los.append(lo); his.append(hi)
        ax.plot(doses, ys, style, color=col, lw=1.1, ms=3.2, label=lab)
        ax.fill_between(doses, los, his, color=col, alpha=0.16, lw=0)
    ax.axhline(5, color=INK, lw=0.7, ls=(0, (4, 2)))
    ax.text(doses[-1], 6.5, "gate threshold", ha="right", fontsize=6, color=INK)
    ax.set_xlabel("Queried correction dose (U)")
    ax.set_ylabel("Contexts with action support (%)")
    ax.set_ylim(-3, 103)
    ax.legend(fontsize=6.2, loc="upper right")
    fig.savefig(f"{FIG}/fig5_action_support.pdf")
    fig.savefig(f"{FIG}/fig5_action_support.png")
    plt.close(fig)
    return True


def main():
    cfg, runs = load()
    fig1_treatment_effect(runs)
    fig2_calibration(runs)
    fig3_closed_loop(runs)
    fig4_prediction_and_edge(runs)
    fig5_support()
    print("figures written to", FIG)
    for f in sorted(os.listdir(FIG)):
        print("  ", f)


if __name__ == "__main__":
    main()
