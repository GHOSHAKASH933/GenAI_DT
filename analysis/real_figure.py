"""Figure for the real-data section, drawn from real_results.json."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(ROOT / "analysis"))
from analyse import boot_ci

R = json.load(open(ROOT / "results" / "real_results.json"))["runs"]
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 0.6})
INK = "#1b1b1b"

fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.1, 2.6), gridspec_kw=dict(width_ratios=[1.25, 1]))

# (a) learned insulin effect
keys = [("raw_association", "Raw\nrecords"), ("ridge_ar", "Ridge AR"),
        ("mlp", "MLP"), ("cvae", "CVAE twin"), ("ar_twin", "AR-Twin")]
m = [np.mean([r["effect"][k] for r in R]) for k, _ in keys]
ci = [boot_ci([r["effect"][k] for r in R]) for k, _ in keys]
col = ["#7a7a7a", "#7a7a7a", "#7a7a7a", "#b3242b", "#1f6f8b"]
x = np.arange(len(keys))
err = np.clip(np.array([[mi - c[0] for mi, c in zip(m, ci)], [c[1] - mi for mi, c in zip(m, ci)]]), 0, None)
a1.bar(x, m, 0.62, color=col, edgecolor=INK, lw=0.4, yerr=err,
       error_kw=dict(ecolor=INK, elinewidth=0.6, capsize=2))
a1.axhline(0, color=INK, lw=0.7)
a1.set_xticks(x); a1.set_xticklabels([l for _, l in keys], fontsize=7)
a1.set_ylabel("Glucose fall at 2 h per IU (mg/dL)")
a1.set_title("(a) Insulin effect learned from real records", loc="left", fontsize=8.5)
a1.text(0.98, 0.04, "above 0: insulin lowers glucose\nbelow 0: wrong direction",
        transform=a1.transAxes, ha="right", va="bottom", fontsize=6.3, color=INK)

# (b) conformal coverage by diabetes type
groups = [("conf_cov", "All"), ("conf_cov_T1DM", "T1DM"), ("conf_cov_T2DM", "T2DM")]
w = 0.36
for j, (mk, lab, c) in enumerate((("cvae", "CVAE twin", "#b3242b"), ("ar_twin", "AR-Twin", "#1f6f8b"))):
    vals = [np.nanmean([r["calibration"][mk][g] for r in R]) for g, _ in groups]
    cis = [boot_ci([r["calibration"][mk][g] for r in R if np.isfinite(r["calibration"][mk][g])]) for g, _ in groups]
    e = np.clip(np.array([[v - c_[0] for v, c_ in zip(vals, cis)], [c_[1] - v for v, c_ in zip(vals, cis)]]), 0, None)
    a2.bar(np.arange(3) + (j - 0.5) * w, vals, w, color=c, edgecolor=INK, lw=0.4, label=lab,
           yerr=e, error_kw=dict(ecolor=INK, elinewidth=0.6, capsize=2))
nom = 1 - json.load(open(ROOT / "results" / "real_results.json"))["config"]["alpha"]
a2.axhline(nom, color=INK, lw=0.7, ls=(0, (4, 2)))
a2.set_xticks(range(3)); a2.set_xticklabels([l for _, l in groups])
a2.set_ylim(0, 1.18); a2.set_ylabel("Coverage after conformal calibration")
a2.set_title("(b) Coverage of the 2-h minimum, test patients", loc="left", fontsize=8.5)
a2.legend(fontsize=6.5, frameon=False, loc="upper center", ncol=2)
fig.tight_layout()
fig.savefig(ROOT / "results" / "figures" / "fig6_real_data.png", dpi=300, bbox_inches="tight")
print("ok")
