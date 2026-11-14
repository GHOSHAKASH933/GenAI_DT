"""Real-data figures, read from results/real_results.json."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.analyse import boot_ci

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "real_results.json"
FIG_DIR = ROOT / "results" / "figures"
INK = "#1b1b1b"


def make_figure(results_path: Path = RESULTS, output_path: Path | None = None) -> Path:
    """Build the real-data effect and coverage panels from saved results."""
    with results_path.open() as f:
        data = json.load(f)
    runs = data["runs"]
    if not runs:
        raise ValueError(f"No real-data experiment runs found in {results_path}")

    plt.rcParams.update({"font.family": "serif", "font.size": 8,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": 0.6})
    fig, (a1, a2) = plt.subplots(
        1, 2, figsize=(7.1, 2.6), gridspec_kw=dict(width_ratios=[1.25, 1])
    )

    keys = [("raw_association", "Raw\nrecords"), ("ridge_ar", "Ridge AR"),
            ("mlp", "MLP"), ("cvae", "CVAE twin"), ("ar_twin", "AR-Twin")]
    means = [np.mean([r["effect"][key] for r in runs]) for key, _ in keys]
    cis = [boot_ci([r["effect"][key] for r in runs]) for key, _ in keys]
    colors = ["#7a7a7a", "#7a7a7a", "#7a7a7a", "#b3242b", "#1f6f8b"]
    x = np.arange(len(keys))
    err = np.clip(np.array([[m - c[0] for m, c in zip(means, cis)],
                            [c[1] - m for m, c in zip(means, cis)]]), 0, None)
    a1.bar(x, means, 0.62, color=colors, edgecolor=INK, lw=0.4, yerr=err,
           error_kw=dict(ecolor=INK, elinewidth=0.6, capsize=2))
    a1.axhline(0, color=INK, lw=0.7)
    a1.set_xticks(x)
    a1.set_xticklabels([label for _, label in keys], fontsize=7)
    a1.set_ylabel("Glucose fall at 2 h per IU (mg/dL)")
    a1.set_title("(a) Insulin effect learned from real records", loc="left", fontsize=8.5)
    a1.text(0.98, 0.04, "above 0: insulin lowers glucose\nbelow 0: wrong direction",
            transform=a1.transAxes, ha="right", va="bottom", fontsize=6.3, color=INK)

    groups = [("conf_cov", "All"), ("conf_cov_T1DM", "T1DM"),
              ("conf_cov_T2DM", "T2DM")]
    width = 0.36
    for j, (model_key, label, color) in enumerate(
        (("cvae", "CVAE twin", "#b3242b"), ("ar_twin", "AR-Twin", "#1f6f8b"))
    ):
        vals = [np.nanmean([r["calibration"][model_key][metric] for r in runs])
                for metric, _ in groups]
        intervals = [boot_ci([r["calibration"][model_key][metric] for r in runs
                              if np.isfinite(r["calibration"][model_key][metric])])
                     for metric, _ in groups]
        errors = np.clip(np.array([[v - ci[0] for v, ci in zip(vals, intervals)],
                                   [ci[1] - v for v, ci in zip(vals, intervals)]]), 0, None)
        a2.bar(np.arange(3) + (j - 0.5) * width, vals, width, color=color,
               edgecolor=INK, lw=0.4, label=label, yerr=errors,
               error_kw=dict(ecolor=INK, elinewidth=0.6, capsize=2))
    nominal = 1 - data["config"]["alpha"]
    a2.axhline(nominal, color=INK, lw=0.7, ls=(0, (4, 2)))
    a2.set_xticks(range(3))
    a2.set_xticklabels([label for _, label in groups])
    a2.set_ylim(0, 1.18)
    a2.set_ylabel("Coverage after conformal calibration")
    a2.set_title("(b) Coverage of the 2-h minimum, test patients", loc="left", fontsize=8.5)
    a2.legend(fontsize=6.5, frameon=False, loc="upper center", ncol=2)
    fig.tight_layout()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    dest = output_path or FIG_DIR / "fig6_real_data.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return dest


def main() -> None:
    if not RESULTS.is_file():
        raise SystemExit(f"Missing {RESULTS}. Run the real-data experiments first.")
    print("Wrote", make_figure())


if __name__ == "__main__":
    main()
