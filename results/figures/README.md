# Generated figures

All publication plots are collected here. From the repository root, after running the corresponding experiments:

```bash
python analysis/figures.py       # Figures 1–5 (simulation)
python analysis/real_figure.py   # Figure 6 (real data)
```

The first two simulation figures are the treatment-effect recovery and interval-calibration plots represented by the reference images supplied with the project. They are generated from the saved simulation result file rather than copied as standalone attachments.

Expected outputs:

- `fig1_treatment_effect.pdf` and `.png`
- `fig2_calibration.pdf` and `.png`
- `fig3_closed_loop.pdf` and `.png`
- `fig4_prediction_edge.pdf` and `.png`
- `fig5_action_support.pdf` and `.png` (requires `results/support.json`)
- `fig6_real_data.png` (requires `results/real_results.json`)
