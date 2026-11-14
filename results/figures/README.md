# Generated figures

This directory collects all figures in one place. Run the scripts from the repository root after producing the corresponding result files:

```bash
python analysis/figures.py       # simulation figures 1–5
python analysis/real_figure.py   # real-data figure 6
```

Expected files:

- `fig1_treatment_effect.pdf` and `.png`
- `fig2_calibration.pdf` and `.png`
- `fig3_closed_loop.pdf` and `.png`
- `fig4_prediction_edge.pdf` and `.png`
- `fig5_action_support.pdf` and `.png` (requires `results/support.json`)
- `fig6_real_data.png` (requires `results/real_results.json`)

The images attached in chat are not available to the repository file tools as original binary files. The scripts above regenerate these plots from the experiment outputs.
