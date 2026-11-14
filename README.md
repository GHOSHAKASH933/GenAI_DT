# When a Digital Twin Should Decline to Answer

Research code accompanying *When a Digital Twin Should Decline to Answer: Treatment-Effect Validity and Calibrated Abstention in Generative Digital Twins for Closed-Loop Glycaemic Control* by Akash Ghosh and Abhraneel Dalui.

## Overview

Generative models can produce plausible patient futures, but a plausible forecast does not by itself establish that a model can answer an intervention question. This project tests that distinction using insulin dosing in a controlled in-silico glucose simulator with known patient-level treatment effects. It evaluates treatment-effect recovery, forecasting, conformal calibration, action support, closed-loop control, federated training, and edge distillation. Observational ShanghaiT1DM/T2DM records are used for forecasting, calibration, action-support, and learned-effect analyses; no closed-loop intervention is performed on real patients.

The central methodological point is that predictive calibration and treatment-effect validity are different properties. The repository evaluates whether a twin's action-response is supported by its training data and includes abstention mechanisms for unsupported queries. All clinical-sounding outcomes are simulator results, not evidence of clinical efficacy or safety.

## Repository structure

```text
genai_dt/                 simulator, models, training, gates, controllers, experiments
analysis/                 statistical analysis and simulation/real-data figure scripts
realdata/                 Shanghai data loader and real-data experiments
data/                     local-only location for external Shanghai files
results/                  experiment outputs and the shared figures directory
scripts/                  repository verification helper
```

The core experiment modules map to the paper as follows:

- `genai_dt/experiments.py`: simulation experiments E1–E5 (treatment effect, forecasting/federation, conformal calibration, closed loop, edge distillation).
- `genai_dt/support_analysis.py`: action-support profiles used by Figure 5.
- `genai_dt/robustness.py`: pump-safeguard robustness analysis.
- `realdata/real_experiments.py`: real-record experiments R1–R4.
- `analysis/analyse.py`, `analysis/figures.py`, `analysis/real_figure.py`: aggregation and figure generation.

## Requirements and installation

Python 3.10 or newer is required. The experiments run on CPU; a GPU is optional. From the repository root, the recommended first-time setup creates `.venv`, installs the package and dependencies in editable mode, and runs syntax/import/model smoke checks:

```bash
bash setup.sh
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
py -3.10 -m venv .venv
.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python scripts/verify_project.py
```

The dependency list is maintained in `pyproject.toml`; `requirements.txt` is provided for requirements-file workflows. Use `make help` to list common commands.

## Validation

```bash
make verify
```

Run a compact simulation to validate the experiment path and output writing. The quick configuration is only a smoke test and does not reproduce paper results:

```bash
make quick
```

## Simulation experiments

Run the full eight-seed study from the repository root:

```bash
make simulation SEEDS=8
```

Equivalent command:

```bash
python -m genai_dt.experiments --seeds 8 --out results/results.json
```

Supplementary analyses:

```bash
make support SEEDS=8       # action-support profiles
make robustness SEEDS=8    # pump-safeguard robustness
```

The study is patient-split, with training, calibration, and test patients kept separate. The seed is the statistical unit for the reported paired comparisons. Full runs are CPU-intensive; run the quick check first.

## Shanghai real-data experiments

Download the ShanghaiT1DM and ShanghaiT2DM datasets from figshare ([DOI: 10.6084/m9.figshare.c.6310860](https://doi.org/10.6084/m9.figshare.c.6310860)) and extract them to:

```text
data/Shanghai_T1DM/
data/Shanghai_T2DM/
```

The source data are not included or redistributed. Run the real-data pipeline with:

```bash
make real-data SEEDS=8
```

Or use the direct command:

```bash
python -m realdata.real_experiments --seeds 8
```

With the dataset installed, `python -m realdata.real_experiments --quick --seeds 1` provides a smaller pipeline check. The real-data analysis is observational; it does not test closed-loop dosing on patients.

## Statistics and figures

After producing the result files required by each analysis:

```bash
make stats
make figures
```

All plots are written to `results/figures/`. The simulation script generates Figures 1–5, including treatment-effect recovery (Figure 1) and interval calibration (Figure 2); the real-data script generates Figure 6. Figure 5 requires `results/support.json`, and Figure 6 requires `results/real_results.json`. See [`results/figures/README.md`](results/figures/README.md) for filenames and inputs.

The figures attached in chat are not available to the repository tools as original binary files. The repository scripts regenerate the corresponding plotted analyses from experiment results; they do not copy the chat attachments.

## Scope and limitations

The virtual-patient model is a simplified Bergman-style simulator, not the UVA/Padova simulator, and it has no regulatory acceptance. Its closed-loop results are properties of the simulation and should not be interpreted as clinical safety estimates. The Shanghai records are observational, come from one city, contain relatively few type 1 patients, record meals without carbohydrate quantities, and provide no ground truth for insulin effect. FedAvg in this implementation has no formal differential-privacy guarantee.

## Citation

```bibtex
@unpublished{ghosh2026decline,
  title  = {When a Digital Twin Should Decline to Answer: Treatment-Effect Validity
            and Calibrated Abstention in Generative Digital Twins},
  author = {Ghosh, Akash and Dalui, Abhraneel},
  note   = {Manuscript},
  year   = {2026}
}

@article{zhao2023chinese,
  title   = {Chinese diabetes datasets for data-driven machine learning},
  author  = {Zhao, Qinpei and Zhu, Jinhao and Shen, Xuan and Lin, Chuwen and Zhang, Yinjia
             and Liang, Yuxiang and Cao, Baige and Li, Jiangfeng and Liu, Xiang and Rao, Weixiong
             and Wang, Congrong},
  journal = {Scientific Data},
  volume  = {10},
  year    = {2023},
  doi     = {10.1038/s41597-023-01940-7}
}
```
