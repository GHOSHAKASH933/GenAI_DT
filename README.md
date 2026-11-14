# When a digital twin should decline to answer

Code and results for the paper *When a Digital Twin Should Decline to Answer: Treatment-Effect Validity and Calibrated Abstention in Generative Digital Twins* (Akash Ghosh, Abhraneel Dalui).

Generative models are often proposed as the engine of clinical digital twins. A twin that can sample many possible futures for a patient looks like a simulator, and it gets asked simulator questions: what happens if we give this dose? This repository tests whether a generative twin can answer that question, using insulin dosing in diabetes as the example.

Short answer: it usually can't, and the standard uncertainty checks won't tell you. A conditional VAE twin trained on routine care records learns only about a seventh of the true insulin effect, yet its conformal prediction intervals are well calibrated. In closed-loop simulation that twin causes more hypoglycaemia than giving no correction doses at all. What fixes it is letting the twin refuse doses that its training data never covered.

Everything in the paper, every table, figure and number in the text, is produced by the scripts here.

## Main results

All intervals are bootstrap 95% intervals over 8 seeds.

**Simulation (true effect known by construction)**

| Model and training data | Share of the true insulin effect recovered |
|---|---|
| CVAE twin, observational records | 0.14 (0.09 to 0.17) |
| CVAE twin, randomised doses | 0.44 (0.39 to 0.50) |
| Deterministic MLP, randomised doses | 1.91 (1.80 to 2.03) |

The simulated patients lose 11.5 mg/dL of glucose per unit of insulin on average. The observational twin believes it is closer to 1.5. Its conformal intervals still reach 0.888 coverage against a 0.90 target, so calibration alone gives no warning.

| Controller (5 simulated days, matched physiology) | Time below 70 mg/dL (%) | Time in range (%) | Correction insulin (U) |
|---|---|---|---|
| No corrections | 3.9 | 92.0 | 0.0 |
| PID controller | 7.4 | 91.1 | 15.5 |
| Twin, no gate | 37.0 | 62.1 | 85.8 |
| Twin, conformal gate only | 40.0 | 59.2 | 94.4 |
| Twin, conformal + action-support gate | 6.2 | 91.6 | 8.3 |
| Action-residual twin (randomised data), both gates | 4.4 | 92.4 | 3.0 |

Adding an insulin-on-board limit and predictive low-glucose suspend to the ungated twin brings time below range down from 37.3% to 21.2%. Bounding the dose does not fix a twin whose estimate of insulin's effect is wrong.

**Real patient data (ShanghaiT1DM / ShanghaiT2DM, 112 patients)**

- Forecasting: ridge regression (27.7 mg/dL RMSE over two hours) and an MLP (27.1) beat both generative twins (about 33).
- The raw records associate each unit of insulin with glucose 2.4 mg/dL *higher* two hours later, because insulin is given when glucose is about to rise. The MLP and CVAE learn the wrong sign in all 8 splits. The action-residual twin gets the right sign in 6 of 8, but learns a tiny effect.
- A 2-unit dose passes the action-support test in only 4.7% of situations.
- Pooled conformal coverage is 0.888 overall but 0.775 for type 1 patients; calibrating by diabetes type brings type 1 coverage to 0.865.

## Repository structure

```text
genai_dt/                 core simulation, models, gates, controllers, experiments
analysis/                 statistics and simulation/real-data figure scripts
realdata/                 Shanghai data loader and real-data experiments
data/                     put the Shanghai datasets here (not redistributed)
results/figures/          all generated figure files in one folder
scripts/                  project verification helper
```

## First-time setup

Python 3.10 or newer is required. The experiments run on CPU; a GPU is not required. From the repository root on macOS/Linux:

```bash
bash setup.sh
source .venv/bin/activate
```

The setup script creates `.venv`, installs `requirements.txt`, then runs syntax, import, and model-shape smoke checks. On Windows PowerShell, create and activate the environment and install dependencies with:

```powershell
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/verify_project.py
```

## Quick validation

This small run checks that the simulation pipeline executes and writes a result file; it is not a reproduction of the paper's reported results.

```bash
python scripts/verify_project.py
python -m genai_dt.experiments --quick --seeds 1 --out results/quick.json
```

## Full experiment run

Run all eight simulation seeds (approximately 5–10 minutes per seed on two CPU cores):

```bash
python -m genai_dt.experiments --seeds 8 --out results/results.json
```

Optional supplementary analyses:

```bash
python -m genai_dt.support_analysis --seeds 8
python -m genai_dt.robustness --seeds 8
```

## Real-data run

Download the ShanghaiT1DM and ShanghaiT2DM datasets from figshare (https://doi.org/10.6084/m9.figshare.c.6310860) and extract them under `data/` as `data/Shanghai_T1DM/` and `data/Shanghai_T2DM/`. The private source data are not included in this repository. Then run:

```bash
python realdata/real_experiments.py --seeds 8
```

To smoke-test the real-data pipeline with the dataset installed, use `--quick`:

```bash
python realdata/real_experiments.py --quick --seeds 1
```

## Figures and statistics

All figure scripts write their outputs to the single directory `results/figures/`. After running the relevant experiments, generate simulation figures 1–5 and the real-data figure 6 with:

```bash
python analysis/analyse.py
python analysis/figures.py
python analysis/real_figure.py
```

`analysis/figures.py` generates the treatment-effect, calibration, closed-loop, prediction/edge, and action-support plots. `analysis/real_figure.py` generates the real-data plot. The action-support plot requires `results/support.json`; the real-data plot requires `results/real_results.json`. See `results/figures/README.md` for the expected outputs. Figure images supplied as chat attachments are not automatically copied into GitHub; these commands regenerate the repository figures from the experiment results.

## Design choices and limitations

- Splits are by patient, never by time window.
- The simulator measures each virtual patient's insulin effect by running paired probes with identical meals and noise.
- Dose randomisation identifies an action effect in the simulated setting; state-dependent titration gives poorer overlap.
- The Shanghai data are observational, come from one city, include only 12 type 1 patients, record meals without carbohydrate amounts, and have no ground truth for insulin effect. No closed-loop control is evaluated on real patients.
- The closed-loop tests use a simplified simulator, not the UVA/Padova model, and do not establish clinical safety. Federated averaging has no formal privacy guarantee.

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

## Contact

Akash Ghosh, Centre for Artificial Intelligence and Machine Learning, Indian Statistical Institute, Kolkata.
