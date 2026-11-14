# GenAI_DT

Research code accompanying *When a Digital Twin Should Decline to Answer: Treatment-Effect Validity and Calibrated Abstention in Generative Digital Twins* by Akash Ghosh and Abhraneel Dalui.

This repository evaluates whether a generative digital twin trained on routine-care data can support counterfactual insulin-dosing questions. It compares predictive calibration with treatment-effect validity, and evaluates action-support abstention in a controlled simulator and in observational Shanghai diabetes records. The simulator is a research testbed, not a clinical device or a validated model of clinical safety.

## Repository contents

- `genai_dt/`: virtual-patient simulator, generative and baseline models, federated training, conformal and action-support gates, and simulation experiments.
- `realdata/`: Shanghai dataset loader and held-out-patient experiments.
- `analysis/`: statistical summaries and figure-generation scripts.
- `results/figures/`: one destination for generated figures.
- `data/`: local location for the Shanghai files; the source data are not redistributed.
- `scripts/`: lightweight syntax, import, and model-shape checks.

## Requirements and installation

Python 3.10 or newer is required. The experiments are CPU-compatible; no GPU is required. The recommended installation creates a project-local virtual environment and installs the package in editable mode:

```bash
bash setup.sh
source .venv/bin/activate
```

The setup script installs dependencies declared in `pyproject.toml` and runs the verification checks. On Windows, use PowerShell:

```powershell
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python scripts/verify_project.py
```

`requirements.txt` is also provided for environments that prefer requirements-file installs. `make help` lists the available development and experiment commands.

## Validation

Run syntax/import and model-forward smoke checks:

```bash
make verify
```

Run a compact simulation to confirm that the experiment pipeline executes and writes output. This quick configuration is for validation only and does not reproduce the reported study results:

```bash
make quick
```

## Reproducing the simulation experiments

Run all eight seeds from the repository root (the full study takes several minutes per seed, depending on hardware):

```bash
make simulation SEEDS=8
```

Equivalent direct command:

```bash
python -m genai_dt.experiments --seeds 8 --out results/results.json
```

Optional supplementary analyses:

```bash
make support SEEDS=8       # action-support profiles
make robustness SEEDS=8    # pump-safeguard robustness experiment
```

## Shanghai real-data experiment

Download the ShanghaiT1DM and ShanghaiT2DM datasets from figshare ([dataset DOI](https://doi.org/10.6084/m9.figshare.c.6310860)) and extract them into:

```text
data/Shanghai_T1DM/
data/Shanghai_T2DM/
```

The patient data are not included in this repository. Then run:

```bash
make real-data SEEDS=8
```

For a shorter pipeline check with the dataset present:

```bash
python -m realdata.real_experiments --quick --seeds 1
```

Real-data analyses are observational and do not evaluate closed-loop control on patients.

## Statistics and figures

After generating the relevant result files, run:

```bash
make stats
make figures
```

All plots are written under `results/figures/`. The simulation figure script generates Figures 1–5, including the treatment-effect recovery and interval-calibration panels shown in the supplied reference images. The real-data script generates Figure 6. Figure 5 requires `results/support.json`; Figure 6 requires `results/real_results.json`. See [`results/figures/README.md`](results/figures/README.md) for the output names and inputs.

The images attached in chat are not available to the repository tools as binary files. The corresponding plots are generated from the experiment outputs by these scripts; no image files are claimed to have been copied from the attachments.

## Scientific scope and limitations

The simulator is simplified and is not the UVA/Padova simulator. Results from in-silico experiments do not establish clinical safety. The Shanghai cohort is observational, represents one city, includes relatively few type 1 patients, records meals without carbohydrate quantities, and has no ground truth for insulin effect. Federated averaging in this code does not provide a formal privacy guarantee.

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
