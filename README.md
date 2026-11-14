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

Adding the safety features found in commercial pumps (an insulin-on-board limit and a predictive low-glucose suspend) to the ungated twin brings time below range down from 37.3% to 21.2%. That still leaves it well short of the gated twin. Bounding the dose doesn't help when the twin's idea of what insulin does is wrong.

**Real patient data (ShanghaiT1DM / ShanghaiT2DM, 112 patients)**

- Forecasting: ridge regression (27.7 mg/dL RMSE over two hours) and an MLP (27.1) beat both generative twins (about 33).
- The raw records associate each unit of insulin with glucose 2.4 mg/dL *higher* two hours later, because insulin is given when glucose is about to rise. Models trained on these records pick up the confounding: the MLP and the CVAE learn the wrong sign in all 8 splits. The action-residual twin, whose effect term can't go negative, gets the right sign in 6 of 8, but the effect it learns is tiny.
- A 2-unit dose passes the action-support test in only 4.7% of situations, so a gated twin would decline most dosing questions on this data.
- Pooled conformal coverage looks fine overall (0.888), but type 1 patients are covered at only 0.775. Calibrating each diabetes type separately brings them to 0.865.

<p align="center">
  <img src="results/figures/fig1_treatment_effect.png" width="85%"><br>
  <em>Share of the true insulin effect each model recovers in simulation.</em>
</p>

<p align="center">
  <img src="results/figures/fig3_closed_loop.png" width="85%"><br>
  <em>Closed-loop results without (top) and with (bottom) unseen exercise.</em>
</p>

<p align="center">
  <img src="results/figures/fig6_real_data.png" width="85%"><br>
  <em>Real records: learned insulin effect, and conformal coverage by diabetes type.</em>
</p>

## What is in the repository

```
genai_dt/                 core package
  simulator.py            virtual patients (Bergman minimal model with meal absorption
                          and subcutaneous insulin kinetics); measures each patient's
                          true insulin response by running them twice
  data.py                 windowing and patient-level splits
  models.py               CVAE twin, action-residual twin, MLP, edge student, baselines
  federated.py            FedAvg and per-site fine-tuning
  conformal.py            split-conformal gate on the two-hour glucose minimum
  intervention.py         randomised vs titrated dose exploration, action-support gate,
                          dose-response probe
  control.py              basal, PID and twin-based controllers, glycaemic metrics
  experiments.py          main simulation study (effect recovery, forecasting,
                          calibration, closed loop, edge distillation)
  support_analysis.py     which doses the support gate allows, by training data
  robustness.py           ungated twin with standard pump safeguards
realdata/
  shanghai.py             loader for the ShanghaiT1DM/T2DM Excel files
  real_experiments.py     forecasting, calibration, action support and learned
                          insulin effect on real records
analysis/
  analyse.py              statistics (Wilcoxon, Holm, bootstrap CIs, power check)
  figures.py              simulation figures
  real_figure.py          real-data figure
results/                  the result files behind the paper, and all figures
data/                     put the Shanghai dataset here (not redistributed)
```

## Setup

Python 3.10 or newer. A CPU is enough; nothing uses a GPU.

```bash
pip install -r requirements.txt
```

## Reproducing the results

Run everything from the repository root. The times below are for two CPU cores.

**1. Main simulation study** (5 to 10 minutes per seed)

```bash
python -m genai_dt.experiments --seeds 8 --out results/results.json
```

For a one-minute check that everything works:

```bash
python -m genai_dt.experiments --quick --seeds 1 --out results/quick.json
```

**2. Supplementary simulation analyses**

```bash
python -m genai_dt.support_analysis --seeds 8      # which doses the gate allows
python -m genai_dt.robustness --seeds 8            # ungated twin + pump safeguards
```

**3. Real patient data** (about 70 seconds per seed)

Download the ShanghaiT1DM and ShanghaiT2DM datasets from figshare (https://doi.org/10.6084/m9.figshare.c.6310860) and unzip them into `data/` so that you have `data/Shanghai_T1DM/` and `data/Shanghai_T2DM/`. Then:

```bash
python realdata/real_experiments.py --seeds 8
```

**4. Statistics and figures**

```bash
python analysis/analyse.py          # prints every comparison reported in the paper
python analysis/figures.py          # simulation figures -> results/figures/
python analysis/real_figure.py      # real-data figure  -> results/figures/
```

The result files used in the paper are already in `results/`, so step 4 works without rerunning anything.

## Design choices worth knowing about

- **Splits are by patient, never by time window.** No patient appears in both training and test data.
- **The true insulin effect is measured, not assumed.** `simulator.measure_sensitivity` runs each simulated patient with and without a unit dose under identical meals and noise, and therapy parameters are derived from that measurement.
- **Dose randomisation matters.** `intervention.randomised_dose_policy` can draw the dose size independently of the patient's state (identifies the effect) or scale it to current glucose (looks safer, but doesn't). Both are reported.
- **No underpowered significance tests.** With 8 seeds the smallest possible Wilcoxon p-value is 0.0078. `analysis/analyse.py` flags any comparison with too few seeds to reach significance.
- **The real-data section can't test closed-loop control.** That would mean dosing real patients. It tests forecasting, calibration, action support and the direction of the learned insulin effect.

## Limitations

The closed-loop experiments run on a simplified simulator, not the UVA/Padova model, and say nothing about clinical safety. The Shanghai records come from one city, include only 12 type 1 patients, record meals without carbohydrate amounts, and have no ground truth for the insulin effect. Federated averaging here has no formal privacy guarantee. The paper discusses each of these.

## Citation

If you use this code, please cite the paper and the dataset:

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

## Development note

Parts of the code and the manuscript were developed with the help of an AI assistant (Claude, Anthropic). All reported results come from running the code in this repository.

## Contact

Akash Ghosh, Centre for Artificial Intelligence and Machine Learning, Indian Statistical Institute, Kolkata.

