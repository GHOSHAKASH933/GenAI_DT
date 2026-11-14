"""
In-silico virtual patient cohort for glycaemic digital twins.

Bergman minimal model extended with:
  - two-compartment oral glucose absorption (meal Ra)
  - two-compartment subcutaneous insulin kinetics

This is a SIMPLIFIED testbed. It is not the UVA/Padova simulator and has no
regulatory acceptance. It is used here to generate controlled heterogeneity
across federated sites, not to make clinical claims.

State:
    G   plasma glucose            [mg/dL]
    X   remote insulin action     [1/min]
    I   plasma insulin deviation  [mU/L]
    S1,S2  subcutaneous insulin compartments [mU]
    D1,D2  gut glucose compartments          [mg]
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, asdict

DT = 1.0          # integration step [min]
CGM_PERIOD = 5    # CGM sampling period [min]


@dataclass
class PatientParams:
    p1: float      # glucose effectiveness [1/min]
    p2: float      # insulin action decay [1/min]
    p3: float      # insulin sensitivity scale [1/(mU/L)/min^2]
    n: float       # plasma insulin clearance [1/min]
    Vg: float      # glucose distribution volume [dL]
    Vi: float      # insulin distribution volume [L]
    tau_i: float   # SC insulin absorption time constant [min]
    tau_m: float   # meal absorption time constant [min]
    Ag: float      # carb bioavailability [-]
    Gb: float      # basal glucose [mg/dL]
    Ib: float      # basal insulin [mU/L]
    basal_rate: float   # basal insulin infusion [mU/min]
    CR: float      # carb ratio [g carb per U insulin]
    CF: float      # correction factor [mg/dL per U]
    weight: float  # [kg]
    true_s_ins: float = 0.0    # measured 2-h glucose fall per unit insulin [mg/dL/U]
    true_s_carb: float = 0.0   # measured 2-h glucose rise per gram carb [mg/dL/g]

    def as_dict(self):
        return asdict(self)


# Site archetypes: each site draws patients from a different physiological regime.
# This produces genuine non-IID federation (covariate + concept shift), not just
# unequal sample counts.
SITE_ARCHETYPES = {
    "A_insulin_sensitive": dict(p3_mult=1.45, tau_m_mult=0.85, Gb_shift=-8.0, CR_mult=1.30),
    "B_insulin_resistant": dict(p3_mult=0.60, tau_m_mult=1.15, Gb_shift=+14.0, CR_mult=0.72),
    "C_slow_absorber":     dict(p3_mult=1.00, tau_m_mult=1.70, Gb_shift=+2.0,  CR_mult=1.00),
    "D_fast_absorber":     dict(p3_mult=1.05, tau_m_mult=0.60, Gb_shift=-2.0,  CR_mult=1.05),
    "E_high_clearance":    dict(p3_mult=0.85, tau_m_mult=1.00, Gb_shift=+6.0,  CR_mult=0.88),
    "F_brittle":           dict(p3_mult=1.30, tau_m_mult=0.75, Gb_shift=+4.0,  CR_mult=1.20),
}


def sample_patient(rng: np.random.Generator, site: str) -> PatientParams:
    arch = SITE_ARCHETYPES[site]
    weight = float(np.clip(rng.normal(74.0, 11.0), 48.0, 110.0))
    return PatientParams(
        p1=float(np.clip(rng.normal(0.028, 0.006), 0.012, 0.050)),
        p2=float(np.clip(rng.normal(0.025, 0.005), 0.012, 0.045)),
        p3=float(np.clip(rng.normal(1.3e-5, 3.0e-6), 4e-6, 2.6e-5) * 4.5 * arch["p3_mult"]),
        n=float(np.clip(rng.normal(0.14, 0.025), 0.07, 0.24)),
        Vg=float(1.6 * weight),
        Vi=float(0.12 * weight),
        tau_i=float(np.clip(rng.normal(25.0, 4.0), 15.0, 38.0)),
        tau_m=float(np.clip(rng.normal(40.0, 8.0), 18.0, 75.0) * arch["tau_m_mult"]),
        Ag=float(np.clip(rng.normal(0.85, 0.07), 0.60, 1.00)),
        Gb=float(np.clip(rng.normal(118.0, 11.0) + arch["Gb_shift"], 85.0, 165.0)),
        Ib=float(np.clip(rng.normal(12.0, 2.5), 5.0, 22.0)),
        basal_rate=float(np.clip(rng.normal(14.0, 3.0), 6.0, 26.0)),
        CR=float(np.clip(rng.normal(11.0, 2.2), 5.0, 20.0) * arch["CR_mult"]),
        CF=float(np.clip(rng.normal(45.0, 9.0), 22.0, 75.0)),
        weight=weight,
    )


def _meal_schedule(rng: np.random.Generator, n_days: int):
    """Return (time_min, carbs_g) for meals, with realistic timing jitter."""
    meals = []
    for d in range(n_days):
        base = d * 1440
        for hour, mu, sd in ((7.5, 45, 12), (12.5, 65, 18), (19.0, 70, 20)):
            t = base + int((hour + rng.normal(0, 0.6)) * 60)
            carbs = float(np.clip(rng.normal(mu, sd), 10, 130))
            meals.append((t, carbs))
        # occasional snack
        if rng.random() < 0.45:
            t = base + int((15.5 + rng.normal(0, 0.8)) * 60)
            meals.append((t, float(np.clip(rng.normal(20, 7), 5, 45))))
    meals.sort()
    return meals


def simulate_patient(
    params: PatientParams,
    n_days: int,
    rng: np.random.Generator,
    cgm_noise_sd: float = 4.0,
    missed_bolus_prob: float = 0.10,
    bolus_error_sd: float = 0.18,
):
    """
    Open-loop simulation with standard bolus therapy (basal + carb-counting bolus).
    Produces the observational record a digital twin would be trained on.

    Returns dict of arrays sampled at CGM_PERIOD:
        glucose [mg/dL], insulin_delivered [U per CGM step], carbs [g per CGM step]
    """
    T = n_days * 1440
    meals = _meal_schedule(rng, n_days)
    meal_arr = np.zeros(T)
    bolus_arr = np.zeros(T)   # U delivered at minute t

    for t, carbs in meals:
        if t >= T:
            continue
        meal_arr[t] += carbs
        if rng.random() > missed_bolus_prob:
            dose = carbs / params.CR
            dose *= float(np.clip(rng.normal(1.0, bolus_error_sd), 0.4, 1.6))
            # bolus given somewhere in [-10, +25] min around the meal
            bt = int(np.clip(t + rng.normal(5, 10), 0, T - 1))
            bolus_arr[bt] += dose

    G = params.Gb
    X = 0.0
    I = 0.0          # deviation from Ib
    S1 = S2 = 0.0
    D1 = D2 = 0.0

    g_trace = np.empty(T)
    for t in range(T):
        # --- inputs ---
        u_sc = params.basal_rate + bolus_arr[t] * 1000.0 / DT   # mU/min
        D1 += (meal_arr[t] * 1000.0 * params.Ag / DT - D1 / params.tau_m) * DT
        D2 += (D1 / params.tau_m - D2 / params.tau_m) * DT
        Ra = D2 / params.tau_m                                   # mg/min

        S1 += (u_sc - S1 / params.tau_i) * DT
        S2 += (S1 / params.tau_i - S2 / params.tau_i) * DT
        u_plasma = S2 / params.tau_i                             # mU/min

        # --- dynamics ---
        dI = -params.n * I + (u_plasma - params.basal_rate) / params.Vi
        dX = -params.p2 * X + params.p3 * I
        dG = -(params.p1 + X) * G + params.p1 * params.Gb + Ra / params.Vg

        I += dI * DT
        X += dX * DT
        G += dG * DT
        G = float(np.clip(G, 25.0, 600.0))
        g_trace[t] = G

    idx = np.arange(0, T, CGM_PERIOD)
    cgm = g_trace[idx] + rng.normal(0.0, cgm_noise_sd, size=idx.shape)
    cgm = np.clip(cgm, 30.0, 600.0)

    ins = bolus_arr.reshape(-1, CGM_PERIOD).sum(axis=1)
    carb = meal_arr.reshape(-1, CGM_PERIOD).sum(axis=1)

    return dict(glucose=cgm.astype(np.float32),
                insulin=ins.astype(np.float32),
                carbs=carb.astype(np.float32),
                truth=g_trace[idx].astype(np.float32))


def measure_sensitivity(params: PatientParams, probe_step: int = 200):
    """
    Measure a patient's TRUE 2-hour responses by direct intervention on the
    simulator: glucose fall per unit of insulin, and glucose rise per gram of
    carbohydrate. Identical seeds and meal schedule across probes, so the
    difference is purely the probed input.

    Therapy parameters are then derived from these measurements, which keeps the
    patient's carb ratio and correction factor consistent with their own
    physiology. Sampling them independently (as in a first pass of this work)
    produces a cohort whose bolus calculator is systematically mis-specified.
    """
    def probe(dose_u=0.0, carbs_g=0.0):
        def ctrl(hg, hi, hc, s):
            return (dose_u if s == probe_step else 0.0), False
        extra_meal = (probe_step * CGM_PERIOD, carbs_g) if carbs_g > 0 else None
        return _probe_trace(params, ctrl, extra_meal, probe_step)

    base = probe()
    ins = probe(dose_u=1.0)
    carb = probe(carbs_g=20.0)
    h = 24  # 2 h in CGM steps
    s_ins = float(base[probe_step + h] - ins[probe_step + h])          # mg/dL per U
    s_carb = float(carb[probe_step + h] - base[probe_step + h]) / 20.0  # mg/dL per g
    return max(s_ins, 1.0), max(s_carb, 0.1)


def _probe_trace(params, controller, extra_meal, probe_step, n_days=3):
    """Deterministic 3-day run used only for sensitivity probing."""
    rng = np.random.default_rng(42)
    T = n_days * 1440
    n_steps = T // CGM_PERIOD
    meals = _meal_schedule(rng, n_days)
    meal_arr = np.zeros(T)
    bolus_arr = np.zeros(T)
    for t, c in meals:
        if t < T:
            meal_arr[t] += c
            dose = c / params.CR
            bt = int(np.clip(t + 5, 0, T - 1))
            bolus_arr[bt] += dose
    if extra_meal is not None:
        tm, cg = extra_meal
        if tm < T:
            meal_arr[tm] += cg

    G, X, I = params.Gb, 0.0, 0.0
    S1 = S2 = D1 = D2 = 0.0
    out = np.zeros(n_steps)
    cgm = np.zeros(n_steps, np.float32)
    ins_step = np.zeros(n_steps, np.float32)
    carb_step = np.zeros(n_steps, np.float32)
    for s in range(n_steps):
        t0 = s * CGM_PERIOD
        extra, _ = controller(cgm[:s], ins_step[:s], carb_step[:s], s)
        bolus_arr[t0] += float(extra)
        for k in range(CGM_PERIOD):
            t = t0 + k
            u_sc = params.basal_rate + bolus_arr[t] * 1000.0 / DT
            D1 += (meal_arr[t] * 1000.0 * params.Ag / DT - D1 / params.tau_m) * DT
            D2 += (D1 / params.tau_m - D2 / params.tau_m) * DT
            Ra = D2 / params.tau_m
            S1 += (u_sc - S1 / params.tau_i) * DT
            S2 += (S1 / params.tau_i - S2 / params.tau_i) * DT
            u_plasma = S2 / params.tau_i
            I += (-params.n * I + (u_plasma - params.basal_rate) / params.Vi) * DT
            X += (-params.p2 * X + params.p3 * I) * DT
            G += (-(params.p1 + X) * G + params.p1 * params.Gb + Ra / params.Vg) * DT
            G = float(np.clip(G, 25.0, 600.0))
        out[s] = G
        cgm[s] = G
        ins_step[s] = bolus_arr[t0:t0 + CGM_PERIOD].sum()
        carb_step[s] = meal_arr[t0:t0 + CGM_PERIOD].sum()
    return out


def build_cohort(seed: int, patients_per_site: int = 12, n_days: int = 10,
                 calibrate_therapy: bool = True):
    """Generate the full federated cohort for one seed."""
    rng = np.random.default_rng(seed)
    cohort = []
    for site in SITE_ARCHETYPES:
        for k in range(patients_per_site):
            p = sample_patient(rng, site)
            if calibrate_therapy:
                s_ins, s_carb = measure_sensitivity(p)
                # therapy parameters derived from the patient's own physiology,
                # with clinically realistic prescribing error retained
                err_cf = float(np.clip(rng.normal(1.0, 0.12), 0.7, 1.3))
                err_cr = float(np.clip(rng.normal(1.0, 0.12), 0.7, 1.3))
                p.CF = float(np.clip(s_ins * err_cf, 10.0, 140.0))
                p.CR = float(np.clip((s_ins / s_carb) * err_cr, 3.0, 40.0))
                p.true_s_ins = s_ins
                p.true_s_carb = s_carb
            rec = simulate_patient(p, n_days, rng)
            rec["site"] = site
            rec["params"] = p
            rec["pid"] = f"{site}:{k:02d}"
            cohort.append(rec)
    return cohort


def simulate_closed_loop(
    params: PatientParams,
    n_days: int,
    rng: np.random.Generator,
    controller,
    cgm_noise_sd: float = 4.0,
    missed_bolus_prob: float = 0.10,
    bolus_error_sd: float = 0.18,
    context_len: int = 24,
    exercise: bool = False,
    exercise_per_day: float = 1.5,
    decision_period: int = 6,
):
    """
    Same physiology, but a controller may issue an additional correction bolus at
    each CGM step. `controller(hist_g, hist_i, hist_c, step) -> units` where the
    history arrays are the realised CGM record so far (noisy, as a device sees it).

    `exercise=True` injects aerobic-exercise bouts, which transiently raise insulin
    sensitivity and non-insulin-dependent glucose uptake. Training records are
    generated WITHOUT exercise, so this is a genuine deployment-time distribution
    shift: the twin has never seen this physiology.

    Returns the CGM record plus the controller's dosing trace.
    """
    T = n_days * 1440
    n_steps = T // CGM_PERIOD
    meals = _meal_schedule(rng, n_days)
    meal_arr = np.zeros(T)
    bolus_arr = np.zeros(T)

    # exercise: multiplier on insulin action and an additive glucose uptake term
    ex_sens = np.ones(T)
    ex_uptake = np.zeros(T)
    if exercise:
        n_bouts = rng.poisson(exercise_per_day * n_days)
        for _ in range(int(n_bouts)):
            start = int(rng.uniform(0, max(T - 120, 1)))
            dur = int(np.clip(rng.normal(45, 15), 20, 90))
            intensity = float(np.clip(rng.normal(1.0, 0.3), 0.4, 1.8))
            end = min(start + dur, T)
            ex_sens[start:end] *= 1.0 + 1.4 * intensity
            ex_uptake[start:end] += 2.5 * intensity
            # post-exercise sensitivity tail
            tail = min(end + 180, T)
            ex_sens[end:tail] *= 1.0 + 0.5 * intensity

    for t, carbs in meals:
        if t >= T:
            continue
        meal_arr[t] += carbs
        if rng.random() > missed_bolus_prob:
            dose = carbs / params.CR
            dose *= float(np.clip(rng.normal(1.0, bolus_error_sd), 0.4, 1.6))
            bt = int(np.clip(t + rng.normal(5, 10), 0, T - 1))
            bolus_arr[bt] += dose

    G, X, I = params.Gb, 0.0, 0.0
    S1 = S2 = D1 = D2 = 0.0

    cgm = np.zeros(n_steps, dtype=np.float32)
    true_g = np.zeros(n_steps, dtype=np.float32)
    ins_step = np.zeros(n_steps, dtype=np.float32)
    carb_step = np.zeros(n_steps, dtype=np.float32)
    corr_step = np.zeros(n_steps, dtype=np.float32)
    abstained = np.zeros(n_steps, dtype=bool)

    for s in range(n_steps):
        t0 = s * CGM_PERIOD
        # controller acts on history available at the start of this step
        extra = 0.0
        if s >= context_len and s % decision_period == 0:
            extra, did_abstain = controller(
                cgm[:s], ins_step[:s], carb_step[:s], s
            )
            extra = float(np.clip(extra, 0.0, 8.0))
            abstained[s] = did_abstain
        corr_step[s] = extra
        bolus_arr[t0] += extra

        for k in range(CGM_PERIOD):
            t = t0 + k
            u_sc = params.basal_rate + bolus_arr[t] * 1000.0 / DT
            D1 += (meal_arr[t] * 1000.0 * params.Ag / DT - D1 / params.tau_m) * DT
            D2 += (D1 / params.tau_m - D2 / params.tau_m) * DT
            Ra = D2 / params.tau_m
            S1 += (u_sc - S1 / params.tau_i) * DT
            S2 += (S1 / params.tau_i - S2 / params.tau_i) * DT
            u_plasma = S2 / params.tau_i

            I += (-params.n * I + (u_plasma - params.basal_rate) / params.Vi) * DT
            X += (-params.p2 * X + params.p3 * ex_sens[t] * I) * DT
            G += (-(params.p1 + X) * G + params.p1 * params.Gb
                  + Ra / params.Vg - ex_uptake[t]) * DT
            G = float(np.clip(G, 25.0, 600.0))

        true_g[s] = G
        cgm[s] = float(np.clip(G + rng.normal(0.0, cgm_noise_sd), 30.0, 600.0))
        ins_step[s] = bolus_arr[t0:t0 + CGM_PERIOD].sum()
        carb_step[s] = meal_arr[t0:t0 + CGM_PERIOD].sum()

    return dict(glucose=cgm, truth=true_g, insulin=ins_step,
                carbs=carb_step, correction=corr_step, abstained=abstained)


if __name__ == "__main__":
    coh = build_cohort(seed=0, patients_per_site=3, n_days=5)
    g = np.concatenate([c["glucose"] for c in coh])
    print(f"patients={len(coh)}  samples/patient={len(coh[0]['glucose'])}")
    print(f"glucose mean={g.mean():.1f}  sd={g.std():.1f}  "
          f"min={g.min():.1f}  max={g.max():.1f}")
    tir = ((g >= 70) & (g <= 180)).mean() * 100
    print(f"time-in-range(70-180)={tir:.1f}%  "
          f"below70={(g < 70).mean()*100:.1f}%  above250={(g > 250).mean()*100:.1f}%")
    for site in SITE_ARCHETYPES:
        gs = np.concatenate([c["glucose"] for c in coh if c["site"] == site])
        print(f"  {site:22s} mean={gs.mean():6.1f}  tir={((gs>=70)&(gs<=180)).mean()*100:5.1f}%")
