"""Simulation experiment driver for paper experiments E1–E5."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

import numpy as np
import torch

from . import data as D
from .simulator import build_cohort, simulate_closed_loop
from .models import (
    CVAETwin, ActionResidualTwin, PointMLP, EdgeStudent,
    train_cvae, train_point, Persistence, RidgeAR,
)
from .federated import train_local_only, train_fedavg, personalise
from .conformal import ConformalGate
from .control import basal_only, pid_controller, twin_controller, glycaemic_metrics
from .intervention import build_interventional_cohort, dose_response_slope, ActionSupportGate

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(OUT, exist_ok=True)


def _stable_seed(*parts) -> int:
    """Stable cross-process seed; unlike hash(), this is independent of PYTHONHASHSEED."""
    payload = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=4).digest(), "big")


def _mean_pred(model, X, n_samples=48, batch=1024):
    out = []
    Xt = torch.as_tensor(X)
    with torch.no_grad():
        for b in range(0, len(Xt), batch):
            out.append(model.sample(Xt[b:b + batch], n_samples=n_samples,
                                    include_obs_noise=False).mean(0).numpy())
    return np.concatenate(out) if out else np.zeros((0, D.HORIZON), np.float32)


def rmse_mae(pred_norm, Y_norm):
    p, y = D.denorm(pred_norm), D.denorm(Y_norm)
    e = p - y
    return float(np.sqrt((e ** 2).mean())), float(np.abs(e).mean())


# ----------------------------------------------------------------- E1

def e1_treatment_effect(datasets, Xte, true_s, seed, cfg):
    """Compare dose-response recovery across training regimes and model types."""
    res = {}
    probe = Xte[:cfg["probe_n"]]
    for dname, (X, Y) in datasets.items():
        for mname, cls in (("cvae", CVAETwin), ("ar_twin", ActionResidualTwin)):
            torch.manual_seed(seed)
            model = train_cvae(cls(), X, Y, epochs=cfg["epochs"], seed=seed)
            slopes = dose_response_slope(model, probe)
            res[f"{dname}|{mname}"] = dict(
                slope_mean=float(slopes.mean()), slope_sd=float(slopes.std()),
                ratio=float(slopes.mean() / true_s),
            )
        torch.manual_seed(seed)
        mlp = train_point(PointMLP(), X, Y, epochs=cfg["epochs"], seed=seed)
        lo = D.CONTEXT * 3
        ends = []
        for dose in (0.0, 1.0, 2.0):
            Xd = probe.copy()
            Xd[:, lo] = dose / D.I_SCALE
            with torch.no_grad():
                ends.append(D.denorm(mlp(torch.as_tensor(Xd)).numpy())[:, -1].mean())
        slope = -(ends[-1] - ends[0]) / 2.0
        res[f"{dname}|point_mlp"] = dict(
            slope_mean=float(slope), slope_sd=float("nan"), ratio=float(slope / true_s)
        )
    res["_true_sensitivity"] = float(true_s)
    return res


# ----------------------------------------------------------------- E2

def e2_prediction(train, test, seed, cfg):
    Xtr, Ytr, _ = D.pack(train)
    Xte, Yte, _ = D.pack(test)
    site_data = {site: D.pack(records)[:2] for site, records in D.by_site(train).items()}
    test_by_site = {site: D.pack(records)[:2] for site, records in D.by_site(test).items()}
    res = {}

    res["persistence"] = rmse_mae(Persistence().predict(Xte), Yte)
    res["ridge_ar"] = rmse_mae(RidgeAR(lam=5.0).fit(Xtr, Ytr).predict(Xte), Yte)
    torch.manual_seed(seed)
    pm = train_point(PointMLP(), Xtr, Ytr, epochs=cfg["epochs"], seed=seed)
    with torch.no_grad():
        res["point_mlp_central"] = rmse_mae(pm(torch.as_tensor(Xte)).numpy(), Yte)

    torch.manual_seed(seed)
    central = train_cvae(ActionResidualTwin(), Xtr, Ytr, epochs=cfg["epochs"], seed=seed)
    res["twin_centralised"] = rmse_mae(_mean_pred(central, Xte), Yte)

    local = train_local_only(site_data, seed=seed, epochs=cfg["epochs"], cls=ActionResidualTwin)
    errs = [(D.denorm(_mean_pred(local[s], Xs)) - D.denorm(Ys)) ** 2
            for s, (Xs, Ys) in test_by_site.items() if len(Xs)]
    e = np.concatenate(errs)
    res["twin_local_only"] = (float(np.sqrt(e.mean())), float(np.sqrt(e).mean()))

    glob = train_fedavg(site_data, seed=seed, rounds=cfg["rounds"],
                        local_epochs=cfg["local_epochs"], cls=ActionResidualTwin)
    res["twin_fedavg"] = rmse_mae(_mean_pred(glob, Xte), Yte)

    personalised = personalise(glob, site_data, seed=seed, epochs=cfg["pers_epochs"],
                               cls=ActionResidualTwin)
    errs = [(D.denorm(_mean_pred(personalised[s], Xs)) - D.denorm(Ys)) ** 2
            for s, (Xs, Ys) in test_by_site.items() if len(Xs)]
    e = np.concatenate(errs)
    res["twin_fedavg_personalised"] = (float(np.sqrt(e.mean())), float(np.sqrt(e).mean()))
    return res, glob, personalised


# ----------------------------------------------------------------- E3/E4

def e3_e4(obs_twin, int_twin, Xtr_obs, Xtr_int, calib, test, seed, cfg):
    Xc, Yc, _ = D.pack(calib)
    Xte, Yte, _ = D.pack(test)

    cal, gates = {}, {}
    for label, model in (("obs", obs_twin), ("int", int_twin)):
        gate = ConformalGate(alpha=cfg["alpha"]).fit(model, Xc, Yc)
        coverage, width = gate.coverage(model, Xte, Yte)
        from .conformal import twin_quantiles
        q_lo, _, q_hi = twin_quantiles(model, Xte, alpha=cfg["alpha"])
        y_true = D.denorm(Yte).min(axis=1)
        cal[label] = dict(
            nominal=1 - cfg["alpha"],
            raw_coverage=float(((y_true >= q_lo) & (y_true <= q_hi)).mean()),
            raw_width=float(np.mean(q_hi - q_lo)), conformal_coverage=coverage,
            conformal_width=width, Q=gate.Q, width_tau=gate.width_tau,
            n_calib=gate.n_calib,
        )
        gates[label] = gate

    support = {
        "obs": ActionSupportGate(k=cfg["knn_k"], min_support=cfg["min_support"]).fit(Xtr_obs),
        "int": ActionSupportGate(k=cfg["knn_k"], min_support=cfg["min_support"]).fit(Xtr_int),
    }
    arms = {
        "basal_only": ("none", None, None, None),
        "pid": ("pid", None, None, None),
        "twin_obs_ungated": ("twin", obs_twin, None, None),
        "twin_obs_conformal": ("twin", obs_twin, gates["obs"], None),
        "twin_obs_dual": ("twin", obs_twin, gates["obs"], support["obs"]),
        "twin_int_dual": ("twin", int_twin, gates["int"], support["int"]),
    }

    chosen = test[:min(cfg["loop_patients"], len(test))]
    regimes = {}
    for regime, use_exercise in (("matched", False), ("shifted", True)):
        out = {}
        for name, (kind, model, gate, sup) in arms.items():
            per_patient = []
            for rec in chosen:
                rng = np.random.default_rng(_stable_seed(seed, rec["pid"], regime))
                params = rec["params"]
                if kind == "none":
                    controller = basal_only(params)
                elif kind == "pid":
                    controller = pid_controller(params)
                else:
                    controller = twin_controller(model, params, gate, support=sup,
                                                 n_samples=cfg["loop_samples"])
                trace = simulate_closed_loop(
                    params, cfg["loop_days"], rng, controller,
                    exercise=use_exercise, decision_period=6,
                )
                metrics = glycaemic_metrics(trace)
                metrics["pid_"] = rec["pid"]
                per_patient.append(metrics)
            out[name] = per_patient
        regimes[regime] = out
    return cal, regimes


# ----------------------------------------------------------------- E5

def e5_edge(twin, train, test, seed, cfg):
    Xtr, _, _ = D.pack(train)
    Xte, Yte, _ = D.pack(test)
    if not len(Xtr) or not len(Xte):
        raise ValueError("Edge distillation requires non-empty train and test windows.")
    alpha = cfg["alpha"]
    Xt = torch.as_tensor(Xtr)
    los, mids, his = [], [], []
    with torch.no_grad():
        for b in range(0, len(Xt), 1024):
            samples = twin.sample(Xt[b:b + 1024], n_samples=48)
            los.append(torch.quantile(samples, alpha / 2, dim=0))
            mids.append(torch.quantile(samples, 0.5, dim=0))
            his.append(torch.quantile(samples, 1 - alpha / 2, dim=0))
    t_lo, t_mid, t_hi = torch.cat(los), torch.cat(mids), torch.cat(his)

    torch.manual_seed(seed)
    student = EdgeStudent()
    optimizer = torch.optim.AdamW(student.parameters(), lr=3e-3, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    for _ in range(cfg["epochs"]):
        perm = torch.randperm(len(Xt), generator=generator)
        for b in range(0, len(Xt), 256):
            idx = perm[b:b + 256]
            lo, mid, hi = student(Xt[idx])
            loss = ((lo - t_lo[idx]) ** 2 + (mid - t_mid[idx]) ** 2
                    + (hi - t_hi[idx]) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    student.eval()

    with torch.no_grad():
        _, mid, _ = student(torch.as_tensor(Xte))
    student_rmse, student_mae = rmse_mae(mid.numpy(), Yte)
    teacher_rmse, teacher_mae = rmse_mae(_mean_pred(twin, Xte), Yte)

    def latency(fn, reps=150):
        x1 = torch.as_tensor(Xte[:1])
        fn(x1)
        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn(x1)
            times.append((time.perf_counter() - t0) * 1000)
        return float(np.median(times)), float(np.percentile(times, 95))

    with torch.no_grad():
        teacher_ms, teacher_p95 = latency(lambda x: twin.sample(x, n_samples=cfg["loop_samples"]))
        student_ms, student_p95 = latency(lambda x: student(x))
    n_params = lambda model: sum(p.numel() for p in model.parameters())
    return dict(
        teacher_rmse=teacher_rmse, teacher_mae=teacher_mae,
        student_rmse=student_rmse, student_mae=student_mae,
        teacher_latency_ms=teacher_ms, teacher_latency_p95=teacher_p95,
        student_latency_ms=student_ms, student_latency_p95=student_p95,
        teacher_params=n_params(twin), student_params=n_params(student),
    )


# ----------------------------------------------------------------- driver

def run_seed(seed, cfg):
    t0 = time.time()
    cohort = build_cohort(seed, patients_per_site=cfg["patients_per_site"], n_days=cfg["n_days"])
    train, calib, test = D.split_by_patient(cohort, seed)
    if not train or not calib or not test:
        raise ValueError("Patient-level split produced an empty partition; increase cohort size.")
    true_s = float(np.mean([rec["params"].true_s_ins for rec in test]))

    Xo, Yo, _ = D.pack(train)
    int_rand = build_interventional_cohort(train, cfg["n_days"], seed, state_independent=True)
    int_tit = build_interventional_cohort(train, cfg["n_days"], seed, state_independent=False)
    Xi, Yi, _ = D.pack(int_rand)
    Xt_, Yt_, _ = D.pack(int_tit)
    Xte, _, _ = D.pack(test)

    datasets = {
        "observational": (Xo, Yo),
        "interventional_randomised": (Xi, Yi),
        "interventional_titrated": (Xt_, Yt_),
    }
    e1 = e1_treatment_effect(datasets, Xte, true_s, seed, cfg)
    e2, _, _ = e2_prediction(train, test, seed, cfg)

    torch.manual_seed(seed)
    obs_twin = train_cvae(CVAETwin(), Xo, Yo, epochs=cfg["epochs"], seed=seed)
    torch.manual_seed(seed)
    int_twin = train_cvae(ActionResidualTwin(), Xi, Yi, epochs=cfg["epochs"], seed=seed)

    calibration, loop = e3_e4(obs_twin, int_twin, Xo, Xi, calib, test, seed, cfg)
    e5 = e5_edge(int_twin, int_rand, test, seed, cfg)
    print(f"  seed {seed} completed in {time.time() - t0:.0f}s", flush=True)
    return dict(
        seed=seed, true_sensitivity=true_s, e1=e1, e2=e2, calibration=calibration,
        loop=loop, e5=e5, n_train=len(train), n_calib=len(calib), n_test=len(test),
        n_windows_train=int(len(Xo)),
    )


CFG_FULL = dict(
    patients_per_site=12, n_days=10, epochs=30, rounds=8,
    local_epochs=3, pers_epochs=5, alpha=0.10,
    loop_patients=10, loop_days=5, loop_samples=24,
    probe_n=500, knn_k=50, min_support=0.05,
)
CFG_QUICK = dict(
    patients_per_site=5, n_days=4, epochs=5, rounds=2,
    local_epochs=1, pers_epochs=1, alpha=0.10,
    loop_patients=2, loop_days=1, loop_samples=12,
    probe_n=100, knn_k=20, min_support=0.05,
)


def main():
    parser = argparse.ArgumentParser(description="Run simulation experiments E1–E5.")
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--quick", action="store_true", help="Use small smoke-test settings.")
    parser.add_argument("--out", default=os.path.join(OUT, "results.json"))
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("--seeds must be at least 1")
    if args.start < 0:
        parser.error("--start must be non-negative")
    cfg = CFG_QUICK if args.quick else CFG_FULL
    print(f"config: {cfg}", flush=True)
    runs = []
    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    for seed in range(args.start, args.start + args.seeds):
        runs.append(run_seed(seed, cfg))
        with open(out_path, "w") as f:
            json.dump(dict(config=cfg, runs=runs), f, indent=2)
    print(f"wrote {out_path} ({len(runs)} seeds)")


if __name__ == "__main__":
    main()
