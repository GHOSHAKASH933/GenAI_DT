"""
Experiment driver. Every number reported in the manuscript comes from here.

E1  treatment-effect validity   : does the twin learn a usable dose-response?
E2  predictive accuracy          : forecasting, centralised vs federated
E3  uncertainty calibration      : conformal coverage of the horizon minimum
E4  closed-loop control          : safety under gating, matched and shifted
E5  edge distillation            : latency / parameter cost
"""
from __future__ import annotations

import argparse, json, os, time
import numpy as np
import torch

from . import data as D
from .simulator import build_cohort, simulate_closed_loop, SITE_ARCHETYPES
from .models import (CVAETwin, ActionResidualTwin, PointMLP, EdgeStudent,
                     train_cvae, train_point, Persistence, RidgeAR)
from .federated import train_local_only, train_fedavg, personalise
from .conformal import ConformalGate
from .control import basal_only, pid_controller, twin_controller, glycaemic_metrics
from .intervention import (build_interventional_cohort, dose_response_slope,
                          ActionSupportGate)

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(OUT, exist_ok=True)


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
    """Dose-response recovery across training regimes and architectures."""
    res = {}
    probe = Xte[:cfg["probe_n"]]
    for dname, (X, Y) in datasets.items():
        for mname, cls in (("cvae", CVAETwin), ("ar_twin", ActionResidualTwin)):
            torch.manual_seed(seed)
            m = train_cvae(cls(), X, Y, epochs=cfg["epochs"], seed=seed)
            sl = dose_response_slope(m, probe)
            res[f"{dname}|{mname}"] = dict(
                slope_mean=float(sl.mean()), slope_sd=float(sl.std()),
                ratio=float(sl.mean() / true_s))
        # deterministic reference on identical data
        torch.manual_seed(seed)
        mlp = train_point(PointMLP(), X, Y, epochs=cfg["epochs"], seed=seed)
        lo = D.CONTEXT * 3
        ends = []
        for d in (0.0, 1.0, 2.0):
            Xd = probe.copy(); Xd[:, lo] = d / D.I_SCALE
            with torch.no_grad():
                ends.append(D.denorm(mlp(torch.as_tensor(Xd)).numpy())[:, -1].mean())
        sl = -(ends[-1] - ends[0]) / 2.0
        res[f"{dname}|point_mlp"] = dict(slope_mean=float(sl), slope_sd=float("nan"),
                                         ratio=float(sl / true_s))
    res["_true_sensitivity"] = float(true_s)
    return res


# ----------------------------------------------------------------- E2

def e2_prediction(train, test, seed, cfg):
    Xtr, Ytr, _ = D.pack(train)
    Xte, Yte, _ = D.pack(test)
    site_data = {s: D.pack(v)[:2] for s, v in D.by_site(train).items()}
    test_by_site = {s: D.pack(v)[:2] for s, v in D.by_site(test).items()}
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

    local = train_local_only(site_data, seed=seed, epochs=cfg["epochs"],
                             cls=ActionResidualTwin)
    errs = [(D.denorm(_mean_pred(local[s], Xs)) - D.denorm(Ys)) ** 2
            for s, (Xs, Ys) in test_by_site.items() if len(Xs)]
    e = np.concatenate(errs)
    res["twin_local_only"] = (float(np.sqrt(e.mean())), float(np.sqrt(e).mean()))

    glob = train_fedavg(site_data, seed=seed, rounds=cfg["rounds"],
                        local_epochs=cfg["local_epochs"], cls=ActionResidualTwin)
    res["twin_fedavg"] = rmse_mae(_mean_pred(glob, Xte), Yte)

    pers = personalise(glob, site_data, seed=seed, epochs=cfg["pers_epochs"],
                       cls=ActionResidualTwin)
    errs = [(D.denorm(_mean_pred(pers[s], Xs)) - D.denorm(Ys)) ** 2
            for s, (Xs, Ys) in test_by_site.items() if len(Xs)]
    e = np.concatenate(errs)
    res["twin_fedavg_personalised"] = (float(np.sqrt(e.mean())), float(np.sqrt(e).mean()))
    return res, glob, pers


# ----------------------------------------------------------------- E3/E4

def e3_e4(obs_twin, int_twin, Xtr_obs, Xtr_int, calib, test, seed, cfg):
    Xc, Yc, _ = D.pack(calib)
    Xte, Yte, _ = D.pack(test)

    cal = {}
    gates = {}
    for lab, model in (("obs", obs_twin), ("int", int_twin)):
        g = ConformalGate(alpha=cfg["alpha"]).fit(model, Xc, Yc)
        cov, width = g.coverage(model, Xte, Yte)
        from .conformal import twin_quantiles
        q_lo, _, q_hi = twin_quantiles(model, Xte, alpha=cfg["alpha"])
        yt = D.denorm(Yte).min(axis=1)
        cal[lab] = dict(nominal=1 - cfg["alpha"],
                        raw_coverage=float(((yt >= q_lo) & (yt <= q_hi)).mean()),
                        raw_width=float(np.mean(q_hi - q_lo)),
                        conformal_coverage=cov, conformal_width=width,
                        Q=g.Q, width_tau=g.width_tau, n_calib=g.n_calib)
        gates[lab] = g

    support = {"obs": ActionSupportGate(k=cfg["knn_k"], min_support=cfg["min_support"]).fit(Xtr_obs),
               "int": ActionSupportGate(k=cfg["knn_k"], min_support=cfg["min_support"]).fit(Xtr_int)}

    arms = {
        "basal_only":       ("none", None, None, None),
        "pid":              ("pid",  None, None, None),
        "twin_obs_ungated": ("twin", obs_twin, None, None),
        "twin_obs_conformal": ("twin", obs_twin, gates["obs"], None),
        "twin_obs_dual":    ("twin", obs_twin, gates["obs"], support["obs"]),
        "twin_int_dual":    ("twin", int_twin, gates["int"], support["int"]),
    }

    chosen = test[:min(cfg["loop_patients"], len(test))]
    regimes = {}
    for regime, use_ex in (("matched", False), ("shifted", True)):
        out = {}
        for name, (kind, model, gate, sup) in arms.items():
            per = []
            for rec in chosen:
                sd = abs(hash((seed, rec["pid"], regime))) % (2 ** 32)
                rng = np.random.default_rng(sd)
                p = rec["params"]
                if kind == "none":
                    ctrl = basal_only(p)
                elif kind == "pid":
                    ctrl = pid_controller(p)
                else:
                    ctrl = twin_controller(model, p, gate, support=sup,
                                           n_samples=cfg["loop_samples"])
                tr = simulate_closed_loop(p, cfg["loop_days"], rng, ctrl,
                                          exercise=use_ex, decision_period=6)
                m = glycaemic_metrics(tr)
                m["pid_"] = rec["pid"]
                per.append(m)
            out[name] = per
        regimes[regime] = out
    return cal, regimes


# ----------------------------------------------------------------- E5

def e5_edge(twin, train, test, seed, cfg):
    Xtr, _, _ = D.pack(train)
    Xte, Yte, _ = D.pack(test)
    a = cfg["alpha"]
    los, mds, his = [], [], []
    Xt = torch.as_tensor(Xtr)
    with torch.no_grad():
        for b in range(0, len(Xt), 1024):
            s = twin.sample(Xt[b:b + 1024], n_samples=48)
            los.append(torch.quantile(s, a / 2, dim=0))
            mds.append(torch.quantile(s, 0.5, dim=0))
            his.append(torch.quantile(s, 1 - a / 2, dim=0))
    t_lo, t_md, t_hi = torch.cat(los), torch.cat(mds), torch.cat(his)

    torch.manual_seed(seed)
    stu = EdgeStudent()
    opt = torch.optim.AdamW(stu.parameters(), lr=3e-3, weight_decay=1e-4)
    g = torch.Generator().manual_seed(seed)
    for _ in range(cfg["epochs"]):
        perm = torch.randperm(len(Xt), generator=g)
        for b in range(0, len(Xt), 256):
            i = perm[b:b + 256]
            lo, md, hi = stu(Xt[i])
            loss = ((lo - t_lo[i]) ** 2 + (md - t_md[i]) ** 2 + (hi - t_hi[i]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    stu.eval()

    with torch.no_grad():
        _, md, _ = stu(torch.as_tensor(Xte))
    s_rmse, s_mae = rmse_mae(md.numpy(), Yte)
    t_rmse, t_mae = rmse_mae(_mean_pred(twin, Xte), Yte)

    def lat(fn, reps=150):
        x1 = torch.as_tensor(Xte[:1]); fn(x1)
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter(); fn(x1); ts.append((time.perf_counter() - t0) * 1000)
        return float(np.median(ts)), float(np.percentile(ts, 95))

    with torch.no_grad():
        tm, tp = lat(lambda x: twin.sample(x, n_samples=cfg["loop_samples"]))
        sm, sp = lat(lambda x: stu(x))
    npar = lambda m: sum(p.numel() for p in m.parameters())
    return dict(teacher_rmse=t_rmse, teacher_mae=t_mae,
                student_rmse=s_rmse, student_mae=s_mae,
                teacher_latency_ms=tm, teacher_latency_p95=tp,
                student_latency_ms=sm, student_latency_p95=sp,
                teacher_params=npar(twin), student_params=npar(stu))


# ----------------------------------------------------------------- driver

def run_seed(seed, cfg):
    t0 = time.time()
    cohort = build_cohort(seed, patients_per_site=cfg["patients_per_site"],
                          n_days=cfg["n_days"])
    train, calib, test = D.split_by_patient(cohort, seed)
    true_s = float(np.mean([r["params"].true_s_ins for r in test]))

    Xo, Yo, _ = D.pack(train)
    int_rand = build_interventional_cohort(train, cfg["n_days"], seed,
                                           state_independent=True)
    int_tit = build_interventional_cohort(train, cfg["n_days"], seed,
                                          state_independent=False)
    Xi, Yi, _ = D.pack(int_rand)
    Xt_, Yt_, _ = D.pack(int_tit)
    Xte, _, _ = D.pack(test)

    datasets = {"observational": (Xo, Yo),
                "interventional_randomised": (Xi, Yi),
                "interventional_titrated": (Xt_, Yt_)}
    e1 = e1_treatment_effect(datasets, Xte, true_s, seed, cfg)
    e2, glob, pers = e2_prediction(train, test, seed, cfg)

    torch.manual_seed(seed)
    obs_twin = train_cvae(CVAETwin(), Xo, Yo, epochs=cfg["epochs"], seed=seed)
    torch.manual_seed(seed)
    int_twin = train_cvae(ActionResidualTwin(), Xi, Yi, epochs=cfg["epochs"], seed=seed)

    cal, loop = e3_e4(obs_twin, int_twin, Xo, Xi, calib, test, seed, cfg)
    e5 = e5_edge(int_twin, int_rand, test, seed, cfg)

    print(f"  seed {seed} done in {time.time() - t0:.0f}s", flush=True)
    return dict(seed=seed, true_sensitivity=true_s, e1=e1, e2=e2,
                calibration=cal, loop=loop, e5=e5,
                n_train=len(train), n_calib=len(calib), n_test=len(test),
                n_windows_train=int(len(Xo)))


CFG_FULL = dict(patients_per_site=12, n_days=10, epochs=30, rounds=8,
                local_epochs=3, pers_epochs=5, alpha=0.10,
                loop_patients=10, loop_days=5, loop_samples=24,
                probe_n=500, knn_k=50, min_support=0.05)
CFG_QUICK = dict(patients_per_site=5, n_days=4, epochs=5, rounds=2,
                 local_epochs=1, pers_epochs=1, alpha=0.10,
                 loop_patients=2, loop_days=1, loop_samples=12,
                 probe_n=100, knn_k=20, min_support=0.05)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=os.path.join(OUT, "results.json"))
    a = ap.parse_args()
    cfg = CFG_QUICK if a.quick else CFG_FULL
    print(f"config: {cfg}", flush=True)
    runs = []
    for s in range(a.start, a.start + a.seeds):
        runs.append(run_seed(s, cfg))
        with open(a.out, "w") as f:
            json.dump(dict(config=cfg, runs=runs), f, indent=2)
    print(f"wrote {a.out} ({len(runs)} seeds)")


if __name__ == "__main__":
    main()
