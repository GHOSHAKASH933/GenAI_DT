"""
Real-data experiments on ShanghaiT1DM/T2DM.

R1  two-hour forecasting on held-out patients
R2  split-conformal coverage of the two-hour glucose minimum (overall, by diabetes type)
R3  action support: which insulin doses have support in the real records
R4  insulin effect learned by each model (no ground truth exists in real data;
    we report sign and size and compare with the confounded raw association)

Closed-loop control is NOT evaluated on real data: it would require
intervening on real patients.
"""
from __future__ import annotations
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, str(ROOT))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---- configure the shared twin code for 15-minute data BEFORE importing models
import genai_dt.data as D
D.CONTEXT, D.HORIZON = 8, 8               # 2 h of history, 2 h ahead
D.CTX_DIM = D.CONTEXT * 3 + D.HORIZON * 2
D.I_SCALE, D.C_SCALE = 6.0, 1.0           # IU per step; meal events are 0/1
import torch
from genai_dt.models import (CVAETwin, ActionResidualTwin, PointMLP, train_cvae,
                             train_point, Persistence, RidgeAR)
from genai_dt.conformal import ConformalGate, twin_quantiles
from genai_dt.intervention import ActionSupportGate
from shanghai import load_all

torch.set_num_threads(2)
DOSES = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]


def windows(rec, stride=2):
    g, i, c = rec["glucose"], rec["insulin"], rec["carbs"]
    K, H = D.CONTEXT, D.HORIZON
    X, Y = [], []
    for s in range(0, len(g) - K - H + 1, stride):
        gw = g[s:s + K + H]
        if np.isnan(gw).any():
            continue
        X.append(D.encode_context(g[s:s + K], i[s:s + K], c[s:s + K],
                                  i[s + K:s + K + H], c[s + K:s + K + H]))
        Y.append((g[s + K:s + K + H] - D.G_MEAN) / D.G_SCALE)
    if not X:
        return np.zeros((0, D.CTX_DIM), np.float32), np.zeros((0, H), np.float32)
    return np.asarray(X, np.float32), np.asarray(Y, np.float32)


def pack(recs, stride=2):
    xs, ys, ts = [], [], []
    for r in recs:
        x, y = windows(r, stride)
        if len(x):
            xs.append(x); ys.append(y); ts += [r["dtype"]] * len(x)
    return np.concatenate(xs), np.concatenate(ys), np.asarray(ts)


def split(recs, seed, frac=(0.6, 0.2)):
    """Patient-level split stratified by diabetes type. All records of one
    patient go to the same group."""
    rng = np.random.default_rng(1000 + seed)
    out = {"train": [], "calib": [], "test": []}
    for dt in ("T1DM", "T2DM"):
        pids = sorted({r["pid"] for r in recs if r["dtype"] == dt})
        rng.shuffle(pids)
        n = len(pids); a = int(round(frac[0] * n)); b = a + int(round(frac[1] * n))
        groups = {p: ("train" if k < a else "calib" if k < b else "test") for k, p in enumerate(pids)}
        for r in recs:
            if r["dtype"] == dt:
                out[groups[r["pid"]]].append(r)
    return out


def rmse(p, y):
    e = D.denorm(p) - D.denorm(y)
    return float(np.sqrt((e ** 2).mean())), float(np.sqrt((e[:, 3] ** 2).mean())), \
        float(np.sqrt((e[:, -1] ** 2).mean())), float(np.abs(e).mean())


def mean_pred(model, X, n=32, bs=4096):
    out = []
    with torch.no_grad():
        for b in range(0, len(X), bs):
            out.append(model.sample(torch.as_tensor(X[b:b + bs]), n_samples=n,
                                    include_obs_noise=False).mean(0).numpy())
    return np.concatenate(out)


def slope(predict, X):
    """Learned 2-h glucose change per IU, probed at contexts with the planned
    meal channel set to zero. Positive = insulin lowers glucose."""
    lo = D.CONTEXT * 3
    Xp = X.copy()
    Xp[:, lo + D.HORIZON:lo + 2 * D.HORIZON] = 0.0
    ends = []
    for d in DOSES:
        Xd = Xp.copy(); Xd[:, lo] = d / D.I_SCALE
        ends.append(D.denorm(predict(Xd))[:, -1])
    E = np.stack(ends); dv = np.asarray(DOSES); dc = dv - dv.mean()
    s = (dc[:, None] * (E - E.mean(0))).sum(0) / (dc ** 2).sum()
    return -s


def raw_association(X, Y):
    """Observed 2-h change vs dose in the records, adjusted for current glucose
    and planned meals: the confounded association a naive model sees."""
    lo = D.CONTEXT * 3
    dose = X[:, lo] * D.I_SCALE
    g0 = X[:, D.CONTEXT - 1] * D.G_SCALE + D.G_MEAN
    meal = X[:, lo + D.HORIZON:lo + 2 * D.HORIZON].sum(1)
    dy = D.denorm(Y)[:, -1] - g0
    A = np.column_stack([dose, g0, meal, np.ones(len(dose))])
    b = np.linalg.lstsq(A, dy, rcond=None)[0]
    return float(-b[0])


def run_seed(recs, seed, cfg):
    t0 = time.time()
    sp = split(recs, seed)
    Xtr, Ytr, _ = pack(sp["train"], cfg["stride"])
    Xca, Yca, Tca = pack(sp["calib"], cfg["stride"])
    Xte, Yte, Tte = pack(sp["test"], cfg["stride"])
    res = dict(seed=seed, n_windows=dict(train=len(Xtr), calib=len(Xca), test=len(Xte)),
               n_patients={k: len({r["pid"] + r["dtype"] for r in v}) for k, v in sp.items()})

    # R1 forecasting
    f = {}
    f["persistence"] = rmse(Persistence().predict(Xte), Yte)
    ridge = RidgeAR(lam=5.0).fit(Xtr, Ytr)
    f["ridge_ar"] = rmse(ridge.predict(Xte), Yte)
    torch.manual_seed(seed)
    mlp = train_point(PointMLP(), Xtr, Ytr, epochs=cfg["epochs"], seed=seed)
    with torch.no_grad():
        f["mlp"] = rmse(mlp(torch.as_tensor(Xte)).numpy(), Yte)
    torch.manual_seed(seed)
    cvae = train_cvae(CVAETwin(), Xtr, Ytr, epochs=cfg["epochs"], seed=seed)
    f["cvae"] = rmse(mean_pred(cvae, Xte), Yte)
    torch.manual_seed(seed)
    art = train_cvae(ActionResidualTwin(), Xtr, Ytr, epochs=cfg["epochs"], seed=seed)
    f["ar_twin"] = rmse(mean_pred(art, Xte), Yte)
    res["forecast"] = f

    # R2 conformal coverage of the 2-h minimum
    c = {}
    ytrue = D.denorm(Yte).min(1)
    for name, m in (("cvae", cvae), ("ar_twin", art)):
        gate = ConformalGate(alpha=cfg["alpha"]).fit(m, Xca, Yca, match_action_support=False)
        ql, _, qh = twin_quantiles(m, Xte, alpha=cfg["alpha"])
        lo_, hi_ = ql - gate.Q, qh + gate.Q
        raw = (ytrue >= ql) & (ytrue <= qh); conf = (ytrue >= lo_) & (ytrue <= hi_)
        c[name] = dict(raw_cov=float(raw.mean()), raw_width=float((qh - ql).mean()),
                       conf_cov=float(conf.mean()), conf_width=float((hi_ - lo_).mean()),
                       conf_cov_T1DM=float(conf[Tte == "T1DM"].mean()) if (Tte == "T1DM").any() else float("nan"),
                       conf_cov_T2DM=float(conf[Tte == "T2DM"].mean()))
        # group-conditional (Mondrian) calibration: one correction per diabetes type
        cov_g = np.zeros(len(ytrue), bool); wid_g = np.zeros(len(ytrue))
        for grp in ("T1DM", "T2DM"):
            mc, mt = Tca == grp, Tte == grp
            if mc.sum() < 50 or mt.sum() == 0:
                continue
            gg = ConformalGate(alpha=cfg["alpha"]).fit(m, Xca[mc], Yca[mc], match_action_support=False)
            l2, h2 = ql[mt] - gg.Q, qh[mt] + gg.Q
            cov_g[mt] = (ytrue[mt] >= l2) & (ytrue[mt] <= h2); wid_g[mt] = h2 - l2
            c[name][f"mondrian_cov_{grp}"] = float(cov_g[mt].mean())
            c[name][f"mondrian_width_{grp}"] = float(wid_g[mt].mean())
        c[name]["mondrian_cov"] = float(cov_g.mean())
        c[name]["mondrian_width"] = float(wid_g.mean())
    res["calibration"] = c

    # R3 action support
    lo = D.CONTEXT * 3
    sup = ActionSupportGate(k=cfg["knn_k"], min_support=cfg["min_support"]).fit(Xtr)
    rng = np.random.default_rng(seed)
    probe = Xte[rng.choice(len(Xte), size=min(cfg["probe_n"], len(Xte)), replace=False)]
    prof = {}
    for d in DOSES:
        Xd = probe.copy(); Xd[:, lo] = d / D.I_SCALE
        prof[f"{d:.0f}"] = float(sup.admissible(Xd).mean())
    res["support"] = dict(profile=prof,
                          frac_windows_with_dose=float((Xtr[:, lo] > 0).mean()),
                          median_dose=float(np.median(Xtr[Xtr[:, lo] > 0, lo] * D.I_SCALE))
                          if (Xtr[:, lo] > 0).any() else 0.0)

    # R4 learned insulin effect (mg/dL fall at 2 h per IU)
    nog = probe
    eff = {}
    eff["cvae"] = float(np.mean(slope(lambda X: mean_pred(cvae, X, n=16), nog)))
    eff["ar_twin"] = float(np.mean(slope(lambda X: mean_pred(art, X, n=16), nog)))
    with torch.no_grad():
        eff["mlp"] = float(np.mean(slope(lambda X: mlp(torch.as_tensor(X)).numpy(), nog)))
    eff["ridge_ar"] = float(np.mean(slope(lambda X: ridge.predict(X), nog)))
    eff["raw_association"] = raw_association(Xtr, Ytr)
    res["effect"] = eff
    res["seconds"] = round(time.time() - t0)
    print(f"  seed {seed} done in {res['seconds']}s", flush=True)
    return res


CFG = dict(epochs=25, alpha=0.10, stride=2, knn_k=50, min_support=0.05, probe_n=1500)
QUICK = dict(epochs=2, alpha=0.10, stride=6, knn_k=20, min_support=0.05, probe_n=200)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "results" / "real_results.json"))
    a = ap.parse_args()
    cfg = QUICK if a.quick else CFG
    recs = load_all(a.data)
    print(f"loaded {len(recs)} records; config {cfg}", flush=True)
    runs = []
    for s in range(a.seeds):
        runs.append(run_seed(recs, s, cfg))
        json.dump(dict(config=cfg, dataset="ShanghaiT1DM+ShanghaiT2DM", runs=runs),
                  open(a.out, "w"), indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
