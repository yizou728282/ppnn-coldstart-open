"""Stage 3 (EUPPBench new-station validation) runner. See protocols/stage3_euppbench.json.

  python scripts/stage3_run.py --data data/stage3_euppbench.npz --out results/stage3 --fold-type random --baselines
  python scripts/stage3_run.py --data ... --out ... --fold-type random --variants unk noemb attr hybrid hybrid_res
  python scripts/stage3_run.py --data ... --out ... --emos-loc-bst-reference

Resumable: every finished (variant, fold, seed) writes its prediction file + log; existing files are skipped.
Per run, predictions for the held-out stations are stored in full (51- and 11-member test inputs); for the
seen (training) stations a running seed-sum of mu and sigma is kept per (variant, fold) so the seen-station
score of the seed ensemble can be computed without storing every seed.
"""
import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ppnn_residual.boosting import BoostedEMOS  # noqa: E402
from ppnn_residual.data import UNK  # noqa: E402
from ppnn_residual.device import device_info, get_device, set_deterministic  # noqa: E402
from ppnn_residual.emos import fit_emos, predict_emos  # noqa: E402
from ppnn_residual.folds import random_folds, spatial_folds  # noqa: E402
from ppnn_residual.metrics import crps_gaussian_np, crps_gaussian_torch  # noqa: E402
from ppnn_residual.stage2_models import StationNet, knn_idw  # noqa: E402

PROT = json.loads((ROOT / "protocols/stage3_euppbench.json").read_text())
LEADS = PROT["data"]["lead_times_h"]
VARIANTS = {
    "unk": dict(emb_mode="unk", station_dropout=0.05, methods=["nn_unk", "nn_knn"]),
    "noemb": dict(emb_mode="none", methods=["nn_noemb"]),
    "attr": dict(emb_mode="attr", methods=["nn_attr"]),
    "hybrid": dict(emb_mode="hybrid", methods=["nn_hybrid", "nn_hybrid_knn"]),
    "hybrid_res": dict(emb_mode="hybrid", residual=True, constrained_spread=True, methods=["nn_hybrid_res"]),
}
FIT_END = np.datetime64("2014-12-31"); VAL_END = np.datetime64("2016-12-31")


# ----------------------------------------------------------------------------------------- data
class Data:
    def __init__(self, path, subsample_stations=None, subsample_seed=0):
        z = np.load(path, allow_pickle=False)
        self.features = [str(f) for f in z["features"]]
        st = pd.read_json(io.StringIO(str(z["st_table"]))).set_index("station")
        st = st[st.included]
        if subsample_stations:
            rng = np.random.default_rng(subsample_seed)
            st = st.loc[np.sort(rng.choice(st.index.to_numpy(), subsample_stations, replace=False))]
        self.st = st
        keep_tr = np.isin(z["train_station"], st.index.to_numpy())
        keep_te = np.isin(z["test_station"], st.index.to_numpy())
        day = lambda a: a.astype("datetime64[D]")
        self.tr = dict(station=z["train_station"][keep_tr], lead=z["train_lead"][keep_tr],
                       valid=day(z["train_valid"][keep_tr]), y=z["train_y"][keep_tr].astype(np.float64),
                       X=z["train_X"][keep_tr])
        self.te = dict(station=z["test_station"][keep_te], lead=z["test_lead"][keep_te],
                       init=day(z["test_init"][keep_te]), valid=day(z["test_valid"][keep_te]),
                       y=z["test_y"][keep_te].astype(np.float64), X=z["test_X"][keep_te], X11=z["test_X11"][keep_te])
        self.i_m, self.i_s = self.features.index("t2m_mean"), self.features.index("t2m_sd")

    def attr(self, stations):
        s = self.st.loc[stations]
        lu = np.eye(5)[s.lu_group.to_numpy().astype(int)]
        return np.c_[s.lat, s.lon, s.alt, s.orog, s.alt - s.orog, lu].astype(np.float64)

    def design(self, part, X):
        """Full predictor matrix: 32 ensemble stats + doy sin/cos + lead one-hot + 10 attributes."""
        doy = (part["valid"] - part["valid"].astype("datetime64[Y]")).astype(int) / 365.25
        lead1h = (part["lead"][:, None] == np.array(LEADS)[None]).astype(np.float64)
        # attributes per row via station lookup
        ust, inv = np.unique(part["station"], return_inverse=True)
        A = self.attr(ust)[inv]
        return np.c_[X.astype(np.float64), np.sin(2 * np.pi * doy), np.cos(2 * np.pi * doy), lead1h, A]


def get_folds(D, ftype):
    f = PROT["folds"]["primary" if ftype == "random" else "secondary"]
    if ftype == "random":
        return random_folds(D.st.index.to_numpy(), f["K"], f["seed"]), f["K"]
    return spatial_folds(D.st[["lat", "lon"]], f["K"], f["seed"]), f["K"]


def split(D, folds, k):
    held_tr = np.array([folds[int(s)] == k for s in D.tr["station"]])
    fit = (~held_tr) & (D.tr["valid"] <= FIT_END)
    val = (~held_tr) & (D.tr["valid"] > FIT_END) & (D.tr["valid"] <= VAL_END)
    held_te = np.array([folds[int(s)] == k for s in D.te["station"]])
    return fit, val, held_te


def savez_atomic(path, **arrs):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp.npz")  # unique: several processes share --out
    np.savez(tmp, **arrs)
    os.replace(tmp, path)


def json_atomic(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=float))
    os.replace(tmp, path)


# ------------------------------------------------------------------------------------ baselines
def run_baselines(D, out, ftype, folds, K, maxit):
    for k in range(K):
        f = out / "preds" / ftype / f"emos_bst_gl_f{k}.npz"
        if f.exists():
            continue
        t0 = time.time()
        fit, val, held = split(D, folds, k)
        tr, te = D.tr, D.te
        info = {"fold": k, "n_fit": int(fit.sum())}
        res = {}
        for tag, X in (("", te["X"]), ("11", te["X11"])):
            res[f"raw_ensemble_gauss{tag}"] = (X[:, D.i_m].astype(np.float64), X[:, D.i_s].astype(np.float64))
        # global EMOS, one parameter set per lead time
        p, keys, inf = fit_emos(tr["X"][fit, D.i_m], tr["X"][fit, D.i_s].astype(np.float64) ** 2, tr["y"][fit],
                                groups=tr["lead"][fit])
        info["emos_gl"] = {"params_per_lead": {int(kk): pp.tolist() for kk, pp in zip(keys, p)}, **inf}
        for tag, X in (("", te["X"]), ("11", te["X11"])):
            mu, sg, _ = predict_emos(p, keys, X[:, D.i_m], X[:, D.i_s].astype(np.float64) ** 2, groups=te["lead"],
                                     fallback=p[0])  # every lead is seen; fallback never used
            res[f"emos_gl{tag}"] = (mu, sg)
        # global boosted EMOS per lead time on all predictors except the lead one-hot
        Ztr = D.design(tr, tr["X"]); keep_cols = [i for i in range(Ztr.shape[1]) if not (34 <= i < 34 + len(LEADS))]
        Ztr = Ztr[:, keep_cols]
        colmean = np.nanmean(Ztr[fit], 0)
        Ztr = np.where(np.isnan(Ztr), colmean, Ztr)
        Zte = {tag: D.design(te, X)[:, keep_cols] for tag, X in (("", te["X"]), ("11", te["X11"]))}
        Zte = {t: np.where(np.isnan(v), colmean, v) for t, v in Zte.items()}
        bmu = {t: np.full(len(te["y"]), np.nan) for t in Zte}; bsg = {t: np.full(len(te["y"]), np.nan) for t in Zte}
        info["emos_bst_gl"] = {}
        for L in LEADS:
            r = fit & (tr["lead"] == L)
            b = BoostedEMOS(nu=0.05, maxit=maxit).fit(Ztr[r], tr["y"][r])
            for t in Zte:
                rt = te["lead"] == L
                bmu[t][rt], bsg[t][rt] = b.predict(Zte[t][rt])
            info["emos_bst_gl"][L] = b.info
        for t in Zte:
            res[f"emos_bst_gl{t}"] = (bmu[t], bsg[t])
        for name in ["raw_ensemble_gauss", "emos_gl", "emos_bst_gl"]:
            m51, s51 = res[name]; m11, s11 = res[name + "11"]
            info[f"{name}_heldout_rows"] = int(held.sum())
            savez_atomic(out / "preds" / ftype / f"{name}_f{k}.npz", mu=m51.astype(np.float32), sigma=s51.astype(np.float32),
                         mu11=m11.astype(np.float32), sigma11=s11.astype(np.float32))
        info["sec"] = round(time.time() - t0, 1)
        json_atomic(out / "logs" / f"baselines_{ftype}_f{k}.json", info)
        print(ftype, "baselines fold", k, "sec", info["sec"], flush=True)


def run_emos_loc_bst_reference(D, out, maxit):
    f = out / "preds" / "reference" / "emos_loc_bst_seen.npz"
    if f.exists():
        return
    t0 = time.time()
    tr, te = D.tr, D.te
    Ztr = D.design(tr, tr["X"]); Zte = D.design(te, te["X"])
    loc_cols = list(range(34))  # ensemble statistics + doy; no lead one-hot, no station attributes
    mu = np.full(len(te["y"]), np.nan); sg = np.full(len(te["y"]), np.nan); ms = []
    trmask = tr["valid"] <= VAL_END
    for s in D.st.index:
        for L in LEADS:
            r = trmask & (tr["station"] == s) & (tr["lead"] == L)
            rt = (te["station"] == s) & (te["lead"] == L)
            if r.sum() < 10 or rt.sum() == 0:
                continue
            Z = Ztr[r][:, loc_cols]; cm = np.nanmean(Z, 0); Z = np.where(np.isnan(Z), cm, Z)
            b = BoostedEMOS(nu=0.05, maxit=maxit).fit(Z, tr["y"][r])
            Zt = Zte[rt][:, loc_cols]; Zt = np.where(np.isnan(Zt), cm, Zt)
            mu[rt], sg[rt] = b.predict(Zt)
            ms.append(b.mstop)
    savez_atomic(f, mu=mu, sigma=sg)
    json_atomic(out / "logs" / "emos_loc_bst_reference.json",
                {"n_fits": len(ms), "mstop_median": float(np.median(ms)), "mstop_min": int(min(ms)),
                 "mstop_max": int(max(ms)), "sec": round(time.time() - t0, 1)})
    print("emos_loc_bst reference done", flush=True)


# ------------------------------------------------------------------------------------- networks
class Arrays:
    def __init__(self, Z, idx, attr, m, sd, y, device):
        t = lambda a, dt=torch.float32: torch.as_tensor(np.ascontiguousarray(a), dtype=dt, device=device)
        self.x, self.idx, self.attr = t(Z), t(idx, torch.long), t(attr)
        self.m, self.sd, self.y = t(m), t(sd), t(y)
        self.n = len(y)


@torch.no_grad()
def predict(model, A, emb_vec=None, bs=65536):
    model.eval()
    mus, sgs = [], []
    for i in range(0, A.n, bs):
        sl = slice(i, i + bs)
        ev = None if emb_vec is None else emb_vec[sl]
        mu, sg = model(A.x[sl], A.idx[sl], A.attr[sl], A.m[sl], A.sd[sl], emb_vec=ev)
        mus.append(mu); sgs.append(sg)
    return torch.cat(mus).cpu().numpy().astype(np.float64), torch.cat(sgs).cpu().numpy().astype(np.float64)


def train_variant(var, Afit, Aval, n_emb, n_feat, n_attr, seed, hp, device, max_epochs):
    set_deterministic(seed, device)
    gen = torch.Generator().manual_seed(seed)  # CPU generator -> device-independent random stream
    cfg = VARIANTS[var]
    model = StationNet(n_feat, n_emb, n_attr, emb_mode=cfg["emb_mode"], emb_dim=hp["emb_dim"], hidden=hp["hidden"],
                       attr_hidden=hp["attr_mlp_hidden"], residual=cfg.get("residual", False),
                       constrained_spread=cfg.get("constrained_spread", False),
                       delta_dropout=hp["hybrid_offset_dropout"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"])
    p_drop = cfg.get("station_dropout", 0.0)
    best, best_state, best_ep, bad, hist = np.inf, None, 0, 0, []
    for ep in range(1, max_epochs + 1):
        model.train(); t0 = time.time()
        perm = torch.randperm(Afit.n, generator=gen).to(device); tot = 0.0
        for i in range(0, Afit.n, hp["batch_size"]):
            b = perm[i:i + hp["batch_size"]]
            idx = Afit.idx[b]
            if p_drop > 0:
                drop = (torch.rand(len(b), generator=gen) < p_drop).to(device)
                idx = torch.where(drop, torch.full_like(idx, UNK), idx)
            mu, sg = model(Afit.x[b], idx, Afit.attr[b], Afit.m[b], Afit.sd[b], training=True, generator=gen)
            loss = crps_gaussian_torch(mu, sg, Afit.y[b]).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()) * len(b)
        mu, sg = predict(model, Aval)
        v = float(crps_gaussian_np(mu, sg, Aval.y.cpu().numpy().astype(np.float64)).mean())
        hist.append({"epoch": ep, "train_crps": tot / Afit.n, "val_crps": v, "sec": round(time.time() - t0, 2)})
        if v < best - 1e-6:
            best, best_ep, bad = v, ep, 0
            best_state = {kk: vv.detach().clone() for kk, vv in model.state_dict().items()}
        else:
            bad += 1
            if bad >= hp["patience"]:
                break
    model.load_state_dict(best_state)
    return model, hist, best_ep, best


def accumulate_seen(path, seed, mu, sg):
    """Running seed-sum of seen-station predictions; records seeds to stay idempotent on resume."""
    if path.exists():
        z = np.load(path)
        seeds = list(z["seeds"]); smu, ssg = z["sum_mu"], z["sum_sigma"]
    else:
        seeds, smu, ssg = [], np.zeros_like(mu), np.zeros_like(sg)
    if seed in seeds:
        return
    savez_atomic(path, seeds=np.array(seeds + [seed]), sum_mu=smu + mu, sum_sigma=ssg + sg)


def run_networks(D, out, ftype, folds, variants, fold_ids, seeds, device, max_epochs):
    hp = PROT["nn_hyperparameters"]
    te = D.te
    for k in fold_ids:
        fit, val, held = split(D, folds, k)
        Zall = D.design(D.tr, D.tr["X"])
        cm = np.nanmean(Zall[fit], 0); cs = np.nanstd(Zall[fit], 0); cs[cs == 0] = 1.0
        norm = lambda Z: ((np.where(np.isnan(Z), cm, Z) - cm) / cs).astype(np.float32)
        train_st = np.unique(D.tr["station"][fit])
        st2idx = {int(s): i + 1 for i, s in enumerate(train_st)}
        n_emb = len(train_st) + 1
        A_all = D.attr(D.st.index.to_numpy()); am, asd = A_all[np.isin(D.st.index, train_st)].mean(0), A_all[np.isin(D.st.index, train_st)].std(0)
        asd[asd == 0] = 1.0
        def arrays(stations, Z, X, y):
            ust, inv = np.unique(stations, return_inverse=True)
            attr = ((D.attr(ust) - am) / asd)[inv]
            idx = np.array([st2idx.get(int(s), UNK) for s in stations])
            return Arrays(norm(Z), idx, attr, X[:, D.i_m], X[:, D.i_s], y, device)
        Afit = arrays(D.tr["station"][fit], Zall[fit], D.tr["X"][fit], D.tr["y"][fit])
        Aval = arrays(D.tr["station"][val], Zall[val], D.tr["X"][val], D.tr["y"][val])
        Ate = {t: arrays(te["station"], D.design(te, X), X, te["y"]) for t, X in (("", te["X"]), ("11", te["X11"]))}
        del Zall
        n_feat = Afit.x.shape[1]
        ref_st = train_st; stt = D.st.loc[ref_st]
        q_st = np.unique(te["station"][held]); qtab = D.st.loc[q_st]
        for var in variants:
            for seed in seeds:
                name0 = VARIANTS[var]["methods"][0]
                logf = out / "logs" / ftype / f"{var}_f{k}_s{seed}.json"
                if logf.exists():
                    continue
                t0 = time.time()
                model, hist, best_ep, best = train_variant(var, Afit, Aval, n_emb, n_feat, 10, seed, hp, device, max_epochs)
                preds = {}
                for t in Ate:
                    preds[(name0, t)] = predict(model, Ate[t])
                if var in ("unk", "hybrid"):
                    with torch.no_grad():
                        W = model.table.weight[torch.as_tensor([st2idx[int(s)] for s in ref_st], device=device)].cpu().numpy()
                    kv = knn_idw(qtab.lat.values, qtab.lon.values, qtab.alt.values, stt.lat.values, stt.lon.values,
                                 stt.alt.values, W, k=5, alt_km_per_m=0.1)
                    lut = {int(s): kv[i] for i, s in enumerate(q_st)}
                    rows = np.nonzero(held)[0]
                    add = torch.as_tensor(np.stack([lut[int(s)] for s in te["station"][rows]]).astype(np.float32), device=device)
                    rows_t = torch.as_tensor(rows, device=device)
                    for t in Ate:
                        with torch.no_grad():
                            ev = model.table(Ate[t].idx) if var == "unk" else model.station_vector(Ate[t].idx, Ate[t].attr)
                            ev = ev.clone()
                            ev[rows_t] = add if var == "unk" else ev[rows_t] + add
                        preds[(VARIANTS[var]["methods"][1], t)] = predict(model, Ate[t], emb_vec=ev)
                seen_crps = {}
                for name in VARIANTS[var]["methods"]:
                    mu, sg = preds[(name, "")]; mu11, sg11 = preds[(name, "11")]
                    savez_atomic(out / "preds" / ftype / f"{name}_f{k}_s{seed}.npz",
                                 mu=mu[held].astype(np.float32), sigma=sg[held].astype(np.float32),
                                 mu11=mu11[held].astype(np.float32), sigma11=sg11[held].astype(np.float32))
                    accumulate_seen(out / "preds" / ftype / f"seen_{name}_f{k}.npz", seed, mu[~held], sg[~held])
                    seen_crps[name] = float(crps_gaussian_np(mu[~held], sg[~held], te["y"][~held]).mean())
                json_atomic(logf, {"variant": var, "fold": k, "seed": seed, "best_epoch": best_ep, "best_val_crps": best,
                                   "history": hist, "sec": round(time.time() - t0, 1), "n_fit": Afit.n, "n_val": Aval.n,
                                   "seen_station_crps_single_seed": seen_crps, **device_info(device)})
                print(ftype, var, "fold", k, "seed", seed, "best_ep", best_ep, "val", round(best, 4),
                      "sec", round(time.time() - t0, 1), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=str(ROOT / "results/stage3"))
    ap.add_argument("--fold-type", choices=["random", "spatial"], default="random")
    ap.add_argument("--baselines", action="store_true")
    ap.add_argument("--emos-loc-bst-reference", action="store_true")
    ap.add_argument("--variants", nargs="*", default=[])
    ap.add_argument("--folds", nargs="*", type=int, default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--max-epochs", type=int, default=None, help="SMOKE TEST ONLY (protocol: 30)")
    ap.add_argument("--bst-maxit", type=int, default=1000, help="SMOKE TEST ONLY (protocol: 1000)")
    ap.add_argument("--subsample-stations", type=int, default=None, help="SMOKE TEST ONLY")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    D = Data(a.data, a.subsample_stations)
    folds, K = get_folds(D, a.fold_type)
    fpath = out / f"folds_{a.fold_type}.csv"
    if not fpath.exists():
        st = D.st[["country", "name", "lat", "lon", "alt", "orog", "lu_group"]].copy()
        st["fold"] = [folds[int(s)] for s in st.index]
        tmp = fpath.with_name(f"{fpath.name}.{os.getpid()}.tmp")
        st.to_csv(tmp); os.replace(tmp, fpath)
    tmeta = out / "test_meta.npz"
    if not tmeta.exists():
        savez_atomic(tmeta, station=D.te["station"], lead=D.te["lead"], init=D.te["init"].astype(np.int32),
                     y=D.te["y"])
    device = get_device(a.device)
    print("device:", device_info(device), flush=True)
    if a.baselines:
        run_baselines(D, out, a.fold_type, folds, K, a.bst_maxit)
    if a.emos_loc_bst_reference:
        run_emos_loc_bst_reference(D, out, a.bst_maxit)
    if a.variants:
        hp = PROT["nn_hyperparameters"]
        seeds = a.seeds if a.seeds is not None else (hp["seeds_primary"] if a.fold_type == "random" else hp["seeds_secondary_spatial"])
        run_networks(D, out, a.fold_type, folds, a.variants, a.folds if a.folds is not None else range(K), seeds, device,
                     a.max_epochs or hp["max_epochs"])


if __name__ == "__main__":
    main()
