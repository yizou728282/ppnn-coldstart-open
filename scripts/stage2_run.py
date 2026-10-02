"""Stage 2 (new-station generalization) runner. See protocols/stage2_new_station.json.

  python scripts/stage2_run.py --fold-type random --baselines
  python scripts/stage2_run.py --fold-type random --variants unk hybrid --folds 0 1 --seeds 0 1
  python scripts/stage2_run.py --emos-loc-bst-reference
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ppnn_residual.data import Preprocessor, feature_columns, load_table, UNK  # noqa: E402
from ppnn_residual.emos import fit_emos, predict_emos  # noqa: E402
from ppnn_residual.boosting import BoostedEMOS  # noqa: E402
from ppnn_residual.folds import station_table, random_folds, spatial_folds  # noqa: E402
from ppnn_residual.metrics import crps_gaussian_torch  # noqa: E402
from ppnn_residual.stage2_models import StationNet, knn_idw  # noqa: E402

PROT = json.loads((ROOT / "protocols/stage2_new_station.json").read_text())
OUT = ROOT / "results/stage2"
ATTR = ["lat", "lon", "alt", "orog"]
VARIANTS = {  # trained network variants -> evaluated method names
    "unk": dict(emb_mode="unk", station_dropout=0.05, methods=["nn_unk", "nn_knn"]),
    "noemb": dict(emb_mode="none", methods=["nn_noemb"]),
    "attr": dict(emb_mode="attr", methods=["nn_attr"]),
    "hybrid": dict(emb_mode="hybrid", methods=["nn_hybrid", "nn_hybrid_knn"]),
    "hybrid_res": dict(emb_mode="hybrid", residual=True, constrained_spread=True, methods=["nn_hybrid_res"]),
}


def get_folds(df, ftype):
    st = station_table(df)
    f = PROT["folds"]["primary" if ftype == "random" else "secondary"]
    if ftype == "random":
        return random_folds(st.index.to_numpy(), f["K"], f["seed"]), f["K"]
    return spatial_folds(st, f["K"], f["seed"]), f["K"]


def attr_matrix(d):
    return np.c_[d[ATTR].to_numpy(np.float64), (d["alt"] - d["orog"]).to_numpy(np.float64)]


def prepare(df, folds, k):
    held = np.array([folds[int(s)] == k for s in df["station"].to_numpy()])
    yr = df["year"].to_numpy()
    fit = df[(~held) & np.isin(yr, PROT["years"]["fit"])].reset_index(drop=True)
    val = df[(~held) & (yr == PROT["years"]["early_stopping"])].reset_index(drop=True)
    test = df[yr == PROT["years"]["test"]].reset_index(drop=True)
    return fit, val, test


def save_pred(ftype, name, k, seed, mu, sigma):
    d = OUT / "preds" / ftype
    d.mkdir(parents=True, exist_ok=True)
    tag = f"{name}_f{k}" + ("" if seed is None else f"_s{seed}")
    np.savez(d / f"{tag}.npz", mu=np.asarray(mu, np.float32), sigma=np.asarray(sigma, np.float32))


def run_baselines(df, cols, ftype, folds, K):
    info = {}
    for k in range(K):
        t0 = time.time()
        fit, val, test = prepare(df, folds, k)
        save_pred(ftype, "raw_ensemble_gauss", k, None, test.t2m_mean, np.sqrt(test.t2m_var))
        p, keys, inf = fit_emos(fit.t2m_mean, fit.t2m_var, fit.obs)
        mu, sg, _ = predict_emos(p, keys, test.t2m_mean, test.t2m_var)
        save_pred(ftype, "emos_gl", k, None, mu, sg)
        # boosted global EMOS on all 38 predictors, AIC stopping (maxit/nu from protocol)
        b = BoostedEMOS(nu=0.05, maxit=1000).fit(fit[cols].to_numpy(), fit.obs.to_numpy())
        mu, sg = b.predict(test[cols].to_numpy())
        save_pred(ftype, "emos_bst_gl", k, None, mu, sg)
        info[k] = {"emos_gl": {"params": p[0].tolist(), **inf}, "emos_bst_gl": b.info,
                   "n_fit": len(fit), "sec": round(time.time() - t0, 1)}
        print(ftype, "fold", k, info[k], flush=True)
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    json.dump(info, open(OUT / f"logs/baselines_{ftype}.json", "w"), indent=1)


def run_emos_loc_bst_reference(df, cols):
    """Per-station boosted EMOS in the stage-1 setting (fit 2007-2015, test 2016, seen stations)."""
    loc_cols = [c for c in cols if c not in ATTR]
    tr = df[df.year <= 2015]; te = df[df.year == 2016].reset_index(drop=True)
    mu = np.full(len(te), np.nan); sg = np.full(len(te), np.nan)
    ms = []
    t0 = time.time()
    for s, g in tr.groupby("station"):
        if len(g) < 10:
            continue
        rows = np.nonzero(te.station.to_numpy() == s)[0]
        if len(rows) == 0:
            continue
        b = BoostedEMOS(nu=0.05, maxit=1000).fit(g[loc_cols].to_numpy(), g.obs.to_numpy())
        mu[rows], sg[rows] = b.predict(te.loc[rows, loc_cols].to_numpy())
        ms.append(b.mstop)
    d = OUT / "preds" / "reference"; d.mkdir(parents=True, exist_ok=True)
    np.savez(d / "emos_loc_bst_seen.npz", mu=mu, sigma=sg, station=te.station.to_numpy(), y=te.obs.to_numpy())
    json.dump({"n_stations": len(ms), "mstop_median": float(np.median(ms)), "mstop_min": int(min(ms)),
               "mstop_max": int(max(ms)), "sec": round(time.time() - t0, 1)},
              open(OUT / "logs/emos_loc_bst_reference.json", "w"), indent=1)
    print("emos_loc_bst reference done", flush=True)


class Arrays:
    def __init__(self, d, pp, amean, asd):
        s = pp.transform(d)
        self.x = torch.from_numpy(s.x); self.idx = torch.from_numpy(s.emb_idx)
        self.m = torch.from_numpy(s.ens_mean); self.sd = torch.from_numpy(s.ens_sd); self.y = torch.from_numpy(s.y)
        self.attr = torch.from_numpy(((attr_matrix(d) - amean) / asd).astype(np.float32))
        self.n = len(s.y)


@torch.no_grad()
def predict(model, A, emb_vec=None, bs=65536):
    model.eval()
    mus, sgs = [], []
    for i in range(0, A.n, bs):
        sl = slice(i, i + bs)
        ev = None if emb_vec is None else emb_vec[sl]
        mu, sg = model(A.x[sl], A.idx[sl], A.attr[sl], A.m[sl], A.sd[sl], emb_vec=ev)
        mus.append(mu); sgs.append(sg)
    return torch.cat(mus).numpy(), torch.cat(sgs).numpy()


def train_variant(var, Afit, Aval, n_emb, n_feat, seed, hp):
    torch.manual_seed(seed); gen = torch.Generator().manual_seed(seed)
    cfg = VARIANTS[var]
    model = StationNet(n_feat, n_emb, 5, emb_mode=cfg["emb_mode"], emb_dim=hp["emb_dim"], hidden=hp["hidden"],
                       attr_hidden=hp["attr_mlp_hidden"], residual=cfg.get("residual", False),
                       constrained_spread=cfg.get("constrained_spread", False), delta_dropout=0.5)
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"])
    p_drop = cfg.get("station_dropout", 0.0)
    best, best_state, best_ep, bad, hist = np.inf, None, 0, 0, []
    for ep in range(1, hp["max_epochs"] + 1):
        model.train(); t0 = time.time()
        perm = torch.randperm(Afit.n, generator=gen); tot = 0.0
        for i in range(0, Afit.n, hp["batch_size"]):
            b = perm[i:i + hp["batch_size"]]
            idx = Afit.idx[b]
            if p_drop > 0:
                idx = torch.where(torch.rand(len(b), generator=gen) < p_drop, torch.full_like(idx, UNK), idx)
            mu, sg = model(Afit.x[b], idx, Afit.attr[b], Afit.m[b], Afit.sd[b], training=True, generator=gen)
            loss = crps_gaussian_torch(mu, sg, Afit.y[b]).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(b)
        mu, sg = predict(model, Aval)
        v = float(crps_gaussian_torch(torch.from_numpy(mu).double(), torch.from_numpy(sg).double(), Aval.y.double()).mean())
        hist.append({"epoch": ep, "train_crps": tot / Afit.n, "val_crps": v, "sec": round(time.time() - t0, 1)})
        if v < best - 1e-6:
            best, best_ep, bad = v, ep, 0
            best_state = {kk: vv.clone() for kk, vv in model.state_dict().items()}
        else:
            bad += 1
            if bad >= hp["patience"]:
                break
    model.load_state_dict(best_state)
    return model, hist, best_ep, best


def run_networks(df, cols, ftype, folds, variants, fold_ids, seeds, threads):
    torch.set_num_threads(threads)
    hp = PROT["nn_hyperparameters"]
    for k in fold_ids:
        fit, val, test = prepare(df, folds, k)
        pp = Preprocessor().fit(fit, cols)
        am, asd = attr_matrix(fit).mean(0), attr_matrix(fit).std(0)
        Afit, Aval, Ate = Arrays(fit, pp, am, asd), Arrays(val, pp, am, asd), Arrays(test, pp, am, asd)
        # station coordinates for kNN
        ref_st = np.array(sorted(pp.station_to_idx, key=pp.station_to_idx.get))
        stt = station_table(df).loc[ref_st]
        te_unk = (Ate.idx == UNK).numpy()
        q_st = np.unique(test.station.to_numpy()[te_unk])
        qtab = station_table(df).loc[q_st]
        for var in variants:
            for seed in seeds:
                name0 = VARIANTS[var]["methods"][0]
                if (OUT / "preds" / ftype / f"{name0}_f{k}_s{seed}.npz").exists():
                    continue
                t0 = time.time()
                model, hist, best_ep, best = train_variant(var, Afit, Aval, pp.n_emb, len(cols), seed, hp)
                mu, sg = predict(model, Ate)
                save_pred(ftype, name0, k, seed, mu, sg)
                if var in ("unk", "hybrid"):  # kNN-IDW station vector for unseen test stations
                    with torch.no_grad():
                        W = model.table.weight[torch.as_tensor([pp.station_to_idx[int(s)] for s in ref_st])].numpy()
                    kv = knn_idw(qtab.lat.values, qtab.lon.values, qtab.alt.values, stt.lat.values, stt.lon.values,
                                 stt.alt.values, W, k=5, alt_km_per_m=0.1)
                    lut = {int(s): kv[i] for i, s in enumerate(q_st)}
                    with torch.no_grad():
                        ev = torch.zeros(Ate.n, hp["emb_dim"])
                        if var == "unk":
                            ev[:] = model.table(Ate.idx)
                        else:
                            ev[:] = model.station_vector(Ate.idx, Ate.attr)
                        rows = np.nonzero(te_unk)[0]
                        add = torch.from_numpy(np.stack([lut[int(s)] for s in test.station.to_numpy()[rows]]).astype(np.float32))
                        if var == "unk":
                            ev[rows] = add
                        else:
                            ev[rows] = ev[rows] + add  # attr part + kNN delta
                    mu2, sg2 = predict(model, Ate, emb_vec=ev)
                    save_pred(ftype, VARIANTS[var]["methods"][1], k, seed, mu2, sg2)
                d = OUT / "logs" / ftype; d.mkdir(parents=True, exist_ok=True)
                json.dump({"variant": var, "fold": k, "seed": seed, "best_epoch": best_ep, "best_val_crps": best,
                           "history": hist, "sec": round(time.time() - t0, 1), "threads": threads},
                          open(d / f"{var}_f{k}_s{seed}.json", "w"), indent=1)
                print(ftype, var, "fold", k, "seed", seed, "best_ep", best_ep, "val", round(best, 4),
                      "sec", round(time.time() - t0, 1), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold-type", choices=["random", "spatial"], default="random")
    ap.add_argument("--baselines", action="store_true")
    ap.add_argument("--emos-loc-bst-reference", action="store_true")
    ap.add_argument("--variants", nargs="*", default=[])
    ap.add_argument("--folds", nargs="*", type=int, default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    df = load_table(str(ROOT / PROT["data"]["path"]), PROT["data"]["drop_columns"])
    cols = feature_columns(df)
    folds, K = get_folds(df, a.fold_type)
    OUT.mkdir(parents=True, exist_ok=True)
    fpath = OUT / f"folds_{a.fold_type}.csv"
    if not fpath.exists():
        st = station_table(df)
        st["fold"] = [folds[int(s)] for s in st.index]
        st.to_csv(fpath)
    tmeta = OUT / "test_meta.npz"
    if not tmeta.exists():
        te = df[df.year == PROT["years"]["test"]].reset_index(drop=True)
        np.savez(tmeta, station=te.station.to_numpy(), date=pd.to_datetime(te.date).dt.strftime("%Y-%m-%d").to_numpy(),
                 y=te.obs.to_numpy(), lat=te.lat.to_numpy(), lon=te.lon.to_numpy(), alt=te.alt.to_numpy())
    if a.baselines:
        run_baselines(df, cols, a.fold_type, folds, K)
    if a.emos_loc_bst_reference:
        run_emos_loc_bst_reference(df, cols)
    if a.variants:
        seeds = a.seeds if a.seeds is not None else (PROT["nn_hyperparameters"]["seeds_primary"] if a.fold_type == "random"
                                                     else PROT["nn_hyperparameters"]["seeds_secondary_spatial"])
        run_networks(df, cols, a.fold_type, folds, a.variants, a.folds if a.folds is not None else range(K), seeds, a.threads)


if __name__ == "__main__":
    main()
