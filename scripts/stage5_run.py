"""Stage 5 runner (protocols/stage5_supplementary.json). Reuses the Stage-2/3/4 training code unchanged.

  stage5_run.py seeds3   --dataset euppbench --variants unk hybrid_res --seeds 3 4 5 6 7 8 9   (Stage-3 nets, primary spatial)
  stage5_run.py seeds2   --variants unk noemb attr hybrid hybrid_res --seeds 3 ... 9            (Stage-2 nets, German spatial)
  stage5_run.py stage4   --dataset D --design spatial|random|part_<pid> --methods drn_lak ... --seeds ...
  stage5_run.py part_bst --pid km7_s2027                                                      (raw/emos_gl/emos_bst_gl)
  stage5_run.py part_nets3 --pid km7_s2027 --seeds 0 1 2                                      (nn_unk/nn_knn, nn_hybrid_res)
  stage5_run.py statbase --dataset D --design random|spatial|part_<pid>                        (interpolated EMOS, samos_lin)
  stage5_run.py bstsens  --fold-type spatial|random                                           (EUPPBench boosting path 4000)
  stage5_run.py m1a      --fold-type spatial|random --methods gnn_geo samos_attn --seeds 0 1 2
Resumable: finished units are skipped. Outputs under results/stage5/<dataset>/<design>/{preds,logs}/<ftype>/.
"""
import argparse
import sys
import time

import numpy as np
import pandas as pd
import torch

from stage5_common import (OLD_DIR, R5, ROOT, SMOKE, design_dir, folds_df, folds_dict, json_atomic, savez_atomic)  # noqa: E402

EUPP_NPZ = ROOT / "data/stage3/stage3_euppbench.npz"


def set_threads(n):
    torch.set_num_threads(n)


# ------------------------------------------------------------------------------ Stage-3 nets (EUPPBench)
def cmd_seeds3(a):
    import stage3_run as s3
    D = s3.Data(str(EUPP_NPZ))
    folds, K = s3.get_folds(D, "spatial")
    ref = folds_dict(folds_df("euppbench", "spatial"))
    assert folds == ref, "primary spatial folds do not reproduce"
    hp = s3.PROT["nn_hyperparameters"]
    s3.run_networks(D, design_dir("euppbench", "spatial"), "spatial", folds, a.variants, [0] if SMOKE else range(K), a.seeds,
                    torch.device("cpu"), 1 if SMOKE else hp["max_epochs"])


def cmd_part_nets3(a):
    import stage3_run as s3
    D = s3.Data(str(EUPP_NPZ))
    design = f"part_{a.pid}"; fdf = folds_df("euppbench", design); folds = folds_dict(fdf); K = int(fdf.fold.max()) + 1
    assert set(folds) == set(int(s) for s in D.st.index)
    hp = s3.PROT["nn_hyperparameters"]
    s3.run_networks(D, design_dir("euppbench", design), "spatial", folds, a.variants, [0] if SMOKE else range(K), a.seeds,
                    torch.device("cpu"), 1 if SMOKE else hp["max_epochs"])


def cmd_part_bst(a):
    import stage3_run as s3
    D = s3.Data(str(EUPP_NPZ))
    design = f"part_{a.pid}"; fdf = folds_df("euppbench", design); folds = folds_dict(fdf); K = int(fdf.fold.max()) + 1
    if SMOKE:
        folds = {s: (f if f == 0 else 1) for s, f in folds.items()}; K = 1
    s3.run_baselines(D, design_dir("euppbench", design), "spatial", folds, K, 50 if SMOKE else 1000)


# ------------------------------------------------------------------------------ Stage-2 nets (German)
def cmd_seeds2(a):
    import stage2_run as s2
    from ppnn_residual.data import feature_columns, load_table
    out = design_dir("german", "spatial"); s2.OUT = out
    df = load_table(str(ROOT / s2.PROT["data"]["path"]), s2.PROT["data"]["drop_columns"])
    cols = feature_columns(df)
    folds, K = s2.get_folds(df, "spatial")
    assert folds == folds_dict(folds_df("german", "spatial")), "German spatial folds do not reproduce"
    if SMOKE:
        s2.PROT["nn_hyperparameters"]["max_epochs"] = 1
    s2.run_networks(df, cols, "spatial", folds, a.variants, [0] if SMOKE else range(K), a.seeds, a.threads)


# ------------------------------------------------------------------------------ Stage-4 methods
def run_stage4(c, fdf, out, ftype, methods, seeds, method_table_fn, fold_cls, tag_suffix=""):
    import stage4_run as s4
    K = int(fdf.fold.max()) + 1
    M = method_table_fn(methods)
    for k in ([0] if SMOKE else range(K)):
        todo = [(m, s) for m in methods for s in seeds
                if not (out / "logs" / ftype / f"{m}{tag_suffix}_f{k}_s{s}.json").exists()]
        if not todo:
            continue
        fd = fold_cls(c, fdf, k)
        for m, s in todo:
            fn, hp = M[m]; hp = dict(hp); t0 = time.time()
            if SMOKE:
                hp["max_epochs"] = 1
            mu, sg, info = fn(fd, s, hp)
            name = f"{m}{tag_suffix}"
            savez_atomic(out / "preds" / ftype / f"{name}_f{k}_s{s}.npz", mu=mu.astype(np.float32), sigma=sg.astype(np.float32))
            info.update({"method": name, "fold": k, "seed": s, "sec": round(time.time() - t0, 1),
                         "threads": torch.get_num_threads(), "torch": torch.__version__, "n_heldout_test_rows": int(len(mu))})
            json_atomic(out / "logs" / ftype / f"{name}_f{k}_s{s}.json", info)
            print(ftype, name, "fold", k, "seed", s, "best_ep", info["best_epoch"], "val", round(info["best_val_crps"], 4),
                  "sec", info["sec"], flush=True)
        del fd


def cmd_stage4(a):
    import stage4_run as s4
    from ppnn_residual.stage4_data import load_cube
    c = load_cube(a.dataset)
    fdf = folds_df(a.dataset, a.design)
    ftype = a.design if a.design in ("random", "spatial") else "spatial"
    run_stage4(c, fdf, design_dir(a.dataset, a.design), ftype, a.methods, a.seeds, s4.method_table, s4.Fold)


# ------------------------------------------------------------------------------ new statistical baselines
def local_emos_params(ds, c):
    """Local EMOS per (station, lead), fitted once on the fit period (each station uses only its own data)."""
    from ppnn_residual.emos import fit_emos
    f = R5 / ds / "local_emos_params.npz"
    if f.exists():
        z = np.load(f); return z["P"]
    m = c.X[:, :, c.i_mean].astype(np.float64); s = c.ens_sd().astype(np.float64); y = c.y.astype(np.float64)
    ok = (c.split == 0)[:, None] & np.isfinite(y) & np.isfinite(m) & np.isfinite(s)
    t, n = np.nonzero(ok)
    li = np.searchsorted(np.array(c.leads), c.lead[t])
    P = np.full((len(c.stations), len(c.leads), 4), np.nan)
    t0 = time.time()
    p, keys, info = fit_emos(m[t, n], s[t, n] ** 2, y[t, n], groups=n * 10 + li)
    for kk, pp in zip(keys, p):
        P[kk // 10, kk % 10] = [pp[0], pp[1], pp[2] ** 2, pp[3] ** 2]  # (a, b, c^2, d^2)
    savez_atomic(f, P=P)
    json_atomic(R5 / ds / "local_emos_info.json", {**info, "sec": round(time.time() - t0, 1),
                                                   "n_station_lead_missing": int(np.isnan(P[..., 0]).sum())})
    return P


def interp_params(P, st, tr_idx, q_idx):
    """Return dict method -> params [len(q), L, 4] for held-out stations q from training stations tr."""
    from ppnn_residual.stage2_models import haversine_km
    lat, lon, alt, orog = (st[c].to_numpy(np.float64) for c in ("lat", "lon", "alt", "orog"))
    A = np.c_[np.ones(len(lat)), lat, lon, alt, orog, alt - orog]
    d = haversine_km(lat[q_idx], lon[q_idx], lat[tr_idx], lon[tr_idx])
    d = np.sqrt(d ** 2 + (0.1 * (alt[q_idx][:, None] - alt[tr_idx][None])) ** 2)
    L = P.shape[1]
    out = {m: np.full((len(q_idx), L, 4), np.nan) for m in ("emos_nn1", "emos_idw", "emos_reg", "emos_regidw")}
    for li in range(L):
        ok = ~np.isnan(P[tr_idx, li, 0]); tri = tr_idx[ok]; Ptr = P[tri, li]; dd = d[:, ok]
        order = np.argsort(dd, 1)
        out["emos_nn1"][:, li] = Ptr[order[:, 0]]
        nn5 = order[:, :5]; w = 1.0 / np.maximum(np.take_along_axis(dd, nn5, 1), 1e-3); w /= w.sum(1, keepdims=True)
        idw = lambda V: (w[:, :, None] * V[nn5]).sum(1)
        out["emos_idw"][:, li] = idw(Ptr)
        Atr = A[tri]; mu_a, sd_a = Atr[:, 1:].mean(0), Atr[:, 1:].std(0); sd_a[sd_a == 0] = 1
        Z = lambda X: np.c_[X[:, :1], (X[:, 1:] - mu_a) / sd_a]
        beta, *_ = np.linalg.lstsq(Z(Atr), Ptr, rcond=None)
        trend_q = Z(A[q_idx]) @ beta; resid = Ptr - Z(Atr) @ beta
        out["emos_reg"][:, li] = trend_q
        out["emos_regidw"][:, li] = trend_q + idw(resid)
    for m in out:
        out[m][..., 2:] = np.maximum(out[m][..., 2:], 1e-6)
    return out


def cmd_statbase(a):
    from ppnn_residual.emos import fit_emos, predict_emos
    from ppnn_residual.stage4_data import load_cube
    from ppnn_residual.stage4_pilot import Climatology
    import stage4_run as s4  # noqa: F401  (protocol paths)
    PP = __import__("json").loads((ROOT / "protocols/stage4_pilot.json").read_text())["methods"]["samos_mlp"]
    c = load_cube(a.dataset)
    P = local_emos_params(a.dataset, c)
    fdf = folds_df(a.dataset, a.design); ftype = a.design if a.design in ("random", "spatial") else "spatial"
    out = design_dir(a.dataset, a.design); K = int(fdf.fold.max()) + 1
    fs = fdf.fold.reindex(c.stations).to_numpy()
    m_all = c.X[:, :, c.i_mean].astype(np.float64); s_all = c.ens_sd().astype(np.float64)
    avail = np.isfinite(c.y) & np.isfinite(m_all)
    leads = np.array(c.leads)
    methods = ["emos_nn1", "emos_idw", "emos_reg", "emos_regidw", "samos_lin"]
    for k in ([0] if SMOKE else range(K)):
        if all((out / "preds" / ftype / f"{mm}_f{k}.npz").exists() for mm in methods):
            continue
        t0 = time.time()
        tr = fs != k; held = fs == k
        tr_idx, q_idx = np.nonzero(tr)[0], np.nonzero(held)[0]
        rows = np.nonzero(held[c.test_n])[0]; tt, nn_ = c.test_t[rows], c.test_n[rows]
        li = np.searchsorted(leads, c.lead[tt]); qpos = np.searchsorted(q_idx, nn_)
        mm_, ss_ = m_all[tt, nn_], s_all[tt, nn_]
        info = {"fold": k, "n_train_st": int(len(tr_idx)), "n_held_st": int(len(q_idx)), "n_rows": int(len(rows))}
        IP = interp_params(P, c.st, tr_idx, q_idx)
        for name, Q in IP.items():
            q = Q[qpos, li]
            mu = q[:, 0] + q[:, 1] * mm_; sg = np.sqrt(q[:, 2] + q[:, 3] * ss_ ** 2)
            savez_atomic(out / "preds" / ftype / f"{name}_f{k}.npz", mu=mu.astype(np.float32), sigma=sg.astype(np.float32))
        # linear SAMOS
        clim = Climatology(c, tr, PP["clim_ridge"])
        fit = avail & (c.split == 0)[:, None] & tr[None]
        t, n = np.nonzero(fit)
        rng = np.random.default_rng(0)
        if len(t) > PP["clim_max_rows"]:
            sel = rng.choice(len(t), PP["clim_max_rows"], replace=False); tc, nc = t[sel], n[sel]
        else:
            tc, nc = t, n
        m_y, s_y = clim.fit_mean_sd(tc, nc, c.y[tc, nc].astype(np.float64))
        m_f, s_f = clim.fit_mean_sd(tc, nc, m_all[tc, nc])
        m_y, s_y, m_f, s_f = (x.astype(np.float64) for x in (m_y, s_y, m_f, s_f))
        zy = (c.y[t, n] - m_y[t, n]) / s_y[t, n]; zf = (m_all[t, n] - m_f[t, n]) / s_f[t, n]; zs = s_all[t, n] / s_f[t, n]
        p, keys, inf = fit_emos(zf, zs ** 2, zy, groups=c.lead[t])
        zf_q = (mm_ - m_f[tt, nn_]) / s_f[tt, nn_]; zs_q = ss_ / s_f[tt, nn_]
        zmu, zsg, _ = predict_emos(p, keys, zf_q, zs_q ** 2, groups=c.lead[tt], fallback=p[0])
        mu = m_y[tt, nn_] + s_y[tt, nn_] * zmu; sg = s_y[tt, nn_] * zsg
        savez_atomic(out / "preds" / ftype / "samos_lin_f{}.npz".format(k), mu=mu.astype(np.float32), sigma=sg.astype(np.float32))
        info["samos_lin"] = {"params_per_lead": {int(kk): pp.tolist() for kk, pp in zip(keys, p)}, **inf, "clim_rows": int(len(tc))}
        info["sec"] = round(time.time() - t0, 1)
        json_atomic(out / "logs" / ftype / f"statbase_f{k}.json", info)
        print(a.dataset, a.design, "statbase fold", k, "sec", info["sec"], flush=True)


# ------------------------------------------------------------------------------ boosting iteration sensitivity
def cmd_bstsens(a):
    import stage3_run as s3
    from ppnn_residual.boosting import BoostedEMOS
    from ppnn_residual.metrics import crps_gaussian_np
    D = s3.Data(str(EUPP_NPZ))
    folds, K = s3.get_folds(D, a.fold_type)
    assert folds == folds_dict(folds_df("euppbench", a.fold_type))
    out = design_dir("euppbench", a.fold_type) / "bstsens"
    MAXIT = 200 if SMOKE else 4000
    M_REP = [m for m in [100, 250, 500, 1000, 2000, 4000] if m <= MAXIT]; GRID = np.arange(0, MAXIT + 1, 50)
    tr, te = D.tr, D.te
    Ztr = D.design(tr, tr["X"]); keep = [i for i in range(Ztr.shape[1]) if not (34 <= i < 34 + len(s3.LEADS))]
    Ztr = Ztr[:, keep]; Zte = D.design(te, te["X"])[:, keep]
    for k in ([0] if SMOKE else range(K)):
        f = out / f"emos_bst_gl_path_f{k}.npz"
        if f.exists():
            continue
        t0 = time.time()
        fit, val, held = s3.split(D, folds, k)
        cm = np.nanmean(Ztr[fit], 0)
        fill = lambda Z: np.where(np.isnan(Z), cm, Z)
        res = {f"mu_m{m}": np.full(held.sum(), np.nan) for m in M_REP + ["aic", "val"]}
        res.update({f"sigma_m{m}": np.full(held.sum(), np.nan) for m in M_REP + ["aic", "val"]})
        info = {"fold": k}
        hrows = np.nonzero(held)[0]
        for L in s3.LEADS:
            r = fit & (tr["lead"] == L); rv = val & (tr["lead"] == L)
            b = BoostedEMOS(nu=0.05, maxit=MAXIT).fit(fill(Ztr[r]), tr["y"][r])
            Zv = fill(Ztr[rv])
            vcrps = np.array([crps_gaussian_np(*b.predict_at(Zv, int(m)), tr["y"][rv]).mean() for m in GRID])
            m_val = int(GRID[int(np.argmin(vcrps))])
            aic_m1000 = int(np.argmin(b.aic[:1001]))
            sel = te["lead"][hrows] == L; rr = hrows[sel]
            for m in M_REP + ["aic", "val"]:
                mm = {"aic": b.mstop, "val": m_val}.get(m, m)
                mu, sg = b.predict_at(fill(Zte[rr]), int(mm))
                res[f"mu_m{m}"][sel], res[f"sigma_m{m}"][sel] = mu, sg
            info[int(L)] = {"mstop_aic_4000": int(b.mstop), "mstop_aic_1000": aic_m1000, "m_val": m_val,
                            "val_crps_grid": dict(zip(GRID.tolist(), vcrps.round(6).tolist()))}
            print("bstsens", a.fold_type, "fold", k, "lead", L, info[int(L)]["mstop_aic_4000"], m_val, flush=True)
        savez_atomic(f, **{kk: v.astype(np.float32) for kk, v in res.items()}, rows=hrows)
        info["sec"] = round(time.time() - t0, 1)
        json_atomic(out / f"bstsens_f{k}.json", info)


# ------------------------------------------------------------------------------ m1a: neighbour availability = forecast availability
def fcavail_cube():
    """EUPPBench cube where forecast features of missing-observation (slice, station) pairs are filled in."""
    from ppnn_residual.stage4_data import eupp_cube
    c = eupp_cube()
    z = np.load(ROOT / "data/stage3/stage3_euppbench.npz", allow_pickle=False)
    sc = np.load(R5 / "euppbench/fcavail_extra.npz")
    stations = c.stations
    keys = []
    for j, p in enumerate(("train", "test")):
        keep = np.isin(z[f"{p}_station"], stations)
        init = z[f"{p}_init"][keep].astype("datetime64[D]").astype(np.int64)
        keys.append((j * 10 ** 7 + init) * 1000 + z[f"{p}_lead"][keep].astype(np.int64))
    uk = np.unique(np.concatenate(keys))
    kx = (sc["part"].astype(np.int64) * 10 ** 7 + sc["init"].astype(np.int64)) * 1000 + sc["lead"].astype(np.int64)
    t = np.searchsorted(uk, kx); ok = (t < len(uk)) & (uk[np.minimum(t, len(uk) - 1)] == kx)
    n = np.searchsorted(stations, sc["station"])
    ok &= np.isin(sc["station"], stations)
    assert np.isnan(c.y[t[ok], n[ok]]).all()
    c.X[t[ok], n[ok]] = sc["X"][ok]
    print("fcavail: filled", int(ok.sum()), "of", len(ok), "extra forecast rows (rest: slice absent)", flush=True)
    return c


def fold_fc_cls():
    import stage4_run as s4

    class FoldFC(s4.Fold):
        """Same as Fold (training rows, normalisation, scoring unchanged: obs available); neighbour availability
        = forecast availability."""
        def __init__(self, c, fdf, k):
            super().__init__(c, fdf, k)
            self.av_obs = self.av
            self.av = torch.from_numpy(np.isfinite(c.X[:, :, c.i_mean]) & np.isfinite(c.X[:, :, c.i_sd]))
            self.n_nb_changed_info = None
    return FoldFC


def train_gnn_fc(fd, seed, hp):
    """stage4_run.train_gnn with aggregation over forecast-available neighbours (fd.av) and loss on observed rows
    (fd.av_obs). Otherwise identical."""
    import stage4_run as s4
    from ppnn_residual.metrics import crps_gaussian_torch
    torch.manual_seed(seed); gen = torch.Generator().manual_seed(seed)
    c = fd.c
    adj_all = torch.from_numpy(((fd.D <= hp["radius_km"]) & ~np.eye(len(c.stations), dtype=bool)).astype(np.float32))
    tri = torch.from_numpy(fd.tr_idx); adj_tr = adj_all[tri][:, tri]
    model = s4.SAGE(fd.n_feat, hp["hidden"], hp["dropout"]); opt = torch.optim.Adam(model.parameters(), lr=hp["lr"])
    fit_t = torch.from_numpy(np.nonzero(c.split == 0)[0]); val_t = torch.from_numpy(np.nonzero(c.split == 1)[0])

    def run(tb, nodes, adj):
        x = fd.node_feats(tb, nodes); av_nb = fd.av[tb][:, nodes].float(); av = fd.av_obs[tb][:, nodes].float()
        mu, sg = model(x, av_nb, adj)
        return mu, sg, crps_gaussian_torch(mu, sg, fd.y[tb][:, nodes]), av

    best, state, best_ep, bad, hist = np.inf, None, 0, 0, []
    B = hp["batch_slices"]
    for ep in range(1, hp["max_epochs"] + 1):
        model.train(); t0 = time.time(); perm = fit_t[torch.randperm(len(fit_t), generator=gen)]; tot = 0.0; nn_ = 0.0
        for i in range(0, len(perm), B):
            _, _, cr, av = run(perm[i:i + B], tri, adj_tr)
            cr = torch.where(av > 0, cr, torch.zeros_like(cr))
            loss = (cr * av).sum() / av.sum()
            opt.zero_grad(); loss.backward(); opt.step(); tot += float((cr * av).sum().detach()); nn_ += float(av.sum())
        model.eval(); vs, vn = 0.0, 0.0
        with torch.no_grad():
            for i in range(0, len(val_t), B):
                _, _, cr, av = run(val_t[i:i + B], tri, adj_tr)
                cr = torch.where(av > 0, cr, torch.zeros_like(cr))
                vs += float((cr.double() * av).sum()); vn += float(av.sum())
        v = vs / vn
        hist.append({"epoch": ep, "train_crps": tot / nn_, "val_crps": v, "sec": round(time.time() - t0, 2)})
        if v < best - 1e-6:
            best, best_ep, bad = v, ep, 0; state = {a_: b_.clone() for a_, b_ in model.state_dict().items()}
        else:
            bad += 1
            if bad >= hp["patience"]:
                break
    model.load_state_dict(state); model.eval()
    test_t = np.nonzero(c.split == 2)[0]; alln = torch.arange(len(c.stations))
    MU = np.full((len(c.split), len(c.stations)), np.nan, np.float32); SG = MU.copy()
    with torch.no_grad():
        for i in range(0, len(test_t), B):
            tb = torch.from_numpy(test_t[i:i + B]); mu, sg, _, _ = run(tb, alln, adj_all)
            MU[tb.numpy()] = mu.numpy(); SG[tb.numpy()] = sg.numpy()
    r = fd.test_rows
    # how many held-out test cases see a different neighbour set than with observation-based availability
    tt, nn2 = c.test_t[r], c.test_n[r]
    A = adj_all.numpy()[nn2] > 0
    changed = ((fd.av.numpy()[tt] != fd.av_obs.numpy()[tt]) & A).any(1)
    return MU[tt, nn2], SG[tt, nn2], {"best_epoch": best_ep, "best_val_crps": best, "history": hist,
                                       "frac_heldout_test_cases_neighbour_set_changed": float(changed.mean())}


def cmd_m1a(a):
    import stage4_run as s4
    c = fcavail_cube()
    FoldFC = fold_fc_cls()

    def mt(requested):
        M = s4.method_table(requested)
        M["gnn_geo"] = (train_gnn_fc, M["gnn_geo"][1])
        return M
    fdf = folds_df("euppbench", a.fold_type)
    run_stage4(c, fdf, design_dir("euppbench", "m1a"), a.fold_type, a.methods, a.seeds, mt, FoldFC, tag_suffix="_fcavail")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--dataset", default="euppbench")
    ap.add_argument("--design", default="spatial")
    ap.add_argument("--pid", default=None)
    ap.add_argument("--fold-type", default="spatial")
    ap.add_argument("--variants", nargs="*", default=["unk", "hybrid_res"])
    ap.add_argument("--methods", nargs="*", default=[])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--threads", type=int, default=3)
    a = ap.parse_args()
    set_threads(a.threads)
    print("stage5", " ".join(sys.argv[1:]), "threads", torch.get_num_threads(), time.strftime("%Y-%m-%dT%H:%M:%S%z"), flush=True)
    globals()[f"cmd_{a.cmd}"](a)
    print("done", time.strftime("%Y-%m-%dT%H:%M:%S%z"), flush=True)


if __name__ == "__main__":
    main()
