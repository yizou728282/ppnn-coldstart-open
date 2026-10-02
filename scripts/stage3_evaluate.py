"""Evaluate Stage-3 (EUPPBench) predictions per protocols/stage3_euppbench.json.
All numbers are computed from <out>/preds/**.npz.   python scripts/stage3_evaluate.py --out results/stage3"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ppnn_residual.metrics import (benjamini_hochberg, block_bootstrap_ci, crps_gaussian_np,  # noqa: E402
                                   diebold_mariano, summary_metrics)
from ppnn_residual.stage2_models import haversine_km  # noqa: E402

PROT = json.loads((ROOT / "protocols/stage3_euppbench.json").read_text())
LEADS = PROT["data"]["lead_times_h"]
BASE = ["raw_ensemble_gauss", "emos_gl", "emos_bst_gl"]
NETS = ["nn_unk", "nn_knn", "nn_noemb", "nn_attr", "nn_hybrid", "nn_hybrid_res", "nn_hybrid_knn"]
PROPOSED = "nn_hybrid"


def load(out, ftype, name, K, seeds, row_fold, tag=""):
    """Held-out pooled (mu, sigma) for the primary forecast + per-seed pooled + seen-station ensemble per fold."""
    d = out / "preds" / ftype
    n = len(row_fold)
    if name in BASE:
        files = [d / f"{name}_f{k}.npz" for k in range(K)]
        if not all(f.exists() for f in files):
            return None
        mu, sg = np.empty(n), np.empty(n); seen = []
        for k, f in enumerate(files):
            z = np.load(f); m, s = z["mu" + tag].astype(np.float64), z["sigma" + tag].astype(np.float64)
            r = row_fold == k
            mu[r], sg[r] = m[r], s[r]
            seen.append((m[~r], s[~r]))
        return {"ens": (mu, sg), "seeds": [], "single": {}, "seen": seen}
    have = [s for s in seeds if all((d / f"{name}_f{k}_s{s}.npz").exists() for k in range(K))]
    if not have:
        return None
    single = {}
    for s in have:
        mu, sg = np.empty(n), np.empty(n)
        for k in range(K):
            z = np.load(d / f"{name}_f{k}_s{s}.npz"); r = row_fold == k
            mu[r], sg[r] = z["mu" + tag], z["sigma" + tag]
        single[s] = (mu, sg)
    ens = (np.mean([single[s][0] for s in have], 0), np.mean([single[s][1] for s in have], 0))
    seen = []
    for k in range(K):
        f = d / f"seen_{name}_f{k}.npz"
        if f.exists():
            z = np.load(f); ns = len(z["seeds"])
            seen.append((z["sum_mu"] / ns, z["sum_sigma"] / ns) if set(z["seeds"].tolist()) >= set(have) and ns == len(have) else None)
        else:
            seen.append(None)
    return {"ens": ens, "seeds": have, "single": single, "seen": seen}


def evaluate(out, ftype, meta, seeds):
    fold_tab = pd.read_csv(out / f"folds_{ftype}.csv", index_col=0)
    K = int(fold_tab["fold"].max()) + 1
    st = meta["station"]; y = meta["y"].astype(np.float64); lead = meta["lead"]; init = meta["init"]
    fmap = fold_tab["fold"].to_dict()
    row_fold = np.array([fmap[int(s)] for s in st])
    dist = {}
    for k in range(K):
        held = fold_tab[fold_tab.fold == k]; tr = fold_tab[fold_tab.fold != k]
        dm = haversine_km(held.lat.values, held.lon.values, tr.lat.values, tr.lon.values)
        dist.update({int(s): float(v) for s, v in zip(held.index, dm.min(1))})
    row_dist = np.array([dist[int(s)] for s in st])
    R = {m: load(out, ftype, m, K, seeds, row_fold) for m in BASE + NETS}
    R = {m: r for m, r in R.items() if r is not None}
    rows, perfold, perlead, crps, metrics = [], [], [], {}, {}
    for m, r in R.items():
        mu, sg = r["ens"]
        met = summary_metrics(mu, sg, y); met["spread_skill"] = float(np.sqrt((sg ** 2).mean()) / met["rmse"])
        metrics[m] = met; crps[m] = crps_gaussian_np(mu, sg, y)
        row = {"method": m, "n_rows": len(y), "n_seeds": len(r["seeds"]), "crps_primary": met["crps"]}
        if r["single"]:
            sc = [crps_gaussian_np(*r["single"][s], y).mean() for s in r["seeds"]]
            row["crps_single_seed_mean"] = float(np.mean(sc))
            row["crps_single_seed_sd"] = float(np.std(sc, ddof=1)) if len(sc) > 1 else np.nan
        if all(x is not None for x in r["seen"]):
            row["crps_seen_stations_mean_over_folds"] = float(np.mean(
                [crps_gaussian_np(r["seen"][k][0], r["seen"][k][1], y[row_fold != k]).mean() for k in range(K)]))
        for kk in ["mae", "rmse", "coverage_80", "coverage_90", "width_80", "pit_reliability_index", "spread_skill",
                   "bias_mean_minus_obs"]:
            row[kk] = met[kk]
        rows.append(row)
        for k in range(K):
            perfold.append({"method": m, "fold": k, "crps": float(crps[m][row_fold == k].mean())})
        for L in LEADS:
            perlead.append({"method": m, "lead": L, "crps": float(crps[m][lead == L].mean())})
    tab = pd.DataFrame(rows)
    for ref in ["raw_ensemble_gauss", "emos_gl"]:
        if ref in metrics:
            tab[f"crpss_vs_{ref}"] = 1 - tab["crps_primary"] / metrics[ref]["crps"]
    tab.to_csv(out / f"summary_{ftype}.csv", index=False, float_format="%.5f")
    pd.DataFrame(perfold).pivot(index="fold", columns="method", values="crps").to_csv(out / f"per_fold_{ftype}.csv", float_format="%.5f")
    pd.DataFrame(perlead).pivot(index="lead", columns="method", values="crps").to_csv(out / f"per_lead_{ftype}.csv", float_format="%.5f")
    json.dump(metrics, open(out / f"metrics_{ftype}.json", "w"), indent=1)
    # 11-member sensitivity
    sens = []
    for m in R:
        r11 = load(out, ftype, m, K, seeds, row_fold, tag="11")
        if r11 is not None:
            sens.append({"method": m, "crps_51_members": float(crps[m].mean()),
                         "crps_11_members": float(crps_gaussian_np(*r11["ens"], y).mean())})
    pd.DataFrame(sens).to_csv(out / f"sensitivity_11members_{ftype}.csv", index=False, float_format="%.5f")
    # distance bins
    sd = pd.Series(dist); edges = np.unique(np.quantile(sd.values, [0, .2, .4, .6, .8, 1]))
    b = np.clip(np.searchsorted(edges, row_dist, side="right") - 1, 0, len(edges) - 2)
    drows = []
    for i in range(len(edges) - 1):
        msk = b == i
        rr = {"bin": i, "dist_km_lo": edges[i], "dist_km_hi": edges[i + 1], "n_rows": int(msk.sum()),
              "n_stations": int(len(np.unique(st[msk])))}
        rr.update({m: float(crps[m][msk].mean()) for m in crps})
        drows.append(rr)
    drows = pd.DataFrame(drows); drows.to_csv(out / f"distance_bins_{ftype}.csv", index=False, float_format="%.4f")
    ps = pd.DataFrame({m: crps[m] for m in crps}); ps["station"] = st
    ps = ps.groupby("station").mean().join(fold_tab[["country", "lat", "lon", "alt", "fold"]])
    ps["dist_nearest_train_km"] = [dist[int(s)] for s in ps.index]
    ps.to_csv(out / f"per_station_{ftype}.csv", float_format="%.4f")
    # significance
    pairs = [tuple(p) for p in PROT["significance"]["primary_family"]]
    sec = [(PROPOSED, m) for m in ["raw_ensemble_gauss", "nn_hybrid_res", "nn_hybrid_knn"]] + \
          [("nn_noemb", "emos_gl"), ("nn_noemb", "emos_bst_gl"), ("nn_unk", "nn_noemb"), ("nn_knn", "nn_unk"),
           ("nn_attr", "nn_noemb"), ("emos_bst_gl", "emos_gl")]
    srows = []
    for fam, plist in [("primary", pairs), ("secondary", sec)]:
        for a, bb in plist:
            if a not in crps or bb not in crps:
                continue
            dfd = pd.DataFrame({"d": init, "a": crps[a], "b": crps[bb]}).groupby("d")[["a", "b"]].mean().sort_index()
            dm = diebold_mariano(dfd.a.values, dfd.b.values, 5)
            bs = block_bootstrap_ci((dfd.a - dfd.b).values, 7, 10000)
            sdf = pd.DataFrame({"st": st, "d": init, "a": crps[a], "b": crps[bb]}).groupby(["st", "d"])[["a", "b"]].mean().reset_index()
            ps_, sg_ = [], []
            for _, g in sdf.groupby("st"):
                g = g.sort_values("d")
                if len(g) < 30:
                    continue
                rr = diebold_mariano(g.a.values, g.b.values, 1); ps_.append(rr["p_value"]); sg_.append(np.sign(rr["mean_diff"]))
            ps_, sg_ = np.array(ps_), np.array(sg_); rej = benjamini_hochberg(ps_, 0.05)
            srows.append({"family": fam, "a": a, "b": bb, "crps_a": crps[a].mean(), "crps_b": crps[bb].mean(),
                          "diff_a_minus_b": crps[a].mean() - crps[bb].mean(),
                          "rel_diff_pct": 100 * (crps[a].mean() / crps[bb].mean() - 1), "n_days": len(dfd),
                          "dm_stat": dm["dm_stat"], "dm_p": dm["p_value"], "boot_lo": bs["ci_low"], "boot_hi": bs["ci_high"],
                          "n_stations": len(ps_), "frac_st_a_better_BH": float((rej & (sg_ < 0)).mean()),
                          "frac_st_a_worse_BH": float((rej & (sg_ > 0)).mean())})
    sig = pd.DataFrame(srows)
    if len(sig):
        prim = sig.family == "primary"; sig["bh_reject_q05"] = False
        sig.loc[prim, "bh_reject_q05"] = benjamini_hochberg(sig.loc[prim, "dm_p"].values, 0.05)
    sig.to_csv(out / f"significance_{ftype}.csv", index=False, float_format="%.6g")
    return tab, sig, drows, ps, metrics


def hypotheses(out, tabs, sigs):
    c = {ft: t.set_index("method")["crps_primary"].to_dict() for ft, t in tabs.items()}
    seen = {ft: t.set_index("method").get("crps_seen_stations_mean_over_folds", pd.Series(dtype=float)).to_dict() for ft, t in tabs.items()}
    nets = ["nn_unk", "nn_knn", "nn_noemb", "nn_attr", "nn_hybrid"]
    res = {}
    def chk(name, f):
        try:
            res[name] = bool(f())
        except KeyError as e:
            res[name] = f"not evaluable (missing {e})"
    chk("H1", lambda: all(c[ft][n] < c[ft]["emos_gl"] and c[ft][n] < c[ft]["emos_bst_gl"] for ft in c for n in nets))
    def h2():
        r = c["random"]; trio = [r["nn_hybrid"], r["nn_attr"], r["nn_noemb"]]
        return max(trio) / min(trio) - 1 <= 0.01 and all(v < r["nn_unk"] for v in trio)
    chk("H2", h2)
    chk("H3", lambda: c["spatial"]["nn_noemb"] < c["spatial"]["nn_attr"] and c["spatial"]["nn_noemb"] < c["spatial"]["nn_hybrid"])
    chk("H4", lambda: all(c[ft]["nn_knn"] > c[ft]["nn_unk"] for ft in c))
    chk("H5", lambda: seen["random"]["nn_hybrid"] < seen["random"]["nn_noemb"])
    # claim rule
    claims = {}
    if "random" in sigs and "spatial" in sigs and len(sigs["random"]):
        sr = sigs["random"][sigs["random"].family == "primary"].set_index("b")
        ss = sigs["spatial"][sigs["spatial"].family == "primary"].set_index("b")
        for b in sr.index:
            ok = bool(sr.loc[b, "bh_reject_q05"] and (sr.loc[b, "boot_hi"] < 0 or sr.loc[b, "boot_lo"] > 0)
                      and b in ss.index and np.sign(ss.loc[b, "diff_a_minus_b"]) == np.sign(sr.loc[b, "diff_a_minus_b"]))
            claims[f"nn_hybrid vs {b}"] = ("hybrid better" if sr.loc[b, "diff_a_minus_b"] < 0 else "hybrid worse") if ok else "no claim"
    json.dump({"hypotheses_supported": res, "claim_rule": claims}, open(out / "hypotheses_and_claims.json", "w"), indent=1)
    return res, claims


def figures(out, fig_dir, ftype, drows, ps, metrics):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir.mkdir(parents=True, exist_ok=True)
    show = [m for m in ["emos_gl", "emos_bst_gl", "nn_unk", "nn_knn", "nn_noemb", "nn_attr", "nn_hybrid", "nn_hybrid_res"] if m in drows.columns]
    fig, ax = plt.subplots(figsize=(7, 4.5)); mid = (drows.dist_km_lo + drows.dist_km_hi) / 2
    for m in show:
        ax.plot(mid, drows[m], marker="o", label=m, lw=2.5 if m == PROPOSED else 1.2)
    ax.set_xlabel("distance to nearest training station (km, quintile bins)"); ax.set_ylabel("mean CRPS 2017-18 (°C)")
    ax.legend(fontsize=7); ax.set_title(f"EUPPBench, {ftype} station folds"); fig.tight_layout()
    fig.savefig(fig_dir / f"crps_vs_distance_{ftype}.png", dpi=150); plt.close(fig)
    pit_show = [m for m in ["raw_ensemble_gauss"] + show if m in metrics]
    nc = 5; nr = int(np.ceil(len(pit_show) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(3 * nc, 2.7 * nr), squeeze=False)
    for ax, m in zip(axes.ravel(), pit_show):
        h = metrics[m]["pit_hist"]; ax.bar(np.arange(len(h)) / len(h), h, width=1 / len(h), align="edge", edgecolor="k")
        ax.axhline(0.1, color="r", ls="--"); ax.set_title(m, fontsize=8)
    for ax in axes.ravel()[len(pit_show):]:
        ax.axis("off")
    fig.suptitle(f"PIT, held-out stations ({ftype} folds)"); fig.tight_layout()
    fig.savefig(fig_dir / f"pit_heldout_{ftype}.png", dpi=150); plt.close(fig)
    if PROPOSED in ps and "emos_gl" in ps:
        fig, axs = plt.subplots(1, 2, figsize=(11, 5.5)); sk = 1 - ps[PROPOSED] / ps["emos_gl"]
        lim = np.nanquantile(np.abs(sk), 0.98)
        sc = axs[0].scatter(ps.lon, ps.lat, c=sk, cmap="RdBu", vmin=-lim, vmax=lim, s=25)
        plt.colorbar(sc, ax=axs[0], label=f"CRPSS {PROPOSED} vs emos_gl"); axs[0].set_title("held-out per-station skill")
        axs[1].scatter(ps.lon, ps.lat, c=ps.fold, cmap="tab10", s=25); axs[1].set_title(f"{ftype} folds")
        fig.tight_layout(); fig.savefig(fig_dir / f"map_{ftype}.png", dpi=150); plt.close(fig)


def reference(out, meta):
    f = out / "preds/reference/emos_loc_bst_seen.npz"
    if not f.exists():
        return None
    z = np.load(f); ok = np.isfinite(z["mu"])
    r = {"emos_loc_bst_seen_crps": float(crps_gaussian_np(z["mu"][ok], z["sigma"][ok], meta["y"][ok]).mean()),
         "n_rows": int(ok.sum())}
    json.dump(r, open(out / "reference_emos_loc_bst.json", "w"), indent=1)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results/stage3"))
    ap.add_argument("--fig", default=None)
    a = ap.parse_args()
    out = Path(a.out); fig = Path(a.fig) if a.fig else out / "figures"
    meta = np.load(out / "test_meta.npz")
    pd.set_option("display.width", 250)
    tabs, sigs = {}, {}
    for ftype in ["random", "spatial"]:
        if not (out / f"folds_{ftype}.csv").exists() or not (out / "preds" / ftype).exists():
            continue
        seeds = PROT["nn_hyperparameters"]["seeds_primary" if ftype == "random" else "seeds_secondary_spatial"]
        tab, sig, drows, ps, metrics = evaluate(out, ftype, meta, seeds)
        tabs[ftype], sigs[ftype] = tab, sig
        figures(out, fig, ftype, drows, ps, metrics)
        print(f"\n===== EUPPBench {ftype} folds: held-out stations, test 2017-2018 =====")
        print(tab.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print(sig.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    if tabs:
        print("\nhypotheses / claims:", hypotheses(out, tabs, sigs))
    print("reference:", reference(out, meta))


if __name__ == "__main__":
    main()
