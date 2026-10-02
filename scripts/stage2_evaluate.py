"""Evaluate stage-2 (new-station) predictions per protocols/stage2_new_station.json.
All numbers are computed from results/stage2/preds/**.npz."""
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

PROT = json.loads((ROOT / "protocols/stage2_new_station.json").read_text())
S2 = ROOT / "results/stage2"
FIG = ROOT / "figures/stage2"
BASE = ["raw_ensemble_gauss", "emos_gl", "emos_bst_gl"]
NETS = ["nn_unk", "nn_knn", "nn_noemb", "nn_attr", "nn_hybrid", "nn_hybrid_res", "nn_hybrid_knn"]
PROPOSED = "nn_hybrid"


def load_method(ftype, name, K, seeds):
    """Return dict version -> list over folds of (mu, sigma) arrays over all 2016 rows."""
    d = S2 / "preds" / ftype
    if name in BASE:
        files = [d / f"{name}_f{k}.npz" for k in range(K)]
        if not all(f.exists() for f in files):
            return None
        return {"det": [tuple(np.load(f)[x].astype(np.float64) for x in ("mu", "sigma")) for f in files]}
    have = [s for s in seeds if all((d / f"{name}_f{k}_s{s}.npz").exists() for k in range(K))]
    if not have:
        return None
    out = {}
    for s in have:
        out[f"s{s}"] = [tuple(np.load(d / f"{name}_f{k}_s{s}.npz")[x].astype(np.float64) for x in ("mu", "sigma"))
                        for k in range(K)]
    out["ens"] = [(np.mean([out[f"s{s}"][k][0] for s in have], 0), np.mean([out[f"s{s}"][k][1] for s in have], 0))
                  for k in range(K)]
    out["_seeds"] = have
    return out


def pooled(folds_pred, row_fold):
    mu = np.empty(len(row_fold)); sg = np.empty(len(row_fold))
    for k, (m, s) in enumerate(folds_pred):
        r = row_fold == k
        mu[r], sg[r] = m[r], s[r]
    return mu, sg


def evaluate(ftype, meta, seeds):
    fold_tab = pd.read_csv(S2 / f"folds_{ftype}.csv", index_col=0)
    K = int(fold_tab["fold"].max()) + 1
    st = meta["station"]; y = meta["y"].astype(np.float64)
    fmap = fold_tab["fold"].to_dict()
    row_fold = np.array([fmap[int(s)] for s in st])
    day = pd.to_datetime(meta["date"])
    # distance of each held-out station to nearest training station of its fold
    dist = {}
    for k in range(K):
        held = fold_tab[fold_tab.fold == k]; tr = fold_tab[fold_tab.fold != k]
        dmat = haversine_km(held.lat.values, held.lon.values, tr.lat.values, tr.lon.values)
        for s, dd in zip(held.index, dmat.min(1)):
            dist[int(s)] = float(dd)
    row_dist = np.array([dist[int(s)] for s in st])

    methods = {}
    for m in BASE + NETS:
        r = load_method(ftype, m, K, seeds)
        if r is not None:
            methods[m] = r
    rows, perfold, crps_ens, metrics = [], [], {}, {}
    for m, r in methods.items():
        main = "det" if m in BASE else "ens"
        mu, sg = pooled(r[main], row_fold)
        met = summary_metrics(mu, sg, y)
        met["spread_skill"] = float(np.sqrt((sg ** 2).mean()) / met["rmse"])
        metrics[m] = met
        crps_ens[m] = crps_gaussian_np(mu, sg, y)
        row = {"method": m, "n_rows": len(y), "n_seeds": 0 if m in BASE else len(r["_seeds"]),
               "crps_primary": met["crps"]}
        if m not in BASE:
            single = [crps_gaussian_np(*pooled(r[f"s{s}"], row_fold), y).mean() for s in r["_seeds"]]
            row["crps_single_seed_mean"] = float(np.mean(single))
            row["crps_single_seed_sd"] = float(np.std(single, ddof=1)) if len(single) > 1 else np.nan
        # seen-station CRPS (same models, training stations of each fold, averaged over folds)
        seen = [crps_gaussian_np(r[main][k][0][row_fold != k], r[main][k][1][row_fold != k], y[row_fold != k]).mean()
                for k in range(K)]
        row["crps_seen_stations_mean_over_folds"] = float(np.mean(seen))
        for kk in ["mae", "rmse", "coverage_80", "coverage_90", "width_80", "pit_reliability_index", "spread_skill",
                   "bias_mean_minus_obs"]:
            row[kk] = met[kk]
        rows.append(row)
        for k in range(K):
            msk = row_fold == k
            perfold.append({"method": m, "fold": k, "n_rows": int(msk.sum()),
                            "crps": float(crps_gaussian_np(r[main][k][0][msk], r[main][k][1][msk], y[msk]).mean())})
    tab = pd.DataFrame(rows)
    for ref in ["raw_ensemble_gauss", "emos_gl"]:
        if ref in metrics:
            tab[f"crpss_vs_{ref}"] = 1 - tab["crps_primary"] / metrics[ref]["crps"]
    tab.to_csv(S2 / f"summary_{ftype}.csv", index=False, float_format="%.5f")
    pd.DataFrame(perfold).pivot(index="fold", columns="method", values="crps").to_csv(
        S2 / f"per_fold_{ftype}.csv", float_format="%.5f")
    json.dump(metrics, open(S2 / f"metrics_{ftype}.json", "w"), indent=1)

    # distance bins (quintiles of the station distance distribution)
    sd = pd.Series(dist)
    edges = np.unique(np.quantile(sd.values, [0, .2, .4, .6, .8, 1]))
    b = np.clip(np.searchsorted(edges, row_dist, side="right") - 1, 0, len(edges) - 2)
    drows = []
    for i in range(len(edges) - 1):
        msk = b == i
        rr = {"bin": i, "dist_km_lo": edges[i], "dist_km_hi": edges[i + 1], "n_rows": int(msk.sum()),
              "n_stations": int(len(np.unique(st[msk])))}
        for m in crps_ens:
            rr[m] = float(crps_ens[m][msk].mean())
        drows.append(rr)
    pd.DataFrame(drows).to_csv(S2 / f"distance_bins_{ftype}.csv", index=False, float_format="%.4f")

    # per-station table
    ps = pd.DataFrame({m: crps_ens[m] for m in crps_ens}); ps["station"] = st
    ps = ps.groupby("station").mean()
    ps = ps.join(fold_tab[["lat", "lon", "alt", "fold"]]); ps["dist_nearest_train_km"] = [dist[int(s)] for s in ps.index]
    ps.to_csv(S2 / f"per_station_{ftype}.csv", float_format="%.4f")

    # significance
    ev = PROT["significance"]
    pairs = [tuple(p) for p in ev["primary_family"]]
    sec = [(PROPOSED, m) for m in ["raw_ensemble_gauss", "nn_hybrid_res", "nn_hybrid_knn"]] + \
          [("nn_noemb", "emos_gl"), ("nn_noemb", "emos_bst_gl"), ("nn_unk", "nn_noemb"), ("nn_knn", "nn_unk"),
           ("nn_attr", "nn_noemb"), ("emos_bst_gl", "emos_gl")]
    srows = []
    for fam, plist in [("primary", pairs), ("secondary", sec)]:
        for a, bb in plist:
            if a not in crps_ens or bb not in crps_ens:
                continue
            dfd = pd.DataFrame({"day": day, "a": crps_ens[a], "b": crps_ens[bb]}).groupby("day")[["a", "b"]].mean()
            dm = diebold_mariano(dfd.a.values, dfd.b.values, 5)
            bs = block_bootstrap_ci((dfd.a - dfd.b).values, 7, 10000)
            sdf = pd.DataFrame({"st": st, "day": day, "a": crps_ens[a], "b": crps_ens[bb]}).sort_values(["st", "day"])
            ps_, sg_ = [], []
            for _, g in sdf.groupby("st"):
                if len(g) < 30:
                    continue
                r = diebold_mariano(g.a.values, g.b.values, 1)
                ps_.append(r["p_value"]); sg_.append(np.sign(r["mean_diff"]))
            ps_, sg_ = np.array(ps_), np.array(sg_)
            rej = benjamini_hochberg(ps_, 0.05)
            srows.append({"family": fam, "a": a, "b": bb, "crps_a": crps_ens[a].mean(), "crps_b": crps_ens[bb].mean(),
                          "diff_a_minus_b": crps_ens[a].mean() - crps_ens[bb].mean(),
                          "rel_diff_pct": 100 * (crps_ens[a].mean() / crps_ens[bb].mean() - 1),
                          "dm_stat": dm["dm_stat"], "dm_p": dm["p_value"], "boot_lo": bs["ci_low"], "boot_hi": bs["ci_high"],
                          "n_stations": len(ps_), "frac_st_a_better_BH": float((rej & (sg_ < 0)).mean()),
                          "frac_st_a_worse_BH": float((rej & (sg_ > 0)).mean())})
    sig = pd.DataFrame(srows)
    if len(sig):
        prim = sig.family == "primary"
        sig["bh_reject_q05"] = False
        sig.loc[prim, "bh_reject_q05"] = benjamini_hochberg(sig.loc[prim, "dm_p"].values, 0.05)
    sig.to_csv(S2 / f"significance_{ftype}.csv", index=False, float_format="%.6g")
    return tab, sig, pd.DataFrame(drows), ps, metrics, row_fold


def figures(ftype, tab, drows, ps, metrics):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)
    show = [m for m in ["raw_ensemble_gauss", "emos_gl", "emos_bst_gl", "nn_unk", "nn_knn", "nn_noemb", "nn_attr",
                        "nn_hybrid", "nn_hybrid_res"] if m in drows.columns]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    mid = (drows.dist_km_lo + drows.dist_km_hi) / 2
    for m in show:
        if m == "raw_ensemble_gauss":
            continue
        ax.plot(mid, drows[m], marker="o", label=m, lw=2.5 if m == PROPOSED else 1.2)
    ax.set_xlabel("distance of held-out station to nearest training station (km, quintile bins)")
    ax.set_ylabel("mean CRPS 2016 (°C)"); ax.legend(fontsize=7); ax.set_title(f"{ftype} station folds")
    fig.tight_layout(); fig.savefig(FIG / f"crps_vs_distance_{ftype}.png", dpi=150); plt.close(fig)
    pit_show = [m for m in show if m in metrics]
    nc = 5; nr = int(np.ceil(len(pit_show) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(3 * nc, 2.7 * nr), squeeze=False)
    for ax, m in zip(axes.ravel(), pit_show):
        h = metrics[m]["pit_hist"]
        ax.bar(np.arange(len(h)) / len(h), h, width=1 / len(h), align="edge", edgecolor="k")
        ax.axhline(0.1, color="r", ls="--"); ax.set_title(m, fontsize=8)
    for ax in axes.ravel()[len(pit_show):]:
        ax.axis("off")
    fig.suptitle(f"PIT, held-out stations ({ftype} folds)"); fig.tight_layout()
    fig.savefig(FIG / f"pit_heldout_{ftype}.png", dpi=150); plt.close(fig)
    if PROPOSED in ps and "emos_gl" in ps:
        fig, axs = plt.subplots(1, 2, figsize=(11, 6))
        sk = 1 - ps[PROPOSED] / ps["emos_gl"]
        lim = np.nanquantile(np.abs(sk), 0.98)
        sc = axs[0].scatter(ps.lon, ps.lat, c=sk, cmap="RdBu", vmin=-lim, vmax=lim, s=14)
        plt.colorbar(sc, ax=axs[0], label=f"CRPSS {PROPOSED} vs emos_gl (held-out)")
        axs[0].set_title("per-station skill on held-out stations")
        sc2 = axs[1].scatter(ps.lon, ps.lat, c=ps.fold, cmap="tab10", s=14)
        axs[1].set_title(f"{ftype} fold assignment")
        for a in axs:
            a.set_xlabel("lon"); a.set_ylabel("lat")
        fig.tight_layout(); fig.savefig(FIG / f"map_{ftype}.png", dpi=150); plt.close(fig)


def reference():
    f = S2 / "preds/reference/emos_loc_bst_seen.npz"
    if not f.exists():
        return None
    z = np.load(f)
    ok = np.isfinite(z["mu"])
    c = crps_gaussian_np(z["mu"][ok], z["sigma"][ok], z["y"][ok])
    out = {"emos_loc_bst_seen_crps": float(c.mean()), "n_rows": int(ok.sum()), "n_rows_missing_unseen": int((~ok).sum()),
           "paper_reported_emos_loc_bst_2007_2015": 0.80}
    s1 = ROOT / "results/preds/emos_loc.npz"
    if s1.exists():
        e = np.load(s1); y = z["y"]
        out["stage1_emos_loc_same_rows_crps"] = float(crps_gaussian_np(e["mu"][ok], e["sigma"][ok], y[ok]).mean())
    json.dump(out, open(S2 / "reference_emos_loc_bst.json", "w"), indent=1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold-types", nargs="*", default=["random", "spatial"])
    a = ap.parse_args()
    meta = np.load(S2 / "test_meta.npz", allow_pickle=True)
    pd.set_option("display.width", 250)
    for ftype in a.fold_types:
        if not (S2 / f"folds_{ftype}.csv").exists():
            continue
        seeds = PROT["nn_hyperparameters"]["seeds_primary" if ftype == "random" else "seeds_secondary_spatial"]
        tab, sig, drows, ps, metrics, _ = evaluate(ftype, meta, seeds)
        figures(ftype, tab, drows, ps, metrics)
        print(f"\n===== {ftype} folds: held-out stations, 2016 =====")
        print(tab.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print(sig.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
        print(drows.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nreference:", reference())


if __name__ == "__main__":
    main()
