"""EXPLORATORY (not pre-registered) diagnosis of hybrid vs hybrid_res on Stage 2 (German) and Stage 3 (EUPPBench).
Uses only existing predictions in results/stage{2,3}/preds. No training.
    python scripts/stage4_diagnose.py
Outputs: results/stage4_analysis/*.csv|json, figures/stage4_analysis/*.png
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import Delaunay

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ppnn_residual.metrics import crps_gaussian_np, pit  # noqa: E402
from ppnn_residual.stage2_models import haversine_km  # noqa: E402

OUT = ROOT / "results/stage4_analysis"; FIG = ROOT / "figures/stage4_analysis"
OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
METHODS = ["raw_ensemble_gauss", "emos_bst_gl", "nn_unk", "nn_knn", "nn_noemb", "nn_attr", "nn_hybrid", "nn_hybrid_res"]
BASE = {"raw_ensemble_gauss", "emos_bst_gl", "emos_gl"}
SEEDS = {"random": range(10), "spatial": range(3)}


def load(ds, ftype):
    """Return row table + dict method -> (mu, sigma) seed-ensemble held-out predictions aligned to test rows."""
    R = ROOT / f"results/{ds}"
    meta = np.load(R / "test_meta.npz", allow_pickle=True)
    folds = pd.read_csv(R / f"folds_{ftype}.csv", index_col=0)
    st = meta["station"].astype(int)
    rf = folds.fold.reindex(st).to_numpy()
    rows = pd.DataFrame({"station": st, "y": meta["y"].astype(float), "fold": rf,
                         "lead": meta["lead"] if "lead" in meta else 48})
    if "init" in meta:
        rows["month"] = pd.to_datetime(meta["init"].astype("datetime64[D]")).month
    else:
        rows["month"] = pd.to_datetime(meta["date"]).month
    K = folds.fold.max() + 1
    P = {}
    for m in METHODS:
        mu = np.full(len(st), np.nan); sg = np.full(len(st), np.nan)
        for k in range(K):
            r = rf == k
            if m in BASE:
                z = np.load(R / "preds" / ftype / f"{m}_f{k}.npz"); mu[r], sg[r] = z["mu"][r], z["sigma"][r]
            else:
                ms, ss = [], []
                for s in SEEDS[ftype]:
                    z = np.load(R / "preds" / ftype / f"{m}_f{k}_s{s}.npz")
                    a, b = z["mu"].astype(float), z["sigma"].astype(float)
                    if len(a) == len(st):  # stage 2 stores all rows
                        a, b = a[r], b[r]
                    ms.append(a); ss.append(b)
                mu[r], sg[r] = np.mean(ms, 0), np.mean(ss, 0)
        P[m] = (mu, sg)
    return rows, folds, P


def station_features(folds):
    """Per held-out station: distance/density/attribute-extrapolation w.r.t. its fold's training stations."""
    A = folds.assign(alt_minus_orog=folds.alt - folds.orog)[["lat", "lon", "alt", "orog", "alt_minus_orog"]]
    out = []
    for k in sorted(folds.fold.unique()):
        h = folds[folds.fold == k]; t = folds[folds.fold != k]
        d = haversine_km(h.lat.values, h.lon.values, t.lat.values, t.lon.values)
        At = A.loc[t.index]; mu, sd = At.mean(), At.std(ddof=0)
        Zh, Zt = ((A.loc[h.index] - mu) / sd).values, ((At - mu) / sd).values
        nov = np.sqrt(((Zh[:, None, :] - Zt[None]) ** 2).sum(-1)).min(1)  # nearest training station in attribute space
        hull = Delaunay(t[["lon", "lat"]].values)
        for i, s in enumerate(h.index):
            r = {"station": s, "fold": k, "lat": h.lat[s], "lon": h.lon[s], "alt": h.alt[s],
                 "dist_nearest_km": d[i].min(), "n_train_50km": int((d[i] <= 50).sum()), "n_train_100km": int((d[i] <= 100).sum()),
                 "attr_novelty": nov[i], "outside_latlon_hull": bool(hull.find_simplex([[h.lon[s], h.lat[s]]])[0] < 0)}
            for c in ["alt", "orog", "alt_minus_orog", "lat", "lon"]:
                v = A.loc[s, c]; lo, hi = At[c].min(), At[c].max()
                r[f"{c}_excess"] = max(0.0, v - hi, lo - v)
            r["any_attr_outside_range"] = any(r[f"{c}_excess"] > 0 for c in ["alt", "orog", "alt_minus_orog", "lat", "lon"])
            out.append(r)
    return pd.DataFrame(out).set_index("station")


def main():
    summ, bias_rows, lead_rows, swap_rows, conc_rows, corr_rows, pit_store = [], [], [], [], [], [], {}
    station_tabs = {}
    for ds, dname in (("stage2", "German RL18"), ("stage3", "EUPPBench")):
        for ftype in ("random", "spatial"):
            rows, folds, P = load(ds, ftype)
            y = rows.y.values
            C = {m: crps_gaussian_np(*P[m], y) for m in P}
            ens_mean = P["raw_ensemble_gauss"][0]
            key = f"{ds}_{ftype}"
            # --- swap decomposition: which part of hybrid_res (mean vs spread) matters?
            (mh, sh), (mr, sr) = P["nn_hybrid"], P["nn_hybrid_res"]
            sw = {"mu_hybrid+sigma_hybrid": crps_gaussian_np(mh, sh, y).mean(), "mu_res+sigma_hybrid": crps_gaussian_np(mr, sh, y).mean(),
                  "mu_hybrid+sigma_res": crps_gaussian_np(mh, sr, y).mean(), "mu_res+sigma_res": crps_gaussian_np(mr, sr, y).mean()}
            swap_rows.append({"dataset": ds, "folds": ftype, **sw})
            # --- bias / spread / calibration per method
            for m in P:
                mu, sg = P[m]; p = pit(mu, sg, y)
                bias_rows.append({"dataset": ds, "folds": ftype, "method": m, "crps": C[m].mean(), "bias": (mu - y).mean(),
                                  "mean_abs_station_bias": pd.Series(mu - y).groupby(rows.station.values).mean().abs().mean(),
                                  "rmse": np.sqrt(((mu - y) ** 2).mean()), "mean_sigma": sg.mean(),
                                  "cov80": ((p >= .1) & (p <= .9)).mean(), "mean_abs_correction_vs_ensmean": np.abs(mu - ens_mean).mean()})
                if m in ("nn_hybrid", "nn_hybrid_res", "nn_unk"):
                    pit_store[(key, m)] = np.histogram(p, 10, (0, 1))[0] / len(p)
            # --- lead time
            for L, g in rows.groupby("lead"):
                ix = g.index.values
                lead_rows.append({"dataset": ds, "folds": ftype, "lead": L, **{m: C[m][ix].mean() for m in P}})
            # --- per station
            ps = pd.DataFrame({m: C[m] for m in P}).groupby(rows.station.values).mean()
            bias_h = pd.Series(mh - y).groupby(rows.station.values).mean(); bias_r = pd.Series(mr - y).groupby(rows.station.values).mean()
            ps["bias_hybrid"], ps["bias_hybrid_res"] = bias_h, bias_r
            ps["bias_raw"] = pd.Series(ens_mean - y).groupby(rows.station.values).mean()
            ps["sigma_hybrid"] = pd.Series(sh).groupby(rows.station.values).mean(); ps["sigma_hybrid_res"] = pd.Series(sr).groupby(rows.station.values).mean()
            ps["n_rows"] = rows.groupby("station").size()
            ps = ps.join(station_features(folds))
            ps["d_hyb_minus_res"] = ps.nn_hybrid - ps.nn_hybrid_res
            ps["contrib"] = ps.d_hyb_minus_res * ps.n_rows / ps.n_rows.sum()  # contribution to pooled CRPS difference
            ps.to_csv(OUT / f"per_station_{key}.csv", float_format="%.5f"); station_tabs[key] = ps
            tot = ps.contrib.sum(); srt = ps.contrib.sort_values(ascending=False)
            def pooled_diff(mask):
                w = ps.n_rows[mask]; return float((ps.d_hyb_minus_res[mask] * w).sum() / w.sum()) if mask.any() else np.nan
            conc_rows.append({"dataset": ds, "folds": ftype, "n_stations": len(ps), "pooled_diff_hyb_minus_res": tot,
                              "top1_share": srt.iloc[:1].sum() / tot, "top5_share": srt.iloc[:5].sum() / tot,
                              "frac_stations_res_better": float((ps.d_hyb_minus_res > 0).mean()),
                              "median_station_diff": float(ps.d_hyb_minus_res.median()),
                              "diff_excl_top5": pooled_diff(~ps.index.isin(srt.index[:5])),
                              "n_attr_outside_range": int(ps.any_attr_outside_range.sum()),
                              "diff_attr_outside_range": pooled_diff(ps.any_attr_outside_range.values),
                              "diff_attr_inside_range": pooled_diff(~ps.any_attr_outside_range.values),
                              "n_alt_outside_range": int((ps.alt_excess > 0).sum()),
                              "diff_alt_outside_range": pooled_diff((ps.alt_excess > 0).values),
                              "n_outside_latlon_hull": int(ps.outside_latlon_hull.sum()),
                              "diff_outside_hull": pooled_diff(ps.outside_latlon_hull.values),
                              "diff_inside_hull": pooled_diff(~ps.outside_latlon_hull.values),
                              "n_isolated_50km": int((ps.n_train_50km == 0).sum()),
                              "diff_isolated_50km": pooled_diff((ps.n_train_50km == 0).values),
                              "diff_not_isolated_50km": pooled_diff((ps.n_train_50km > 0).values),
                              "top5_stations": ";".join(f"{s}(alt {ps.alt[s]:.0f}m, d {ps.dist_nearest_km[s]:.0f}km, dCRPS {ps.d_hyb_minus_res[s]:+.2f})" for s in srt.index[:5])})
            for f in ["dist_nearest_km", "n_train_50km", "attr_novelty", "alt_excess", "alt"]:
                r = stats.spearmanr(ps[f], ps.d_hyb_minus_res)
                corr_rows.append({"dataset": ds, "folds": ftype, "feature": f, "spearman_rho": r.statistic, "p": r.pvalue})
            summ.append((ds, ftype, dname))
    pd.DataFrame(swap_rows).to_csv(OUT / "swap_decomposition.csv", index=False, float_format="%.5f")
    pd.DataFrame(bias_rows).to_csv(OUT / "bias_spread.csv", index=False, float_format="%.5f")
    pd.DataFrame(lead_rows).to_csv(OUT / "per_lead.csv", index=False, float_format="%.5f")
    pd.DataFrame(conc_rows).to_csv(OUT / "concentration.csv", index=False, float_format="%.5f")
    pd.DataFrame(corr_rows).to_csv(OUT / "correlations.csv", index=False, float_format="%.4f")
    # --- figures
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    keys = [f"{d}_{f}" for d in ("stage2", "stage3") for f in ("random", "spatial")]
    title = {"stage2": "German", "stage3": "EUPPBench"}
    fig, axs = plt.subplots(2, 4, figsize=(17, 8))
    for j, k in enumerate(keys):
        ps = station_tabs[k]; d, f = k.split("_")
        c = np.where(ps.any_attr_outside_range, "r", "k")
        axs[0, j].scatter(ps.attr_novelty, ps.d_hyb_minus_res, c=c, s=12)
        axs[0, j].axhline(0, color="grey", lw=.8); axs[0, j].set_xlabel("attribute novelty (std. units)")
        axs[0, j].set_ylabel("CRPS hybrid − hybrid_res"); axs[0, j].set_title(f"{title[d]}, {f} (red: attr outside train range)", fontsize=8)
        axs[1, j].scatter(ps.dist_nearest_km, ps.d_hyb_minus_res, c=c, s=12); axs[1, j].axhline(0, color="grey", lw=.8)
        axs[1, j].set_xlabel("distance to nearest training station (km)"); axs[1, j].set_ylabel("CRPS hybrid − hybrid_res")
        for s in ps.d_hyb_minus_res.abs().sort_values().index[-3:]:
            axs[0, j].annotate(f"{s} ({ps.alt[s]:.0f} m)", (ps.attr_novelty[s], ps.d_hyb_minus_res[s]), fontsize=6)
    fig.tight_layout(); fig.savefig(FIG / "station_diff_vs_novelty_distance.png", dpi=140); plt.close(fig)
    fig, axs = plt.subplots(1, 4, figsize=(16, 3.6))
    for j, k in enumerate(keys):
        ps = station_tabs[k]; srt = ps.contrib.sort_values(ascending=False)
        axs[j].plot(np.arange(1, len(srt) + 1), srt.cumsum() / srt.sum()); axs[j].axhline(1, color="grey", lw=.8)
        axs[j].set_xlabel("stations (sorted by contribution)"); axs[j].set_ylabel("cum. share of pooled Δ"); axs[j].set_title(k, fontsize=9)
        axs[j].set_xscale("log")
    fig.tight_layout(); fig.savefig(FIG / "concentration_curves.png", dpi=140); plt.close(fig)
    fig, axs = plt.subplots(3, 4, figsize=(15, 8), sharey=True)
    for j, k in enumerate(keys):
        for i, m in enumerate(["nn_unk", "nn_hybrid", "nn_hybrid_res"]):
            h = pit_store[(k, m)]; axs[i, j].bar(np.arange(10) / 10, h, width=.1, align="edge", edgecolor="k")
            axs[i, j].axhline(.1, color="r", ls="--"); axs[i, j].set_title(f"{k}: {m}", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "pit_hybrid_vs_res.png", dpi=140); plt.close(fig)
    lr = pd.DataFrame(lead_rows); lr = lr[lr.dataset == "stage3"]
    fig, ax = plt.subplots(figsize=(6, 4))
    for f, g in lr.groupby("folds"):
        ax.plot(g.lead, g.nn_hybrid - g.nn_hybrid_res, marker="o", label=f"{f}: hybrid − hybrid_res")
        ax.plot(g.lead, g.nn_noemb - g.nn_unk, marker="x", ls="--", label=f"{f}: noemb − unk")
    ax.axhline(0, color="grey"); ax.set_xlabel("lead (h)"); ax.set_ylabel("ΔCRPS (°C)"); ax.legend(fontsize=7); ax.set_title("EUPPBench, ΔCRPS by lead")
    fig.tight_layout(); fig.savefig(FIG / "stage3_diff_by_lead.png", dpi=140); plt.close(fig)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_colwidth", 200)
    for f in ["swap_decomposition", "concentration", "correlations", "bias_spread", "per_lead"]:
        print("==", f); print(pd.read_csv(OUT / f"{f}.csv").round(4).to_string())


if __name__ == "__main__":
    main()
