"""Evaluate saved 2016 test predictions: metrics, skill scores, significance tests, figures.
Every number written here is computed from results/preds/*.npz."""
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

# Rasp & Lerch (2018, MWR 146, Table 2), mean CRPS on 2016, training period 2007-2015, as REPORTED
# in the paper (not recomputed here). Their raw-ensemble CRPS uses the 50 members; ours uses a
# Gaussian N(ens_mean, ens_sd) approximation. Their NN numbers are ensembles of 10 networks.
PAPER_2007_2015 = {"Raw ensemble (50 members)": 1.16, "EMOS-gl": 1.00, "EMOS-loc": 0.90, "EMOS-loc-bst": 0.80,
                   "QRF": 0.81, "FCN": 1.01, "FCN-aux": 0.91, "FCN-emb": 0.91, "FCN-aux-emb": 0.87,
                   "NN-aux": 0.86, "NN-aux-emb (10-net ensemble)": 0.78}
PAPER_MATCH = {"raw_ensemble_gauss": "Raw ensemble (50 members)", "emos_gl": "EMOS-gl", "emos_loc": "EMOS-loc",
               "nn_aux_emb": "NN-aux-emb (10-net ensemble)", "nn_aux_emb_ens3": "NN-aux-emb (10-net ensemble)"}
BASELINES = ["raw_ensemble_gauss", "emos_gl", "emos_loc"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/default.json"))
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--figures", default=str(ROOT / "figures"))
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    ev = cfg["evaluation"]
    res, figdir = Path(args.results), Path(args.figures)
    figdir.mkdir(exist_ok=True)
    meta = np.load(res / "preds/test_meta.npz")
    y, unseen, station = meta["y"].astype(np.float64), meta["unseen"], meta["station"]
    dates = pd.to_datetime(meta["date"])
    subsets = {"all": np.ones_like(unseen, bool), "seen": ~unseen, "unseen": unseen}

    # ---- collect predictions
    preds = {}
    for b in BASELINES:
        z = np.load(res / f"preds/{b}.npz")
        preds[b] = (z["mu"], z["sigma"])
    seeds_of = {}
    for net in cfg["networks"]:
        ss = [s for s in cfg["seeds"] if (res / f"preds/{net}_seed{s}.npz").exists()]
        if not ss:
            continue
        seeds_of[net] = ss
        for s in ss:
            z = np.load(res / f"preds/{net}_seed{s}.npz")
            preds[f"{net}_seed{s}"] = (z["mu"], z["sigma"])
        # seed ensemble: average of distribution parameters (as in Rasp & Lerch 2018)
        preds[f"{net}_ens{len(ss)}"] = (np.mean([preds[f"{net}_seed{s}"][0] for s in ss], 0),
                                        np.mean([preds[f"{net}_seed{s}"][1] for s in ss], 0))

    # ---- metrics per prediction and subset
    metrics = {k: {sub: summary_metrics(mu[m], sg[m], y[m], ev["pit_bins"]) for sub, m in subsets.items()}
               for k, (mu, sg) in preds.items()}
    crps_rows = {k: crps_gaussian_np(mu, sg, y) for k, (mu, sg) in preds.items()}
    json.dump(metrics, open(res / "metrics_all.json", "w"), indent=1)

    # ---- summary table: networks as seed mean +- sd (ddof=1), baselines deterministic
    scalar = ["crps", "mae", "rmse", "coverage_80", "coverage_90", "width_80", "width_90",
              "pit_reliability_index", "mean_sigma", "bias_mean_minus_obs"]
    rows = []
    for sub in subsets:
        ref_raw = metrics["raw_ensemble_gauss"][sub]["crps"]
        ref_emos = metrics["emos_loc"][sub]["crps"]
        for b in BASELINES:
            r = {"model": b, "subset": sub, "n_seeds": 0, "n": metrics[b][sub]["n"]}
            for s in scalar:
                r[s + "_mean"] = metrics[b][sub][s]
                r[s + "_sd"] = np.nan
            rows.append(r)
        for net, ss in seeds_of.items():
            r = {"model": net, "subset": sub, "n_seeds": len(ss), "n": metrics[f"{net}_seed{ss[0]}"][sub]["n"]}
            for s in scalar:
                vals = np.array([metrics[f"{net}_seed{x}"][sub][s] for x in ss])
                r[s + "_mean"] = vals.mean()
                r[s + "_sd"] = vals.std(ddof=1) if len(vals) > 1 else np.nan
                if s == "crps":
                    r["crps_per_seed"] = ";".join(f"{v:.4f}" for v in vals)
            rows.append(r)
            e = f"{net}_ens{len(ss)}"
            re_ = {"model": e, "subset": sub, "n_seeds": len(ss), "n": metrics[e][sub]["n"]}
            for s in scalar:
                re_[s + "_mean"] = metrics[e][sub][s]
                re_[s + "_sd"] = np.nan
            rows.append(re_)
        for r in rows:
            if r["subset"] == sub:
                r["crpss_vs_raw"] = 1 - r["crps_mean"] / ref_raw
                r["crpss_vs_emos_loc"] = 1 - r["crps_mean"] / ref_emos
    table = pd.DataFrame(rows)
    table.to_csv(res / "summary_table.csv", index=False, float_format="%.5f")

    # ---- significance
    day = dates.normalize()
    sig_rows = []
    for a, b in ev["significance_pairs"]:
        if a not in seeds_of and a not in preds:
            continue
        versions = []
        if a in seeds_of:
            versions += [(f"{a}_seed{s}", f"{b}_seed{s}" if b in seeds_of else b, f"seed{s}") for s in seeds_of[a]]
            ea = f"{a}_ens{len(seeds_of[a])}"
            eb = f"{b}_ens{len(seeds_of[b])}" if b in seeds_of else b
            versions.append((ea, eb, "seed_ensemble"))
        for ka, kb, ver in versions:
            if ka not in crps_rows or kb not in crps_rows:
                continue
            for sub in ("all", "seen"):
                msk = subsets[sub]
                d = pd.DataFrame({"day": day[msk], "a": crps_rows[ka][msk], "b": crps_rows[kb][msk]})
                daily = d.groupby("day")[["a", "b"]].mean().sort_index()
                dm = diebold_mariano(daily["a"].to_numpy(), daily["b"].to_numpy(), ev["dm_hac_lag_days"])
                bs = block_bootstrap_ci((daily["a"] - daily["b"]).to_numpy(), ev["bootstrap_block_days"],
                                        ev["bootstrap_reps"])
                row = {"model_a": a, "model_b": b, "version": ver, "subset": sub,
                       "crps_a": float(crps_rows[ka][msk].mean()), "crps_b": float(crps_rows[kb][msk].mean()),
                       "mean_diff_a_minus_b": float(crps_rows[ka][msk].mean() - crps_rows[kb][msk].mean()),
                       "dm_stat_daily": dm["dm_stat"], "dm_p_daily": dm["p_value"], "n_days": dm["n"],
                       "boot_ci_low": bs["ci_low"], "boot_ci_high": bs["ci_high"]}
                if sub == "seen":  # station-wise DM + Benjamini-Hochberg (as in Rasp & Lerch, App. B.3)
                    st = pd.DataFrame({"st": station[msk], "day": day[msk], "a": crps_rows[ka][msk],
                                       "b": crps_rows[kb][msk]}).sort_values(["st", "day"])
                    ps, signs = [], []
                    for _, g in st.groupby("st"):
                        if len(g) < 30:
                            continue
                        r = diebold_mariano(g["a"].to_numpy(), g["b"].to_numpy(), ev["station_dm_hac_lag"])
                        ps.append(r["p_value"]); signs.append(np.sign(r["mean_diff"]))
                    ps, signs = np.array(ps), np.array(signs)
                    rej = benjamini_hochberg(ps, 0.05)
                    row.update({"n_stations_tested": int(len(ps)),
                                "frac_stations_a_sig_better_BH": float((rej & (signs < 0)).mean()),
                                "frac_stations_a_sig_worse_BH": float((rej & (signs > 0)).mean()),
                                "frac_stations_a_lower_crps": float((signs < 0).mean())})
                sig_rows.append(row)
    sig = pd.DataFrame(sig_rows)
    sig.to_csv(res / "significance.csv", index=False, float_format="%.6g")

    # ---- reproduction check vs paper
    comp = []
    for k, label in PAPER_MATCH.items():
        if k in metrics:
            comp.append({"ours": k, "paper_model": label, "paper_crps_2016_reported": PAPER_2007_2015[label],
                         "ours_crps_all": metrics[k]["all"]["crps"], "ours_crps_seen": metrics[k]["seen"]["crps"]})
    if "nn_aux_emb" in seeds_of:
        comp.append({"ours": "nn_aux_emb (single-seed mean)", "paper_model": "NN-aux-emb (10-net ensemble)",
                     "paper_crps_2016_reported": 0.78,
                     "ours_crps_all": float(np.mean([metrics[f"nn_aux_emb_seed{s}"]["all"]["crps"] for s in seeds_of["nn_aux_emb"]])),
                     "ours_crps_seen": float(np.mean([metrics[f"nn_aux_emb_seed{s}"]["seen"]["crps"] for s in seeds_of["nn_aux_emb"]]))})
    pd.DataFrame(comp).to_csv(res / "paper_comparison.csv", index=False, float_format="%.4f")

    # ---- monthly and per-station CRPS
    keys = BASELINES + [f"{n}_ens{len(s)}" for n, s in seeds_of.items()]
    mon = pd.DataFrame({k: crps_rows[k] for k in keys})
    mon["month"] = dates.month
    mon.groupby("month").mean().to_csv(res / "monthly_crps.csv", float_format="%.4f")
    pst = pd.DataFrame({k: crps_rows[k] for k in keys})
    pst["station"], pst["lat"], pst["lon"], pst["unseen"] = station, meta["lat"], meta["lon"], unseen
    pst_g = pst.groupby("station").agg({**{k: "mean" for k in keys}, "lat": "first", "lon": "first", "unseen": "first"})
    pst_g["n"] = pst.groupby("station").size()
    pst_g.to_csv(res / "per_station_crps.csv", float_format="%.4f")

    # ---- figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    show = ["raw_ensemble_gauss", "emos_loc"] + [f"{n}_ens{len(s)}" for n, s in seeds_of.items()]
    fig, axes = plt.subplots(2, (len(show) + 1) // 2, figsize=(3.2 * ((len(show) + 1) // 2), 6), squeeze=False)
    for ax, k in zip(axes.ravel(), show):
        h = metrics[k]["all"]["pit_hist"]
        ax.bar(np.arange(len(h)) / len(h), h, width=1 / len(h), align="edge", edgecolor="k")
        ax.axhline(1 / len(h), color="r", ls="--")
        ax.set_title(k, fontsize=8); ax.set_xlabel("PIT")
    for ax in axes.ravel()[len(show):]:
        ax.axis("off")
    fig.tight_layout(); fig.savefig(figdir / "pit_histograms.png", dpi=150); plt.close(fig)

    mm = mon.groupby("month").mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    for k in show:
        ax.plot(mm.index, mm[k], marker="o", label=k)
    ax.set_xlabel("month (2016)"); ax.set_ylabel("mean CRPS (°C)"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(figdir / "monthly_crps.png", dpi=150); plt.close(fig)

    if "residual_spread" in seeds_of:
        k = f"residual_spread_ens{len(seeds_of['residual_spread'])}"
        diff = pst_g[k] - pst_g["emos_loc"]
        fig, ax = plt.subplots(figsize=(5, 6))
        lim = np.nanquantile(np.abs(diff), 0.98)
        sc = ax.scatter(pst_g["lon"], pst_g["lat"], c=diff, cmap="RdBu_r", vmin=-lim, vmax=lim, s=12)
        ax.scatter(pst_g.loc[pst_g.unseen, "lon"], pst_g.loc[pst_g.unseen, "lat"], marker="x", c="k", s=40,
                   label="unseen stations")
        plt.colorbar(sc, label="CRPS(residual_spread) - CRPS(EMOS-loc)")
        ax.set_xlabel("lon"); ax.set_ylabel("lat"); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(figdir / "station_crps_diff_vs_emos_loc.png", dpi=150); plt.close(fig)

    # console summary
    pd.set_option("display.width", 250)
    cols = ["model", "subset", "n", "crps_mean", "crps_sd", "crps_per_seed", "mae_mean", "rmse_mean",
            "coverage_80_mean", "coverage_90_mean", "pit_reliability_index_mean", "crpss_vs_raw", "crpss_vs_emos_loc"]
    print(table[[c for c in cols if c in table.columns]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(sig.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(pd.DataFrame(comp).to_string(index=False))


if __name__ == "__main__":
    main()
