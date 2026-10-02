"""Evaluate Stage-4 methods (protocols/stage4_baselines.json, protocols/stage4_pilot.json) together with the existing
Stage-2/3 methods, on held-out stations. Everything is computed from stored prediction files.
    python scripts/stage4_evaluate.py            # all available
    python scripts/stage4_evaluate.py --promising-only german   # prints PROMISING / NOT (pilot phase-2 condition)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from ppnn_residual.metrics import (benjamini_hochberg, block_bootstrap_ci, crps_gaussian_np,  # noqa: E402
                                   diebold_mariano, summary_metrics)
from stage4_diagnose import station_features  # noqa: E402

BASE = ["raw_ensemble_gauss", "emos_gl", "emos_bst_gl"]
OLD_NETS = ["nn_unk", "nn_knn", "nn_noemb", "nn_attr", "nn_hybrid", "nn_hybrid_res"]
NEW = ["drn_lak", "gnn_geo", "samos_attn", "samos_mlp"]
PB = json.loads((ROOT / "protocols/stage4_baselines.json").read_text())
PP = json.loads((ROOT / "protocols/stage4_pilot.json").read_text())
OLD_DIR = {"german": ROOT / "results/stage2", "euppbench": ROOT / "results/stage3"}


def load_method(ds, ftype, m, seeds, row_fold, K):
    n = len(row_fold)
    if m in BASE:
        d = OLD_DIR[ds] / "preds" / ftype
        fs = [d / f"{m}_f{k}.npz" for k in range(K)]
        if not all(f.exists() for f in fs):
            return None
        mu, sg = np.empty(n), np.empty(n)
        for k, f in enumerate(fs):
            z = np.load(f); r = row_fold == k; mu[r], sg[r] = z["mu"][r], z["sigma"][r]
        return {"ens": (mu, sg), "single": {}}
    d = (OLD_DIR[ds] if m in OLD_NETS else ROOT / "results/stage4" / ds) / "preds" / ftype
    have = [s for s in seeds if all((d / f"{m}_f{k}_s{s}.npz").exists() for k in range(K))]
    if len(have) != len(seeds):
        return None
    single = {}
    for s in have:
        mu, sg = np.empty(n), np.empty(n)
        for k in range(K):
            z = np.load(d / f"{m}_f{k}_s{s}.npz"); r = row_fold == k
            a, b = z["mu"].astype(float), z["sigma"].astype(float)
            if len(a) == n:
                a, b = a[r], b[r]
            mu[r], sg[r] = a, b
        single[s] = (mu, sg)
    return {"ens": (np.mean([v[0] for v in single.values()], 0), np.mean([v[1] for v in single.values()], 0)), "single": single}


def compare(crps, day, st, a, b):
    dfd = pd.DataFrame({"d": day, "a": crps[a], "b": crps[b]}).groupby("d")[["a", "b"]].mean().sort_index()
    dm = diebold_mariano(dfd.a.values, dfd.b.values, 5); bs = block_bootstrap_ci((dfd.a - dfd.b).values, 7, 10000)
    return {"a": a, "b": b, "crps_a": crps[a].mean(), "crps_b": crps[b].mean(), "diff_a_minus_b": crps[a].mean() - crps[b].mean(),
            "rel_diff_pct": 100 * (crps[a].mean() / crps[b].mean() - 1), "n_days": len(dfd), "dm_stat": dm["dm_stat"],
            "dm_p": dm["p_value"], "boot_lo": bs["ci_low"], "boot_hi": bs["ci_high"]}


def evaluate(ds, ftype):
    out = ROOT / "results/stage4" / ds; out.mkdir(parents=True, exist_ok=True)
    meta = np.load(OLD_DIR[ds] / "test_meta.npz", allow_pickle=True)
    folds = pd.read_csv(OLD_DIR[ds] / f"folds_{ftype}.csv", index_col=0); K = int(folds.fold.max()) + 1
    st = meta["station"].astype(int); y = meta["y"].astype(float)
    row_fold = folds.fold.reindex(st).to_numpy()
    day = meta["init"] if "init" in meta else meta["date"]
    lead = meta["lead"] if "lead" in meta else np.full(len(y), 48)
    R = {}
    for m in BASE + OLD_NETS + NEW:
        r = load_method(ds, ftype, m, [0, 1, 2], row_fold, K)
        if r is not None:
            R[m] = r
    if ftype == "random":  # 10-seed reference for the old networks
        for m in OLD_NETS:
            r = load_method(ds, ftype, m, list(range(10)), row_fold, K)
            if r is not None:
                R[m + "@10seeds"] = r
    crps, rows = {}, []
    for m, r in R.items():
        mu, sg = r["ens"]; met = summary_metrics(mu, sg, y); crps[m] = crps_gaussian_np(mu, sg, y)
        row = {"method": m, "n_seeds": len(r["single"]), "crps": met["crps"], "mae": met["mae"], "rmse": met["rmse"],
               "cov80": met["coverage_80"], "cov90": met["coverage_90"], "pit_ri": met["pit_reliability_index"],
               "spread_skill": float(np.sqrt((sg ** 2).mean()) / met["rmse"]), "bias": met["bias_mean_minus_obs"]}
        if r["single"]:
            sc = [crps_gaussian_np(*v, y).mean() for v in r["single"].values()]
            row["single_seed_mean"], row["single_seed_sd"] = float(np.mean(sc)), float(np.std(sc, ddof=1))
        rows.append(row)
    tab = pd.DataFrame(rows)
    tab.to_csv(out / f"summary_{ftype}.csv", index=False, float_format="%.5f")
    pd.DataFrame({m: [crps[m][row_fold == k].mean() for k in range(K)] for m in crps}).to_csv(out / f"per_fold_{ftype}.csv", float_format="%.5f")
    pd.DataFrame({m: [crps[m][lead == L].mean() for L in np.unique(lead)] for m in crps}, index=np.unique(lead)).to_csv(
        out / f"per_lead_{ftype}.csv", float_format="%.5f")
    sf = station_features(folds)
    rs = sf.reindex(st)
    groups = {"all": np.ones(len(y), bool), "attr_outside_train_range": rs.any_attr_outside_range.to_numpy(bool),
              "attr_inside_train_range": ~rs.any_attr_outside_range.to_numpy(bool),
              "no_train_station_within_50km": (rs.n_train_50km == 0).to_numpy(), "dist_top20pct": (rs.dist_nearest_km >= np.quantile(sf.dist_nearest_km, .8)).to_numpy()}
    pd.DataFrame({g: {m: crps[m][msk].mean() for m in crps} for g, msk in groups.items()}).assign(
        **{"n_stations_" + g: len(np.unique(st[msk])) for g, msk in groups.items()}).to_csv(out / f"station_groups_{ftype}.csv", float_format="%.5f")
    # tests
    sig = []
    fam_b = [tuple(p) for p in PB["tests"]["descriptive_pairs"] if p[0] in crps and p[1] in crps]
    res = [compare(crps, day, st, a, b) for a, b in fam_b]
    if res:
        rej = benjamini_hochberg(np.array([r["dm_p"] for r in res]), 0.05)
        sig += [{**r, "family": "baselines_descriptive", "bh_reject_q05": bool(x)} for r, x in zip(res, rej)]
    prim = [m for m in PP["comparators"]["primary"] if m in crps]
    if "samos_attn" in crps:
        sig += [{**compare(crps, day, st, "samos_attn", b), "family": "pilot_primary", "bh_reject_q05": None} for b in prim]
        for b in PP["comparators"]["strict_secondary"] + ["samos_mlp"]:
            if b in crps and b not in prim:
                sig.append({**compare(crps, day, st, "samos_attn", b), "family": "pilot_secondary", "bh_reject_q05": None})
    sig = pd.DataFrame(sig)
    return tab, sig, crps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--promising-only", default=None)
    a = ap.parse_args()
    dsets = [a.promising_only] if a.promising_only else ["german", "euppbench"]
    decision = {}
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    for ds in dsets:
        tabs, sigs = {}, {}
        for ft in ["random", "spatial"]:
            if not (OLD_DIR[ds] / f"folds_{ft}.csv").exists():
                continue
            tabs[ft], sigs[ft], _ = evaluate(ds, ft)
        # BH over the 6 pilot primary tests (both fold types)
        pr = [(ft, i) for ft in sigs if len(sigs[ft]) for i in sigs[ft].index[sigs[ft].family == "pilot_primary"]]
        if pr:
            rej = benjamini_hochberg(np.array([sigs[ft].loc[i, "dm_p"] for ft, i in pr]), 0.05)
            for (ft, i), x in zip(pr, rej):
                sigs[ft].loc[i, "bh_reject_q05"] = bool(x)
        for ft in sigs:
            sigs[ft].to_csv(ROOT / "results/stage4" / ds / f"significance_{ft}.csv", index=False, float_format="%.6g")
            print(f"\n===== {ds} {ft} =====\n", tabs[ft].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
            if len(sigs[ft]):
                print(sigs[ft].to_string(index=False, float_format=lambda v: f"{v:.4g}"))
        # pilot decision
        if all(ft in tabs and "samos_attn" in set(tabs[ft].method) for ft in ["random", "spatial"]):
            d = {}
            for ft in ["random", "spatial"]:
                t = tabs[ft].set_index("method").crps
                best = min(PP["comparators"]["primary"], key=lambda m: t[m])
                sg_ = sigs[ft]; s = sg_[(sg_.family == "pilot_primary") & (sg_.b == best)].iloc[0]
                d[ft] = {"best_primary_comparator": best, "crps_best": float(t[best]), "crps_samos_attn": float(t["samos_attn"]),
                         "diff": float(s.diff_a_minus_b), "dm_p": float(s.dm_p), "bh_reject": bool(s.bh_reject_q05),
                         "boot_ci": [float(s.boot_lo), float(s.boot_hi)]}
            sp, rd = d["spatial"], d["random"]
            go = sp["bh_reject"] and sp["boot_ci"][1] < 0 and sp["diff"] < 0 and rd["diff"] < 0
            d["promising_point_estimate_spatial"] = bool(sp["diff"] < 0)
            d["decision"] = "GO" if go else "NO-GO"
            decision[ds] = d
        elif "spatial" in tabs and "samos_attn" in set(tabs["spatial"].method):
            t = tabs["spatial"].set_index("method").crps; best = min(PP["comparators"]["primary"], key=lambda m: t[m])
            decision[ds] = {"promising_point_estimate_spatial": bool(t["samos_attn"] < t[best]), "partial": True}
    if decision:
        prev = {}
        f = ROOT / "results/stage4/pilot_decision.json"
        if f.exists():
            prev = json.loads(f.read_text())
        prev.update(decision); f.write_text(json.dumps(prev, indent=1))
        print("\npilot decision:", json.dumps(decision, indent=1))
    if a.promising_only:
        print("PROMISING" if decision.get(a.promising_only, {}).get("promising_point_estimate_spatial") else "NOT_PROMISING")


if __name__ == "__main__":
    main()
