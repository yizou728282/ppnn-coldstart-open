"""Render results/stage2/*.csv into results/stage2/summary_tables.md (no new numbers)."""
import pandas as pd
out = []
for ft, lab in [("random", "Primary: random station folds (K=7, 10 seeds)"), ("spatial", "Secondary: spatial k-means folds (K=7, 3 seeds)")]:
    t = pd.read_csv(f"results/stage2/summary_{ft}.csv")
    out.append(f"\n**{lab}** — held-out stations, 2016, n = {int(t.n_rows.iloc[0])} rows\n")
    out.append("| method | CRPS (seed-ens.) | single-seed CRPS mean ± sd | CRPS on seen stations (same models) | CRPSS vs raw | CRPSS vs EMOS-gl | MAE | RMSE | cov80 | cov90 | PIT RI | spread/skill |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in t.iterrows():
        ss = "" if pd.isna(r.get("crps_single_seed_mean")) else f"{r.crps_single_seed_mean:.4f} ± {r.crps_single_seed_sd:.4f}"
        out.append(f"| {r.method} | {r.crps_primary:.4f} | {ss} | {r.crps_seen_stations_mean_over_folds:.4f} | {r.crpss_vs_raw_ensemble_gauss:.3f} | {r.crpss_vs_emos_gl:.3f} | {r.mae:.4f} | {r.rmse:.4f} | {r.coverage_80:.3f} | {r.coverage_90:.3f} | {r.pit_reliability_index:.3f} | {r.spread_skill:.3f} |")
    s = pd.read_csv(f"results/stage2/significance_{ft}.csv")
    out.append(f"\nSignificance ({ft} folds; A−B < 0 means A is better):\n")
    out.append("| family | A | B | A−B | rel. % | DM stat | p | BH reject (primary) | 95% block-bootstrap CI | stations A better / worse (BH) |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    for _, r in s.iterrows():
        bh = "yes" if r.family == "primary" and str(r.bh_reject_q05) == "True" else ("no" if r.family == "primary" else "–")
        out.append(f"| {r.family} | {r.a} | {r.b} | {r.diff_a_minus_b:+.4f} | {r.rel_diff_pct:+.2f} | {r.dm_stat:.2f} | {r.dm_p:.2g} | {bh} | [{r.boot_lo:+.4f}, {r.boot_hi:+.4f}] | {r.frac_st_a_better_BH:.3f} / {r.frac_st_a_worse_BH:.3f} |")
    d = pd.read_csv(f"results/stage2/distance_bins_{ft}.csv")
    cols = ["emos_gl", "emos_bst_gl", "nn_unk", "nn_knn", "nn_noemb", "nn_attr", "nn_hybrid"]
    out.append(f"\nCRPS by distance to nearest training station ({ft} folds, station quintiles):\n")
    out.append("| km range | stations | " + " | ".join(cols) + " |"); out.append("|---|---|" + "---|" * len(cols))
    for _, r in d.iterrows():
        out.append(f"| {r.dist_km_lo:.1f}–{r.dist_km_hi:.1f} | {int(r.n_stations)} | " + " | ".join(f"{r[c]:.4f}" for c in cols) + " |")
    f = pd.read_csv(f"results/stage2/per_fold_{ft}.csv", index_col=0)
    c2 = ["emos_gl", "emos_bst_gl", "nn_unk", "nn_noemb", "nn_attr", "nn_hybrid"]
    out.append(f"\nPer-fold CRPS ({ft} folds, seed ensembles):\n")
    out.append("| fold | " + " | ".join(c2) + " |"); out.append("|---|" + "---|" * len(c2))
    for k, r in f.iterrows():
        out.append(f"| {k} | " + " | ".join(f"{r[c]:.4f}" for c in c2) + " |")
open("results/stage2/summary_tables.md", "w").write("\n".join(out) + "\n")
print("\n".join(out))
