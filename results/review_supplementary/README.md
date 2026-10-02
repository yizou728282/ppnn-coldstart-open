# Supplementary DiD intervals (matched seeds, forecast-only GNN, regional blocks)

Post hoc analyses computed from the preserved per-seed predictions; no model was retrained and no existing
table or number was changed.

| File | Contents |
|---|---|
| `did_euppbench.json`, `did_german.json` | Point estimates, 95% percentile intervals and P(DiD>0) for every contrast and resampling scheme, cluster definitions and reproduction checks |
| `did_review_supplementary.csv` | Flat summary of the same results |

Calculation script: `analysis/review_supplementary_did.py`. The checked-in CSV summarises the saved JSON results.

## Method

DiD = (F - E)_spatial - (F - E)_random with E = global Boosted EMOS, pooled CRPS over held-out records.

* **Station bootstrap (primary procedure).** The unchanged `Design` class of `scripts/stage5_inference.py`:
  2000 replicates, station multinomial weights from `default_rng(2026)` shared by both designs, and independent
  seed resampling from `default_rng(99 + crc32(design) % 1000)` consumed in the original method order. Additional
  ensembles (`drn_lak@3`, and `gnn_geo@3` for the German data) are appended after the original method list, so
  every original replicate is unchanged.
* **Matched seeds.** Spatial ensemble restricted to seeds 0-2 (`F@3`) versus the three-seed random ensemble.
* **Forecast-only GNN.** `gnn_geo_fcavail` (seeds 0-2) in both designs (EUPPBench only; the German files contain no
  forecast-only neighbour runs).
* **Paired regional block bootstrap.** Whole clusters are drawn with replacement; the same cluster weights apply to
  both designs, nested with the same seed-resampled station sums. Clusters: the 7 primary spatial folds
  (`default_rng(4242)`), the 5 EUPPBench countries (`default_rng(5005)`), and k-means regions on station
  coordinates using `ppnn_residual.folds.spatial_folds(seed=2026, n_init=20)` with G = 10/20/30 (EUPPBench) and
  20/50/100 (German), weights from `default_rng(31000 + G)`.

## Reproduction checks (all pass)

* Primary DiD, its interval, P(DiD>0) and both F - E intervals equal `results/stage5/eval/final/decisions.json`
  (tolerance 1e-12) for DRN and GNN-Geo on both datasets.
* The forecast-only versus historical GNN station intervals equal the delivered pair tables (6 significant figures).
* The EUPPBench primary-fold paired block intervals equal the original fold-block DiD supplement
  (`results/stage5/eval/final_supp/did_fold_block.json` in the research workspace, tolerance 1e-12).

## Running

```bash
PPNN_DATA_ROOT=/path/to/research/workspace OMP_NUM_THREADS=3 python analysis/review_supplementary_did.py
```

`PPNN_DATA_ROOT` must contain `results/stage2`, `results/stage3`, `results/stage4/<dataset>/preds` and
`results/stage5` as used by the final Stage-5 evaluation (provider observations are needed in `test_meta.npz`).
Run time on 3-4 CPU threads: about 4 min (German) and 10 min (EUPPBench).
