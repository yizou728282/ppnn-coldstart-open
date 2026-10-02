# Stage 2 pre-registered protocol: generalization to new stations

Registered on 2026-09-26, before any Stage-2 model was trained or evaluated. The machine-readable version is `stage2_new_station.json`. If the two files disagree, the JSON is authoritative.

## Question
How should a post-processing network forecast at a station it has never seen, and how much skill is lost compared with seen stations?

## Design
* **Data:** Rasp & Lerch (2018) PPNN (537 DWD stations, ECMWF ensemble; `sm_*` columns dropped).
* **Folds:**
  * **Primary:** station-held-out 7-fold CV. All 537 station ids are randomly permuted (seed 2026) and split into 7 folds of about 77 stations (about 14%).
  * **Secondary robustness check:** 7 spatially blocked folds from k-means on (lat, lon), seed 2026.
* **Isolation:** held-out stations are excluded from *every* fitting step in *every* year (fitting, early stopping, normalization, EMOS). Only their 2016 records are scored.
* **Years:** fit on 2007–2014. Early stopping uses the 2015 records of the training stations (patience 3, max 30 epochs). No refit. 2016 is the test year.
* **Seeds:** 10 seeds (0–9) per network for the random folds. 3 seeds for the spatial folds (runtime budget).

## Methods
| id | description |
|---|---|
| raw_ensemble_gauss | N(ens mean, ens sd) |
| emos_gl | global EMOS, min-CRPS |
| emos_bst_gl | global boosted EMOS: Gaussian, log-link scale, maximum likelihood, component-wise boosting over all 38 predictors, step 0.05, at most 1000 iterations, AIC stopping (the crch settings of Rasp & Lerch, but global because local fitting is impossible for a new station) |
| nn_unk | NN-aux-emb with a reserved UNK embedding (5% station dropout) |
| nn_knn | same trained nets as nn_unk. A new station's embedding is the inverse-distance-weighted mean over the 5 nearest training stations, with distance √(d_km² + (0.1·Δalt_m)²) |
| nn_noemb | no embedding (attributes lat, lon, alt, orog are inputs, as for all networks) |
| nn_attr | embedding = MLP(lat, lon, alt, orog, alt−orog) → 2-d, trained jointly |
| **nn_hybrid (proposed)** | embedding = MLP(attributes) + a per-station offset δ_s. δ_s is zeroed with p = 0.5 during training. New stations use δ = 0 |
| nn_hybrid_res (exploratory) | nn_hybrid + residual mean + constrained spread |
| nn_hybrid_knn (exploratory) | nn_hybrid with δ of a new station taken from the kNN-IDW of training δ |

Shared network settings: 1 hidden layer of 512 ReLU units, Adam with lr 0.002, batch 4096, Gaussian CRPS loss.

**Reference on seen stations:** EMOS-loc-bst, fitted per station on 2007–2015 and tested on 2016 (Stage-1 setting; the paper reports 0.80).

## Metrics
* **Primary:** CRPS over all 2016 records of held-out stations, pooled across folds (each station is held out exactly once), for the 10-seed ensemble forecast (mean of μ and σ over seeds).
* **Also reported:**
  * mean ± sd of single-seed CRPS, and per-fold CRPS
  * CRPSS vs the raw ensemble and vs emos_gl
  * MAE and RMSE
  * 80% and 90% coverage, PIT histogram and reliability index
  * mean σ vs RMSE
  * CRPS as a function of distance to the nearest training station

## Significance (fixed in advance)
* **Primary family (6 comparisons):** nn_hybrid vs {emos_gl, emos_bst_gl, nn_unk, nn_knn, nn_noemb, nn_attr}.
* **Tests:**
  * DM test on the 366 daily-mean CRPS differences (Newey–West lag 5), with Benjamini–Hochberg correction at q = 0.05.
  * Moving-block bootstrap (7-day blocks, 10,000 replicates) 95% CI.
  * Station-wise DM (lag 1) + BH: fractions of held-out stations significantly better or worse.
* **Claim rule:** "better" requires a BH-significant DM test AND a bootstrap CI that excludes 0 AND the same sign in the spatial-fold check.
* Exploratory variants are never promoted, whatever their results. No choice is made using 2016 data.
