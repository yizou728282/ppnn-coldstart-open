# Stage 3 protocol: independent validation of the new-station study on EUPPBench (pre-registered)

Registered on 2026-09-26, **before any Stage-3 model was trained and before any Stage-3 test-period score was computed**. The machine-readable version is `stage3_euppbench.json`; if the two disagree, the JSON is authoritative. The design copies Stage 2 (`stage2_new_station.*`) wherever possible. The only changes are the ones the new dataset forces.

## Data
* **Source:** EUPPBench v1.0 station data, Zenodo 10.5281/zenodo.7708362 (`EUPPBench-stations.zip`, 18,324,361,137 bytes, md5 `e409457279b3494d18f2dfb41f3f449b`).
* **Download:** only the needed Zarr stores and variables are fetched, from the official climetlab/ECMWF object store. Every file is checked by size and CRC32 against the central directory of the Zenodo zip (`scripts/stage3_download.py`).
* **Licence:** the Zenodo record says "other-at" and ships a LICENSE file. ECMWF forecasts are CC BY 4.0; station observations come from RMI, Météo-France, ZAMG, KNMI and DWD; static fields come from Copernicus (CLMS/EEA).
* **Countries:** AT, BE, FR, DE and NL. Swiss stations are not in the public dataset.
* **Station inclusion:** a station is used if it has at least 50% non-missing t2m observations at the five leads, both in the reforecast period (valid dates up to 2016) and in the test period. This rule looks at data availability only. Excluded stations are listed in `results/stage3/stations.csv`.
* **Target and lead times:** station 2 m temperature (°C) at leads 24, 48, 72, 96 and 120 h from 00 UTC runs.
* **Predictors:** ensemble mean and standard deviation (ddof = 1) of 16 variables:
  * surface: t2m, u10, v10, tcc, sd, stl1, swvl1, tcwv, cape
  * 6-h window ending at the lead: mx2t6, mn2t6, sshf6, slhf6, ssr6, str6
  * 850 hPa: t
  * plus day-of-year sin/cos, a lead-time one-hot, and the station attributes as plain inputs.
* **Station attributes:** lat, lon, station altitude, model orography, altitude − orography, and a one-hot of the CORINE level-1 land-use class (artificial, agricultural, forest/semi-natural, wetland, water/no-data).
* **Ensemble-size mismatch:**
  * Reforecasts (fit and early stopping) have 11 members; test forecasts have 51. Models only ever see ensemble mean and sd statistics.
  * **Primary:** test statistics from all 51 members.
  * **Pre-registered sensitivity:** members 0–10 only, i.e. the control plus 10 perturbed members. This uses the same trained models with no refit.
* **Missing values:**
  * Rows with a missing observation or missing t2m statistics are dropped.
  * Any other missing predictor is set to its training mean.

## Periods
* **Fit:** reforecasts whose valid date is in 1997–2014.
* **Early stopping:** reforecasts whose valid date is in 2015–2016 (training stations only).
* **Discarded:** reforecasts valid on or after 2017-01-01. The 2018 model-date reforecasts cover 2017, which would overlap the test period.
* **Test:** all 730 daily 00 UTC forecasts from 2017-01-01 to 2018-12-31, leads 24–120 h.
* **No refit.**

## Lead-time handling
* **Networks:** one network per (variant, fold, seed), with a lead-time one-hot input and a station representation shared across leads. This means 5× fewer runs, 5× more data behind each station embedding, and the hidden layer can still learn lead interactions.
* **Linear baselines:** global EMOS, global boosted EMOS and the local boosted-EMOS reference are fitted separately per lead, as is standard. A single linear model cannot represent lead-dependent coefficients.

## Folds
* **Primary:** 7 random station folds (seed 2026).
* **Robustness:** 7 spatial k-means folds on lat/lon (seed 2026).
* Held-out stations are removed from every fitting step.
* There are about 120 stations, so each fold holds out about 17. This is far fewer than in Stage 2 (535), so power is lower.

## Methods (definitions unchanged from Stage 2)
* raw_ensemble_gauss
* emos_gl (per lead)
* emos_bst_gl (per lead; standardized response, step 0.05, max 1000 iterations, AIC stopping)
* nn_unk
* nn_knn (IDW over k = 5, altitude-penalized distance)
* nn_noemb
* nn_attr
* **nn_hybrid** (the Stage-2 proposal)
* Exploratory: nn_hybrid_res, nn_hybrid_knn

Network hyperparameters are identical to Stage 2 (hidden 512, embedding dimension 2, Adam lr 0.002, batch 4096, at most 30 epochs, patience 3). Nothing is tuned. Seeds: 0–9 for random folds, 0–2 for spatial folds.

Reference on seen stations: per-station, per-lead boosted EMOS fitted on the station's reforecasts valid 1997–2016.

## Compute
* **Networks:** all Stage-3 networks run in the same Google Colab GPU environment (T4, CUDA), with deterministic algorithms, `CUBLAS_WORKSPACE_CONFIG=:4096:8` and cudnn.benchmark off.
* **Random streams:** permutations and dropout masks come from seeded CPU generators, so they do not depend on the device.
* **Baselines:** deterministic numpy code.
* **Smoke test:** a tiny CPU pipeline test on the box was run beforehand. Its outputs are discarded and never reported.

## Metrics and tests
* **Primary:** mean CRPS over all held-out test rows (5 leads), pooled over folds, using the 10-seed ensemble.
* **Secondary:**
  * CRPSS vs raw ensemble and vs emos_gl
  * CRPS per lead
  * MAE, RMSE
  * 80% and 90% coverage
  * PIT histogram and reliability index
  * spread–skill ratio and bias
  * seen-station CRPS
  * single-seed mean ± sd
  * per-fold CRPS
  * CRPS vs distance to the nearest training station
  * the 11-member sensitivity
* **Primary tests:** nn_hybrid vs {emos_gl, emos_bst_gl, nn_unk, nn_knn, nn_noemb, nn_attr}.
  * Diebold–Mariano test on the 730 per-initialization-date means (over held-out stations and leads), Newey–West lag 5.
  * Moving-block bootstrap: 7-day blocks, 10,000 reps.
  * Benjamini–Hochberg correction, q = 0.05.
  * Station-wise DM (lag 1) with BH correction.
* **Claim rule (as in Stage 2):** all three must hold: BH-significant, the bootstrap CI excludes 0, and the same sign on spatial folds.

## Replication hypotheses carried over from Stage 2
* **H1:** every network beats emos_gl and emos_bst_gl on held-out stations, on both fold types.
* **H2:** on random folds, hybrid, attr and noemb are within 1% of each other, and all beat unk.
* **H3:** on spatial folds, noemb beats attr and hybrid.
* **H4:** knn is worse than unk.
* **H5:** on seen stations, hybrid beats noemb.

Each hypothesis is reported as supported or not supported. No other claims will be made.
