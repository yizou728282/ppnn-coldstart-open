# Stage 5: supplementary robustness analyses

Registered 2026-09-26 ~23:45 UTC-6. **No Stage-5 model had been trained and no Stage-5 result had been looked at when this was written.**
The only computations before this commit were geometric: the alternative spatial partitions below were computed from station coordinates alone, to check that they differ from the primary one. No forecasts or observations were used.
Machine-readable version: `protocols/stage5_supplementary.json`.

Stage 5 addresses the following methodological questions:
* **M1**: the inference unit (station / region sampling variance)
* **M2**: seed variance
* **M5**: HAC bandwidth
* **M6**: missing cold-start statistical baselines
* **m3**: a single spatial partition
* **m1a**: neighbour availability based on observations

**Status.** Stage 5 is a *supplementary robustness* stage. It adds seeds, partitions, baselines and inference to the existing Stage-2/3/4 designs, without changing their data, folds, periods, hyper-parameters or code paths. The core "random vs spatial" narrative was **not** pre-registered in earlier stages. Stage 5 pre-states the decision rules (Section F) under which that narrative will be worded *before* the new results exist. It does not make the narrative retroactively pre-registered.

## A. Seeds (M2)
* **EUPPBench, primary spatial folds** (k-means, seed 2026, K = 7): every main method is extended from seeds 0–2 to seeds **0–9**. New seeds 3–9 use the existing code, hyper-parameters and thread settings: Stage-3 nets 3 torch threads, Stage-4 methods 3 threads (Stage 4 originally used 4).
  * Methods: nn_unk (+ nn_knn), nn_noemb, nn_attr, nn_hybrid (+ nn_hybrid_knn), nn_hybrid_res, drn_lak, gnn_geo, samos_mlp, samos_attn.
  * emos_bst_gl is deterministic and needs no seeds.
* **German, primary spatial folds**: the same methods get seeds 3–9, again all main methods, because it is cheap. Order: first the five Stage-2 nets, then drn_lak and samos_mlp, then gnn_geo and samos_attn (most expensive).
* Random folds are unchanged: Stage-2/3 nets have 10 seeds; drn_lak, gnn_geo and samos_* have 3.
* The seed ensemble is defined as before: mean of μ and mean of σ over seeds.

## B. Repeated spatial partitions (m3), EUPPBench only
Six additional station partitions:

| id | rule | fold sizes |
|---|---|---|
| km7_s2027 | k-means, K=7, **single** k-means++ start (n_init=1), seed 2027 | 13 11 34 16 17 23 3 |
| km7_s2028 | same, seed 2028 | 14 17 34 18 14 3 17 |
| km7_s2029 | same, seed 2029 | 4 7 17 21 20 27 21 |
| km5_s2026 | k-means, K=5, n_init=20, seed 2026 (the primary rule with K=5) | 20 11 26 35 25 |
| km10_s2026 | k-means, K=10, n_init=20, seed 2026 | 15 11 15 15 16 3 10 13 7 12 |
| loco | leave-one-country-out, 5 folds | AT 4, BE 27, DE 49, FR 7, NL 30 |

* Why single-start for the extra K=7 partitions: with the primary rule (n_init=20), other seeds mostly return the primary partition or near-copies (ARI 0.79–1.0 vs primary for seeds 2027–2043; checked from coordinates only). Single-start k-means gives genuinely different spatial blockings: ARI vs primary 0.72 / 0.82 / 0.74.
* Methods: emos_bst_gl (1000 iterations, as primary), nn_unk, nn_hybrid_res, drn_lak, gnn_geo, samos_mlp, **3 seeds (0–2)** each, plus the new statistical baselines of Section C (deterministic).
* No buffer zones. A buffered leave-one-out and a variogram analysis are **not** done (scope). This is disclosed.

## C. New cold-start statistical baselines (M6)
All are fitted on training stations only, over the same fit period as emos_bst_gl: German 2007–2014; EUPPBench reforecasts valid ≤ 2014.

1. **Local EMOS per station and lead**: μ = a + b·m, σ² = c² + d²·s², minimum CRPS, the existing `fit_emos` code. Each station's fit uses only its own data, so it is computed once and reused by every fold. A held-out station's own fit is never used.
2. **Interpolation of local coefficients** (a, b, c², d²) to the new station, separately for each lead:
   * `emos_nn1`: coefficients of the nearest training station, using the distance defined for `emos_idw`.
   * `emos_idw`: IDW over the 5 nearest training stations, with distance √(d_km² + (0.1 km/m · Δalt)²) and weights 1/d. This is the same kernel as `nn_knn` (`knn_idw`).
   * `emos_reg`: OLS regression of each coefficient on the training stations' [1, lat, lon, alt, orog, alt−orog].
   * **`emos_regidw` (primary interpolated baseline)**: regression trend from `emos_reg` plus IDW of the station residuals, analogous to regression kriging.
   * Negative interpolated c² or d² are floored at 1e-6.
3. **`samos_lin`** (linear SAMOS):
   * The climatology (mean and sd of obs and of the t2m ensemble mean) is the attribute-based ridge climatology of the Stage-4 pilot, fitted on training stations. It uses the `Climatology` class with the pilot's `clim_ridge` and `clim_max_rows`.
   * Standardised anomalies: z_y, z_f = (f − m_f)/s_f and z_s = s/s_f.
   * A global EMOS per lead is fitted in anomaly space, z_y ~ N(a + b·z_f, c² + d²·z_s²), and back-transformed.
   * This is the linear counterpart of samos_mlp.
4. **Boosted-EMOS iteration sensitivity**, EUPPBench spatial (primary folds) first, then random:
   * One boosting path with maxit = 4000 per fold and lead, on the same data and settings as primary.
   * Held-out CRPS, spread–skill ratio and coverage are computed at m ∈ {100, 250, 500, 1000, 2000, 4000}, at the AIC-selected mstop, and at a **validation-selected m**: the path is fitted on ≤ 2014 and m is chosen by minimum CRPS on the *training stations'* 2015–2016 reforecasts.
   * Only the validation-selected value may be proposed as an alternative setting. The fixed-m test scores are sensitivity only.
* Datasets: EUPPBench and German, random and primary spatial folds, plus every partition of Section B (EUPPBench).
* These baselines are **descriptive comparators**. No superiority hypothesis is attached to them.

## D. Inference (M1, M2, M5)
Unit of analysis: held-out-station forecasts, test period, pooled over folds, exactly as the reported CRPS.
For a pair (A, B), Δ = CRPS_A − CRPS_B (negative means A is better).

**Primary: station-cluster bootstrap with nested seed resampling.**
* In each of B = 2000 replicates:
  1. For each method, resample its seeds with replacement from its available seeds and rebuild the seed ensemble.
  2. Resample stations with replacement.
  3. Compute Δ as a row-weighted ratio (sum of row differences / sum of rows), which reproduces the pooled mean.
* Report percentile 95% CIs and P(Δ < 0).
* Deterministic methods have no seed resampling.
* The un-nested station bootstrap (fixed full ensembles, B = 10000) is also reported.

**Co-primary: fold-block bootstrap.**
* Resample whole folds with replacement (spatial regions; K = 7 blocks), nested with seed resampling, B = 2000, row-weighted.
* With only 7 blocks this interval is coarse. It is reported, and it is used in the wording rules of Section F.

**Rank tests.**
* Fold-level: exact two-sided sign test and Wilcoxon signed-rank test on the K per-fold mean Δ.
* Station-level: the same two tests on the per-station mean Δ. Stations are spatially correlated, so these are descriptive and are labelled as such.

**Diebold–Mariano on the daily series.**
* Series: the per-day (German: date; EUPPBench: initialisation date) mean Δ over held-out stations.
* Long-run variance: Bartlett kernel with **automatic bandwidth by Newey & West (1994)**. This is primary. The Andrews (1991) AR(1) plug-in is a sensitivity.
* **HLN small-sample correction** with h = maximum lead in days (German 2, EUPPBench 5), and a t(n−1) reference.
* Also reported:
  * fixed lags 5, 20 and 40;
  * the Kiefer–Vogelsang fixed-b 5% decision for the automatic bandwidth;
  * moving-block bootstrap CIs of the daily mean with block lengths **7, 14 and 30 days** (10000 replicates).

**Multiplicity.** BH at q = 0.05 within each (dataset, fold type) family. The family is the pair list below, tested with the primary DM (automatic NW + HLN).

**Ranking probabilities.** From the nested station bootstrap:
* pairwise P(A < B), i.e. the fraction of replicates with Δ < 0;
* P(method has rank 1) among the methods present.

From seeds alone (no station resampling):
* P(single-seed CRPS of A < single-seed CRPS of B) over all seed pairs;
* the per-seed CRPS list for every method.

**Pair list** (primary families):
* each of nn_unk, nn_knn, nn_noemb, nn_attr, nn_hybrid, nn_hybrid_res, drn_lak, gnn_geo, samos_mlp, samos_attn, emos_regidw and samos_lin **vs emos_bst_gl**;
* drn_lak, gnn_geo, nn_unk and samos_mlp **vs nn_hybrid_res**;
* emos_regidw vs emos_gl.

**Where this is applied.**
* Existing 3- and 10-seed predictions (Stage 2/3/4). This runs first, so that results exist even if later blocks are cut.
* 10-seed EUPPBench and German spatial predictions.
* Every partition of Section B.

## E. Neighbour availability fix (m1a)
* Stage-4 `Fold.avail = isfinite(y) & isfinite(ens mean)` is used as neighbour availability in gnn_geo aggregation (`stage4_run.py:49`) and in samos_attn neighbour selection (`stage4_pilot.py:144`). A neighbour therefore counted as available only when its *observation* existed.
* The fix defines neighbour availability as **forecast availability** only. Training loss and scoring still use rows where the observation exists.
* **EUPPBench**: the preprocessed file dropped rows with missing observations. `scripts/stage5_preprocess_fcavail.py` re-reads the local EUPPBench zarr copy and stores the forecast features of the missing-observation (slice, station) pairs of included stations. These are forecast inputs only, with no observations.
* **German RL18**: the published file contains forecast rows only where an observation exists, so forecast availability cannot be recovered from it. The German runs are therefore unaffected by construction (in our data, forecast availability = observation availability). This is disclosed; nothing is rerun.
* Reruns on EUPPBench, seeds 0–2, with the fixed availability:
  * gnn_geo on spatial and random folds;
  * samos_attn on spatial folds.
* Impact: Δ CRPS (fixed − original) per fold type with a station bootstrap CI, plus the fraction of held-out test (slice, station) cases whose neighbour set changes.
* **Rule:** the original predictions stay primary. If |Δ| ≥ 0.005 °C for a method, the fixed numbers are reported alongside.

## F. Pre-stated decision rules for the "random vs spatial" narrative
* E denotes emos_bst_gl.
* Flexible nets: F ∈ {drn_lak, gnn_geo}, the published competitors that win on random folds.
* Secondary set: {nn_unk, nn_noemb, nn_attr, nn_hybrid, samos_mlp}.

For each F, define the per-station difference-in-differences
  DiD_F = (F − E)_spatial − (F − E)_random.
Every station is held out exactly once in each fold design, so this is paired per station. Its inference uses the seed-nested station bootstrap, with seeds resampled independently for the random and spatial runs.

**C1. "Random station splits overstate cold-start transfer of flexible networks relative to EMOS in the sparse (EUPPBench) network"**
* **Supported** if, on EUPPBench primary spatial folds with 10 seeds:
  1. both DiD_drn and DiD_gnn are > 0 with station-bootstrap 95% CI excluding 0; **and**
  2. for each F, the point estimate (F − E)_partition − (F − E)_random is > 0 in ≥ 4 of the 6 partitions of Section B.
* **Contradicted** if either DiD CI lies entirely below 0, or if either F has DiD ≤ 0 in ≥ 4 of the 6 partitions.
* **Otherwise: inconclusive.** Wording is then limited to "in this benchmark, the primary spatial partition suggests …", with the interval given.
* A fold-block bootstrap CI that includes 0 does **not** overturn "supported", but the text must add that the effect is concentrated in a few regions and name them from the per-fold table.

**C2. "The ranking of F and E reverses between random and spatial folds"** (EUPPBench) may be stated only if both of these hold:
* on random folds, F − E < 0 with station-bootstrap CI excluding 0;
* on spatial folds, F − E > 0 with station-bootstrap CI excluding 0.

Otherwise the wording is "F's advantage on random folds disappears on spatial folds (spatial difference not distinguishable from 0)", or the reverse, as appropriate.

**C3. Sparse vs dense.** "Specific to sparse networks" may be written only if C1 is supported on EUPPBench **and** the German DiD_F CI includes 0 or lies below 0 for both F. With a single sparse and a single dense network, any such statement must say that network density is confounded with domain, lead times and predictors.

**C4. Pairwise method claims** ("A better than B"), in any setting:
* **Claim** only if the seed-nested station-bootstrap CI excludes 0 **and** the BH-adjusted primary DM (automatic NW + HLN) rejects.
* If the station CI excludes 0 but the fold-block CI includes 0, the wording is "better on average over stations, not consistently across regions".
* If the station CI includes 0, the wording is "statistically tied / no evidence of a difference".

**C5. Seeds.** If P(A < B) from the nested bootstrap lies in [0.1, 0.9], no ordering of A and B is stated.

**C6. New baselines.** If emos_regidw or samos_lin has a lower spatial CRPS than every network on EUPPBench, this is reported as such, and the baseline comparison and conclusions must include it. No outcome of Stage 5 is allowed to remove a previously reported result.

## G. Compute plan and runtime estimate
* **Machine:** local 8-core CPU box, shared. **At most 6 threads**: 2 worker processes × 3 threads, at most 2 heavy processes at a time (≈ 4–5 GB RSS each).
* **Launcher:** a single Python orchestrator (`scripts/stage5_launch.py`) with a prioritised job queue with dependencies, detached with `setsid nohup`. Logs go to `results/stage5/logs/`. Every job is resumable: finished (method, fold, seed) units are skipped.

Estimates come from existing per-run logs: Stage-3 spatial nets at 3 threads 28–39 s per run; Stage-4 EUPPBench at 4 threads drn 62 s, gnn 93 s, samos_mlp 21 s, samos_attn 119 s; German at 4 threads drn 35 s, gnn 135 s, samos_mlp 6 s, samos_attn 89 s. Stage-4 times are ×1.25 for 3 threads, and there is about 1 min of data loading per fold and job.

| priority | block | lane-hours (est.) |
|---|---|---|
| 1 | D on existing predictions (inference) | 0.3 |
| 2 | A: EUPPBench spatial seeds 3–9, Stage-3 nets (5 variants × 49 runs) | 2.4 |
| 3 | C: new statistical baselines, both datasets, random + spatial | 0.5 |
| 4 | A: EUPPBench spatial seeds 3–9, drn_lak, gnn_geo, samos_mlp, samos_attn | 5.0 |
| 5 | C: bst iteration sensitivity, EUPPBench spatial | 0.9 |
| 6 | interim evaluation (EUPPBench 10 seeds + inference) | 0.3 |
| 7 | B: 6 partitions × (bst baselines, stat baselines, 5 nets × 3 seeds) | 11.4 |
| 8 | A: German spatial seeds 3–9 (Stage-2 nets; drn, samos_mlp; gnn, samos_attn) | 5.6 |
| 9 | E: m1a preprocessing + reruns (gnn spatial/random, samos_attn spatial) | 3.5 |
| 10 | C: bst iteration sensitivity, EUPPBench random | 0.9 |
| 11 | final evaluation | 0.5 |

* **Total** ≈ 31 lane-hours on 2 lanes ≈ **16–18 h wall**, including contention.
* **Scope-cut rule (pre-stated):** if 22 h of wall time have passed since launch, jobs not yet started are dropped in reverse priority order (11 always runs). Dropped blocks are documented in `NOTES_stage5.md`. Any partition of Section B that is incomplete is excluded as a whole, not partially.

## H. Things that are not changed
* Data, folds, periods, hyper-parameters and early stopping are all unchanged. Early stopping still uses seen-station validation; the nested variant (m2) is not done here.
* No test-period score is used to choose any setting.
* Smoke tests of the new scripts use tiny subsamples with outputs written to `/tmp`, and are discarded.
