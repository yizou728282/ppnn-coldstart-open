# Stage 4 protocol B: pilot of the new method (pre-registered)

Registered on 2026-09-26, **before any pilot training run**. The JSON (`stage4_pilot.json`) is authoritative. Implementation: `src/ppnn_residual/stage4_pilot.py`, run by `scripts/stage4_run.py`.

## Motivation
The exploratory diagnosis (`NOTES_stage4_analysis.md`, not pre-registered) found that new-station errors on EUPPBench come from unconstrained extrapolation in station-attribute space, and that an ensemble-anchored mean limits that error. The pilot tests a design that (i) models climatologically standardised anomalies and (ii) borrows information from nearby *training* stations instead of extrapolating from attributes.

## Method `samos_attn`
1. **SAMOS-style standardisation.** Fitted on training stations and the fit period only.
   * A ridge regression gives the climatological mean and log-variance of both the observations and the t2m ensemble mean.
   * Regressors: [1, lat, lon, alt, alt−orog] × [1, first and second annual harmonics]. For EUPPBench, lead one-hot × [1, sin, cos] is added.
   * The fitted climatology gives m_y, s_y, m_f and s_f for every (time, station), held-out stations included.
2. **Output in standardised space.** The mean is anchored at the forecast anomaly:
   * μ = m_y + s_y·(z_f + g)
   * σ = s_y·√(softplus(c)² + softplus(d)²·z_s²)
3. **Attention over the K = 8 nearest available training stations**, with the target itself excluded, plus a learned "null" slot.
   * Each neighbour token contains: the neighbour's standardised forecast and spread; its anomaly difference to the target; the distance; the altitude difference; the alt−orog difference; and a learned 4-d embedding of that training station. The embedding is the only channel through which observation-derived station information enters.
   * The target's own embedding is never used.
4. **Random station masking during training** creates pseudo-new stations:
   * per batch, a fraction U(0, 0.5) of training stations is dropped from the neighbour pool;
   * per sample, with probability 0.5, all training stations within r ~ U(0, 300) km of the target are dropped.
5. **Training settings:** hidden 128, attention dimension 32, Adam lr 1e-3, batch 4096, at most 30 epochs, patience 3. Nothing is tuned.

**Ablation `samos_mlp`:** the same standardisation and output, with no neighbours, attention or masking.

## Design
* **Phase 1:** German data, random and spatial folds, seeds 0–2, both methods.
* **Phase 2:** EUPPBench with the same design. It runs only if Phase 1 is *promising*, meaning samos_attn has a lower German spatial-fold CRPS than the best primary comparator (point estimate).
* **Primary comparators:** nn_unk, nn_knn and emos_bst_gl. Networks are compared as their seeds 0–2 ensembles.
* **Strict secondary comparators:** the primary comparators plus nn_noemb, nn_hybrid, nn_hybrid_res, drn_lak and gnn_geo. These are descriptive only.
* **Tests:** as in Stage 2/3 (DM on daily means, NW lag 5, 7-day block bootstrap). BH correction is applied over 6 tests: samos_attn vs each primary comparator, on each fold type.

## Go / no-go (decided before any pilot run)
**GO** requires all of the following on the German data:
1. On **spatial** folds, compared with the best primary comparator (lowest CRPS of the three), samos_attn is:
   * BH-significant;
   * with a bootstrap CI that excludes 0;
   * better.
2. On **random** folds, samos_attn has the same sign (lower CRPS) against the best primary comparator on those folds.

Otherwise the result is **NO-GO**. The outcome is reported either way. If Phase 2 runs, EUPPBench is judged by the same rule as a replication check.

## Disclosure
* The 2-epoch smoke test used 300 fit and 300 early-stopping slices, and one spatial fold per dataset (German fold 6, EUPPBench fold 4).
* It printed the held-out test CRPS of these untrained models:
  * German: samos_attn 1.04, samos_mlp 1.06
  * EUPPBench: 1.82 and 1.85
* No design choice or hyperparameter was changed after this output. The smoke outputs were not saved.
