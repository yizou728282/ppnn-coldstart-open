# Stage 4 protocol A: Lakatos-style baselines on both new-station benchmarks (pre-registered)

Registered on 2026-09-26, **before any Stage-4 model was trained**. Code smoke tests on tiny subsamples are allowed; their outputs are discarded and never reported. The JSON (`stage4_baselines.json`) is authoritative.

## Why these baselines
The closest prior work is Lakatos (2026, arXiv:2609.07512). It post-processes German ECMWF t2m at unobserved stations using EMOS-R/C/L, a DRN, a Transformer and GNN-Geo, with a single 70/30 station split, no spatial blocking and no station embeddings. We add two of its methods to our benchmarks, for direct comparison. We add the DRN and GNN-Geo but not the Transformer. The Transformer is beyond the scope of this step, and that is stated as a limitation.

The paper's details were checked in the PDF: Sec. 3.2.1–3.2.2, Table A1 (predictors) and Table A2 (hyperparameters).

## drn_lak (Lakatos DRN)
* **Model:** a 1-hidden-layer MLP with no station embedding and a Gaussian output, trained on CRPS.
  * **This is the same model class as our `nn_noemb`.**
  * What we match to Lakatos:
    * hyperparameters (T2M, extended feature set): hidden 256, dropout 0.3, Adam lr 0.01, batch 4800;
    * inputs: lat, lon, alt and orog; sin(doy), doy and month; no alt−orog; no land-use.
  * The meteorological predictor sets are the ones our two datasets provide.
* **Early stopping:** as in Stage 2/3 (at most 30 epochs, patience 3). The paper does not state its epoch settings.
* **Deviation from the paper:** no rolling daily refit. Their 3283-day window is roughly the whole history.

## gnn_geo (Lakatos GNN-Geo)
* **Architecture:** GraphSAGE with mean aggregation and one layer (hidden 256), dropout 0.2, Adam lr 0.02, batches of 256 time slices.
* **Graph:**
  * edge if the haversine distance is ≤ 50 km; altitude is ignored;
  * binary edges, no self-loops;
  * isolated nodes use only the root term.
* **Node features:** the same as drn_lak.
* **Inductive use:**
  * During fit and early stopping, the graph contains only the training stations of the fold.
  * At test time, all stations are nodes. A held-out station aggregates the observation-free forecast features of every station within 50 km. Only held-out nodes are scored.
* **Epochs:** at most 100, patience 5. With 256-slice batches an epoch has only about 12–70 optimiser steps.
* **Deviations from the paper:**
  * no 361-day rolling refit;
  * unobserved stations are not part of the training graph, which is stricter than the paper.

## Design
* **Folds and periods:** identical to Stage 2 (German) and Stage 3 (EUPPBench).
* **Seeds:** 0–2 on both fold types. This is modest, to save time.
* **Fairness on random folds:** existing networks are compared through their seeds 0–2 ensembles. Their 10-seed numbers are shown for reference only.
* **Metrics:** the same as Stage 2/3.
* **Tests:** descriptive only. DM tests with a block bootstrap and BH correction within each (dataset, fold type) for:
  * gnn_geo vs {nn_unk, nn_knn, emos_bst_gl, drn_lak}
  * drn_lak vs {nn_noemb, nn_unk, emos_bst_gl}
* **No superiority claims** are attached to these baselines, and all results are reported.
