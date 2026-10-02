# Reproducibility of code and evaluation outputs

## Inspection and replay

Configurations, internal protocol copies, aggregate CSV/JSON results and existing audit records are available for inspection. Independent recomputation additionally requires provider observations, the preserved prediction arrays, exact row alignment and preparation of the evaluation workspace. Prediction archive upload is pending; see [PREDICTIONS.md](PREDICTIONS.md).

Observation-free `test_index.npz` files preserve identifiers and row order. They cannot replace `test_meta.npz`, which also supplies observed values and, for some tools, predictors or station attributes. Obtain these fields from the providers and verify index order before scoring.

## Existing numerical audit

The 2026-09-29 audit reports 70 score combinations, 1,986 matching score/test fields and 112 matching core DRN/GNN difference-in-differences fields, including six additional spatial partitions. Its records are under `handoff/original_prediction_audit/`. Copying these records does not constitute rerunning that audit.

The full auxiliary pair/ranking/fold-block/sensitivity suite was not regenerated in that audit. Post hoc matched-seed, forecast-only and regional-block DiD intervals are stored under `results/review_supplementary/`. Their primary DiD intervals reproduce `results/stage5/eval/final/decisions.json` before adding the extra contrasts. Agreement does not remove station dependence, unmatched predictors, seen-station validation, unequal ensemble sizes or overlapping additional partitions.

The audit entry points retain a historical development commit and workspace assumptions. Those must be routed to the correct frozen source and restored data before running. The numerical reference path in the public score audit is `provenance/derived_numbers.json`.

## File integrity

The intended archive inventory has 4,313 prediction NPZ files, three observation-free indexes and four primary fold CSVs. Every prediction file except one retains its original file bytes. One Stage-2 reference file omits the observation field while preserving predictive array payloads. The manifests record both file hashes and removed fields. `scripts/restore_public_predictions.py` verifies archive parts, combined archives and individual members.

These are file-integrity checks, separate from scientific recomputation. No training, evaluation or bootstrap was run while preparing this code snapshot.

## Analysis plans and scope

Protocol analysis definitions, dates and decision rules are retained. References to an unavailable review document and presentation-specific instructions were converted to technical wording. Such edits do not change model settings or numerical outputs. Historical terms such as "registered" refer to internally recorded plans, without independent external preregistration. The new code history does not establish the chronology of the original development history.

Data-provider licences remain applicable. No archival DOI is assigned to these materials.
