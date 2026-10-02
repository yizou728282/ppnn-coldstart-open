# Preserved original prediction outputs

**Archive upload pending.** The target download location is the [v1.1.0-original-predictions Release](https://github.com/yizou728282/ppnn-coldstart-open/releases/tag/v1.1.0-original-predictions). Prediction archives are not yet available there. The manifests in `prediction_manifests/` record the intended inventory and checksums.

After all nine assets are uploaded, download every `predictions_*.tar.gz.part*` asset, both `MANIFEST_*.json` assets and `SHA256SUMS.txt` into one directory.

Large binaries are Release assets rather than Git-tracked files. The archives preserve the original `results/.../preds/...` paths; filenames retain method, fold and seed identifiers. Archive compression and splitting have changed, but the manifests record both the original and public SHA-256 of every included file.

The preserved inventory contains 4,313 prediction NPZ files: 1,794 from Stage-2/3/4 and 2,519 from Stage-5. Of these, 3,489 have an individual seed identifier in the filename; the remaining files include baselines, references and seen-station accumulators. There are also three observation-free test indexes and four primary fold CSVs. Six archive shards total 4,940,105,566 bytes (about 4.94 GB).

## Restore and verify

From the root of a clone of this repository:

```bash
python scripts/restore_public_predictions.py --assets /path/to/downloads --verify-only
python scripts/restore_public_predictions.py --assets /path/to/downloads
```

The first command checks every shard, every archive member and the combined archive hash without extraction. The second repeats those checks and restores predictions, row indexes and primary fold CSVs. It uses the Python standard library and works on Windows, Linux and macOS. This verifies preserved file identity; it does not recalculate scientific results.

## Observations and index alignment

- Raw observations, predictor datasets, checkpoints, training logs and source credentials are not included.
- Each preserved `test_meta.npz` was replaced by an observation-free `test_index.npz` containing only available `station`, `date`, `lead`, `init` and `unseen` identifiers. Object-typed identifier arrays were converted to Unicode. Row order was preserved.
- One Stage-2 reference file, `results/stage2/preds/reference/emos_loc_bst_seen.npz`, originally contained observations under `y`. The public file omits `y`; all other `.npy` member payloads are preserved byte-for-byte. This exception is recorded in `removed_fields` and the two file hashes. All other prediction NPZ files are copied byte-for-byte.
- Primary random/spatial fold assignments are included in the Stage-2/3 CSVs. Additional Stage-5 partitions are defined by `protocols/stage5_supplementary.json` and `scripts/stage5_common.py`; they are not supplied as original CSVs by the Stage-5 source archive.
- German Stage-2 predictions generally cover the complete test row order; select the appropriate held-out station mask for each fold. EUPPBench neural predictions and Stage-4/5 predictions generally contain only held-out rows in test-index order. Baseline and seen-station files have different shapes. Follow the loaders in `scripts/stage2_run.py`, `scripts/stage3_run.py`, `src/ppnn_residual/stage4_data.py` and the original audit code; do not concatenate folds by filename alone.
- `init` in the EUPPBench Stage-3 index represents days since the Unix epoch. `lead` is in hours. `mu` and `sigma` are the Gaussian predictive location and scale in the study's temperature units; `mu11`/`sigma11` retain the alternative ensemble-input predictions where present. `sum_mu`, `sum_sigma`, `seeds` are seen-station accumulators, not individual seed files.

Obtain observations from the providers linked in [README.md](README.md), preprocess them using the original code and confirm exact identifier order against `test_index.npz` before scoring. An index file cannot replace `test_meta.npz` in existing evaluation code: the latter also needs `y` and, for some tools, station attributes or forecast features. The preserved audit entry points also reference the original development history and audit workspace; adapt their source/workspace routing to this public snapshot before executing them. Archive access does not by itself establish a fully validated independent replay of every statistical output.

## Provenance and packaging

Source asset SHA-256 values are recorded in the manifests. Stage-2/3/4 outputs came from the preserved `original-preds-stage2-4` archive; Stage-5 outputs came from `stage5-preds-2026-09-27`. The relevant research snapshot is commit `7213a55cfef8b195d8d9b283941933b091553130` in the original development repository. The packaging program is preserved at `scripts/package_preserved_predictions.py` for inspection.

Packaging involved checking archive identity, inspecting prediction-field names, removing observation fields from the identified files, producing indexes and repackaging. No predictions were fitted or regenerated, no metrics were recomputed, and no reported result was changed. The original numerical audit's scope and limits remain documented in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

The implementation's MIT licence does not replace third-party dataset terms. Access to the observation data remains governed by the original providers. No archival DOI has been assigned.
