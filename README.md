# Ensemble temperature post-processing: code and evaluation outputs

Implementation, configurations, internally recorded analysis plans, aggregate evaluation outputs and numerical audit records for random and spatial station holdouts.

## Contents

| Directory | Contents |
|---|---|
| `src/`, `scripts/` | Data preparation, training, evaluation and prediction restoration |
| `configs/`, `protocols/` | Settings and internally recorded analysis plans |
| `results/` | Saved aggregate metrics and statistical comparisons |
| `analysis/` | Matched-seed, forecast-only and regional-block DiD calculations |
| `handoff/original_prediction_audit/` | Existing numerical audit code and results |
| `provenance/` | Saved numerical reference values |
| `prediction_manifests/` | File inventories and SHA-256 checksums |

## Prediction access

**Archive upload pending.** The preserved prediction archives have not yet been uploaded here. The checked-in manifests describe the intended files; they do not provide the prediction arrays. See [PREDICTIONS.md](PREDICTIONS.md) for the target download location, exact inventory and restoration instructions.

## Provider data

- German station data: [figshare, DOI 10.6084/m9.figshare.13516301.v1](https://doi.org/10.6084/m9.figshare.13516301.v1), CC BY 4.0.
- EUPPBench: [Zenodo, DOI 10.5281/zenodo.7708362](https://doi.org/10.5281/zenodo.7708362). Forecasts are CC BY 4.0; observation terms are supplied with that record.

Provider observations, predictor datasets and model checkpoints are obtained or rebuilt separately. The MIT code licence does not override dataset terms.

## Setup

```bash
python -m venv .venv
# Activate the environment using the command for your operating system.
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pytest -q tests
python scripts/download_data.py
python scripts/stage3_download.py --out data/euppbench/zarr
python scripts/stage3_preprocess.py --zarr data/euppbench/zarr --out data/stage3/stage3_euppbench.npz
```

Follow `protocols/` for stage settings and decision rules. Shell launchers use their original interpreter default; set `PY=python` when needed. The recorded runs used CPUs. Bitwise equality across hardware and thread settings is not guaranteed.

Protocol copies are internal working plans. They do not establish independent external preregistration. [REPRODUCIBILITY.md](REPRODUCIBILITY.md) describes the existing checks and the requirements for independent replay. No archival DOI is assigned to these materials.

Implementation licence: [MIT](LICENSE).
