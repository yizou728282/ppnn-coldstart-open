#!/usr/bin/env bash
# Full reproduction: download + verify data, tests, baselines, 5 networks x 3 seeds, evaluation.
set -euo pipefail
cd "$(dirname "$0")"
python scripts/download_data.py
python -m pytest -q tests
python scripts/run_experiment.py --baselines
python scripts/run_experiment.py --networks all
python scripts/evaluate.py | tee results/evaluate_stdout.txt
