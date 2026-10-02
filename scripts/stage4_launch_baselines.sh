#!/usr/bin/env bash
# Stage 4 step 3 (protocols/stage4_baselines.json) on the local CPU box: two lanes x 4 torch threads. Resumable.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/workspace/ppnn_venv/bin/python}   # override with e.g. PY=python
L=results/stage4/logs_launcher; mkdir -p "$L"
echo "start baselines $(date -Is)" >> "$L/launcher.log"
lane() {  # dataset
  export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
  for m in drn_lak gnn_geo; do for ft in random spatial; do
    $PY scripts/stage4_run.py --dataset $1 --fold-type $ft --methods $m --threads 4 >> "$L/$1.log" 2>&1
    echo "$1 $ft $m exit $? $(date -Is)" >> "$L/launcher.log"
  done; done
}
lane german & A=$!
lane euppbench & B=$!
wait $A $B
echo "baselines done $(date -Is)" >> "$L/launcher.log"
