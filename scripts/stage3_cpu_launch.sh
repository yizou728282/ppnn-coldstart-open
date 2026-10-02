#!/usr/bin/env bash
# Stage 3 on the local 8-core CPU box (replaces the Colab T4 run; see NOTES_stage3.md).
# Two lanes, at most two big processes at a time (~4.2 GB RSS each; box has 15 GB, shared):
#   lane A: random-fold networks, 5 torch threads
#   lane B: spatial-fold baselines -> local-bst reference (3 BLAS threads) -> spatial-fold networks (3 torch threads)
# Everything is resumable (finished units are skipped). Afterwards: evaluation.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/workspace/ppnn_venv/bin/python}   # override with e.g. PY=python
DATA=data/stage3/stage3_euppbench.npz
OUT=results/stage3
L=$OUT/cpu_logs; mkdir -p "$L"
echo "start $(date -Is)" >> "$L/launcher.log"
(
  OMP_NUM_THREADS=5 MKL_NUM_THREADS=5 OPENBLAS_NUM_THREADS=5 \
  $PY scripts/stage3_run.py --data $DATA --out $OUT --fold-type random \
      --variants unk noemb attr hybrid hybrid_res --device cpu --threads 5 >> "$L/networks_random.log" 2>&1
  echo "lane A exit $? $(date -Is)" >> "$L/launcher.log"
) &
A=$!
(
  export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
  $PY scripts/stage3_run.py --data $DATA --out $OUT --fold-type spatial --baselines --device cpu --threads 3 >> "$L/baselines.log" 2>&1
  $PY scripts/stage3_run.py --data $DATA --out $OUT --emos-loc-bst-reference --device cpu --threads 3 >> "$L/baselines.log" 2>&1
  echo "baselines exit $? $(date -Is)" >> "$L/launcher.log"
  $PY scripts/stage3_run.py --data $DATA --out $OUT --fold-type spatial \
      --variants unk noemb attr hybrid hybrid_res --device cpu --threads 3 >> "$L/networks_spatial.log" 2>&1
  echo "lane B exit $? $(date -Is)" >> "$L/launcher.log"
) &
B=$!
wait $A $B
$PY scripts/stage3_evaluate.py --out $OUT --fig figures/stage3 >> "$L/evaluate.log" 2>&1
echo "evaluate exit $? $(date -Is)" >> "$L/launcher.log"
