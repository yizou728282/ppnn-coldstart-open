#!/usr/bin/env bash
# Stage 4 step 4 (protocols/stage4_pilot.json). Two lanes that start when the corresponding baseline lane has finished
# (at most 2 heavy processes on the box). Jobs are claimed with mkdir locks; everything is resumable.
# Phase 2 (EUPPBench) runs only if scripts/stage4_evaluate.py --promising-only german prints PROMISING.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/workspace/ppnn_venv/bin/python}   # override with e.g. PY=python
L=results/stage4/logs_launcher; LK=results/stage4/locks; mkdir -p "$L" "$LK"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
log() { echo "$* $(date -Is)" >> "$L/launcher.log"; }
wait_for() { until grep -q "$1" "$L/launcher.log"; do sleep 60; done; }
run_jobs() {  # lane-name job...
  local lane=$1; shift
  for job in "$@"; do
    set -- $job  # dataset ftype
    if mkdir "$LK/$1_$2" 2>/dev/null; then
      $PY scripts/stage4_run.py --dataset $1 --fold-type $2 --methods samos_mlp samos_attn --threads 4 >> "$L/pilot_$1.log" 2>&1
      log "pilot $1 $2 exit $? ($lane)"; touch "$LK/$1_$2/done"
    fi
  done
}
wait_all() { for j in "$@"; do until [ -e "$LK/$j/done" ]; do sleep 60; done; done; }
log "pilot launcher start"
( wait_for "german spatial gnn_geo exit"; run_jobs P1 "german random" "german spatial" ) &
( wait_for "euppbench spatial gnn_geo exit"; run_jobs P2 "german spatial" "german random" ) &
wait_all german_random german_spatial
$PY scripts/stage4_evaluate.py --promising-only german > "$L/pilot_phase1_eval.txt" 2>&1
if tail -n 1 "$L/pilot_phase1_eval.txt" | grep -qx PROMISING; then
  log "phase 1 PROMISING -> phase 2 (EUPPBench)"
  ( wait_for "euppbench spatial gnn_geo exit"; run_jobs P1b "euppbench random" "euppbench spatial" ) &
  ( wait_for "euppbench spatial gnn_geo exit"; run_jobs P2b "euppbench spatial" "euppbench random" ) &
  wait
else
  log "phase 1 NOT PROMISING -> no phase 2"
fi
wait
$PY scripts/stage4_evaluate.py > "$L/final_eval.txt" 2>&1
log "pilot launcher done (final eval exit $?)"
