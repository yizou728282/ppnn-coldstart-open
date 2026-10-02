"""Stage 5 orchestrator (protocols/stage5_supplementary.json, section G).
Two workers x 3 threads (<= 6 threads total), prioritised job queue with dependencies. Resumable: a job with a
state/<id>.done file (exit 0) is skipped; the underlying scripts also skip finished units.
Scope cut: after CUT_H hours since the first launch, unstarted jobs (except 'final_eval') are dropped.

  setsid nohup /workspace/ppnn_venv/bin/python scripts/stage5_launch.py > results/stage5/logs/launcher_stdout.txt 2>&1 &
Monitor: results/stage5/logs/launcher.log (job start/end), results/stage5/logs/jobs/<id>.log (job output)
"""
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = os.environ.get("PY", "/workspace/ppnn_venv/bin/python")
L = ROOT / "results/stage5/logs"; J = L / "jobs"; S = L / "state"
for d in (L, J, S):
    d.mkdir(parents=True, exist_ok=True)
THREADS = 3; WORKERS = 2; CUT_H = 22.0
R = "scripts/stage5_run.py"
SEEDS_NEW = "3 4 5 6 7 8 9"
PARTS = ["km7_s2027", "km7_s2028", "km7_s2029", "loco", "km5_s2026", "km10_s2026"]

JOBS = []  # (id, command, deps) in priority order


def job(i, cmd, deps=()):
    JOBS.append((i, cmd, list(deps)))


job("inf_existing", f"scripts/stage5_inference.py --tag existing --seed-cap-spatial 3 --no-partitions")
job("eupp_seeds_s3a", f"{R} seeds3 --variants unk hybrid_res --seeds {SEEDS_NEW}")
job("eupp_seeds_s3b", f"{R} seeds3 --variants noemb attr hybrid --seeds {SEEDS_NEW}")
job("stat_eupp_spatial", f"{R} statbase --dataset euppbench --design spatial")
job("stat_eupp_random", f"{R} statbase --dataset euppbench --design random", ["stat_eupp_spatial"])
job("stat_german_spatial", f"{R} statbase --dataset german --design spatial")
job("stat_german_random", f"{R} statbase --dataset german --design random", ["stat_german_spatial"])
for m in ["drn_lak", "samos_mlp", "gnn_geo", "samos_attn"]:
    job(f"eupp_seeds_{m}", f"{R} stage4 --dataset euppbench --design spatial --methods {m} --seeds {SEEDS_NEW}")
job("bstsens_spatial", f"{R} bstsens --fold-type spatial")
EUPP_SEED_JOBS = ["eupp_seeds_s3a", "eupp_seeds_s3b", "eupp_seeds_drn_lak", "eupp_seeds_samos_mlp", "eupp_seeds_gnn_geo",
                  "eupp_seeds_samos_attn", "stat_eupp_spatial", "stat_eupp_random", "stat_german_spatial", "stat_german_random"]
job("inf_interim", "scripts/stage5_inference.py --tag interim --no-partitions", EUPP_SEED_JOBS)
for p in PARTS:
    job(f"part_{p}_bst", f"{R} part_bst --pid {p}")
    job(f"part_{p}_stat", f"{R} statbase --dataset euppbench --design part_{p}", ["stat_eupp_spatial"])
    job(f"part_{p}_nets3", f"{R} part_nets3 --pid {p} --variants unk hybrid_res --seeds 0 1 2")
    job(f"part_{p}_nets4", f"{R} stage4 --dataset euppbench --design part_{p} --methods drn_lak gnn_geo samos_mlp --seeds 0 1 2")
job("german_seeds_s2", f"{R} seeds2 --variants unk noemb attr hybrid hybrid_res --seeds {SEEDS_NEW}")
job("german_seeds_drn_samosmlp", f"{R} stage4 --dataset german --design spatial --methods drn_lak samos_mlp --seeds {SEEDS_NEW}")
job("german_seeds_gnn", f"{R} stage4 --dataset german --design spatial --methods gnn_geo --seeds {SEEDS_NEW}")
job("german_seeds_samos_attn", f"{R} stage4 --dataset german --design spatial --methods samos_attn --seeds {SEEDS_NEW}")
job("m1a_prep", "scripts/stage5_preprocess_fcavail.py --zarr /workspace/euppbench/zarr")
job("m1a_gnn_spatial", f"{R} m1a --fold-type spatial --methods gnn_geo --seeds 0 1 2", ["m1a_prep"])
job("m1a_gnn_random", f"{R} m1a --fold-type random --methods gnn_geo --seeds 0 1 2", ["m1a_prep"])
job("m1a_samos_attn_spatial", f"{R} m1a --fold-type spatial --methods samos_attn --seeds 0 1 2", ["m1a_prep"])
job("bstsens_random", f"{R} bstsens --fold-type random")
job("final_eval", "scripts/stage5_inference.py --tag final", [j[0] for j in JOBS])

lock = threading.Lock()
claimed = set()


def log(msg):
    with lock:
        with open(L / "launcher.log", "a") as f:
            f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} {msg}\n")


def done(i):
    f = S / f"{i}.done"
    return f.exists() and f.read_text().strip() in ("0", "skipped")


def finished(i):  # done, failed or skipped: dependents may proceed (final_eval uses whatever exists)
    return (S / f"{i}.done").exists()


def start_time():
    f = S / "launch_time"
    if not f.exists():
        f.write_text(str(time.time()))
    return float(f.read_text())


T0 = start_time()


def next_job():
    with lock:
        for i, cmd, deps in JOBS:
            if i in claimed or finished(i):
                continue
            if i != "final_eval" and (time.time() - T0) / 3600 > CUT_H:
                (S / f"{i}.done").write_text("skipped"); claimed.add(i)
                with open(L / "launcher.log", "a") as f:
                    f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} SCOPE-CUT skip {i}\n")
                continue
            if all(finished(d) for d in deps):
                claimed.add(i)
                return i, cmd
        return None


def worker(w):
    base = {k: v for k, v in os.environ.items() if not k.startswith("STAGE5_")}  # never inherit smoke-test settings
    env = dict(base, OMP_NUM_THREADS=str(THREADS), MKL_NUM_THREADS=str(THREADS), OPENBLAS_NUM_THREADS=str(THREADS),
               PYTHONUNBUFFERED="1")
    while True:
        nj = next_job()
        if nj is None:
            if all(finished(i) for i, _, _ in JOBS):
                return
            time.sleep(30); continue
        i, cmd = nj
        extra = f" --threads {THREADS}" if cmd.startswith(R) else ""
        log(f"START {i} (worker {w}): {cmd}{extra}")
        t0 = time.time()
        with open(J / f"{i}.log", "a") as f:
            rc = subprocess.call(f"{PY} {cmd}{extra}", shell=True, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env=env)
        (S / f"{i}.done").write_text(str(rc))
        log(f"END {i} exit {rc} ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    log(f"launcher start pid {os.getpid()} ({len(JOBS)} jobs, {WORKERS} workers x {THREADS} threads, cut {CUT_H} h after {time.ctime(T0)})")
    ts = [threading.Thread(target=worker, args=(w,)) for w in range(WORKERS)]
    for t in ts:
        t.start(); time.sleep(5)
    for t in ts:
        t.join()
    log("launcher done")
