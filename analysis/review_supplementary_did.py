"""Reviewer-requested supplementary DiD intervals, computed from preserved per-seed predictions only.

Nothing is trained. The seed-nested station bootstrap of scripts/stage5_inference.py is reused unchanged
(same Design class, same 2000 replicates, same shared station weights from default_rng(2026), same per-design
seed-resampling stream default_rng(99 + crc32(design) % 1000) consumed in the same method order). Extra method
entries needed here (e.g. "drn_lak@3") are appended AFTER the original method list, so the replicates of every
original method are identical to those behind results/stage5/eval/final/decisions.json; this is checked.

Analyses
  1. Matched-seed DiD: spatial ensemble restricted to seeds 0-2 ("F@3") versus the three-seed random ensemble.
  2. Forecast-only GNN (gnn_geo_fcavail, seeds 0-2, EUPPBench) DiD with the same procedure.
  3. Paired regional (cluster) block bootstrap of DiD: the same resampled clusters of stations weight both
     designs, nested with the same seed resampling. Clusters: the primary 7 spatial folds; countries
     (EUPPBench); k-means regions on station coordinates (repo's spatial_folds, seed 2026, n_init 20).

  PPNN_DATA_ROOT=/path/to/research/workspace OMP_NUM_THREADS=3 python analysis/review_supplementary_did.py --datasets euppbench
Output: results/review_supplementary/did_<dataset>.json (+ CSV written by the table script).
The data root must contain results/stage2, results/stage3 (test_meta.npz, folds_*.csv, preds/),
results/stage4/<ds>/preds and results/stage5 exactly as used by the final Stage-5 evaluation.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key, "3")
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("PPNN_DATA_ROOT", REPO)).resolve()
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts"))
import stage5_common as sc  # noqa: E402
import stage5_inference as si  # noqa: E402
from ppnn_residual.folds import spatial_folds  # noqa: E402

# point the unchanged evaluation code at the data workspace
sc.R5 = si.R5 = DATA / "results/stage5"
sc.OLD_DIR.update({"german": DATA / "results/stage2", "euppbench": DATA / "results/stage3"})
si.ROOT = DATA
OUT = REPO / "results/review_supplementary"
E = "emos_bst_gl"
KMEANS = {"euppbench": [10, 20, 30], "german": [20, 50, 100]}


def q(b):
    return float(np.quantile(b, .025)), float(np.quantile(b, .975))


def did(Dsp, Drd, Fsp, Frd, W=None):
    assert np.array_equal(Dsp.stations, Drd.stations) and np.array_equal(Dsp.cnt, Drd.cnt)
    spb = Dsp.boot_stat(Fsp, W) - Dsp.boot_stat(E, W)
    rdb = Drd.boot_stat(Frd, W) - Drd.boot_stat(E, W)
    b = spb - rdb
    sp = float(Dsp.cr[Fsp].mean() - Dsp.cr[E].mean()); rd = float(Drd.cr[Frd].mean() - Drd.cr[E].mean())
    lo, hi = q(b); slo, shi = q(spb); rlo, rhi = q(rdb)
    return {"F_spatial": Fsp, "F_random": Frd, "n_seeds_spatial": len(Dsp.single[Fsp]), "n_seeds_random": len(Drd.single[Frd]),
            "crps_F_spatial": float(Dsp.cr[Fsp].mean()), "crps_E_spatial": float(Dsp.cr[E].mean()),
            "crps_F_random": float(Drd.cr[Frd].mean()), "crps_E_random": float(Drd.cr[E].mean()),
            "did": sp - rd, "lo": lo, "hi": hi, "p_did_gt0": float((b > 0).mean()),
            "spatial_F_minus_E": sp, "spatial_F_minus_E_lo": slo, "spatial_F_minus_E_hi": shi,
            "random_F_minus_E": rd, "random_F_minus_E_lo": rlo, "random_F_minus_E_hi": rhi}


def cluster_weights(labels, reps, seed):
    G = int(labels.max()) + 1
    WC = np.random.default_rng(seed).multinomial(G, np.ones(G) / G, size=reps).astype(float)
    return WC[:, labels]


def block_schemes(ds, stations, Dsp, reps):
    st = pd.read_csv(sc.OLD_DIR[ds] / "folds_spatial.csv", index_col=0).reindex(stations)
    assert st[["lat", "lon"]].notna().all().all()
    out = {}
    # primary spatial folds; RNG 4242 as in the original fold-block DiD supplement (aggregation is equivalent)
    out["primary_folds_7"] = (Dsp.st_fold.astype(int), 4242)
    if ds == "euppbench":
        countries = sorted(st.country.unique())
        out[f"countries_{len(countries)}"] = (st.country.map({c: i for i, c in enumerate(countries)}).to_numpy(int), 5005)
    for K in KMEANS[ds]:
        f = spatial_folds(st[["lat", "lon"]], K, 2026, n_init=20)
        out[f"kmeans_{K}"] = (np.array([f[int(s)] for s in stations]), 31000 + K)
    return {k: {"labels": lab, "seed": seed, "W": cluster_weights(lab, reps, seed)} for k, (lab, seed) in out.items()}


def run(ds, reps):
    t0 = time.time()
    Ns = len(np.unique(np.load(sc.OLD_DIR[ds] / "test_meta.npz", allow_pickle=True)["station"]))
    W = np.random.default_rng(2026).multinomial(Ns, np.ones(Ns) / Ns, size=reps).astype(float)
    if ds == "euppbench":
        base = si.ALL + si.M1A + ["gnn_geo@3", "samos_attn@3"]
        meth = {"random": base, "spatial": base + ["drn_lak@3"]}
    else:
        meth = {"random": si.ALL, "spatial": si.ALL + ["drn_lak@3", "gnn_geo@3"]}
    Drd = si.Design(ds, "random", meth["random"], None, reps, W, 99)
    Dsp = si.Design(ds, "spatial", meth["spatial"], None, reps, W, 99)
    contrasts = {"primary_10v3_drn": ("drn_lak", "drn_lak"), "primary_10v3_gnn": ("gnn_geo", "gnn_geo"),
                 "matched_3v3_drn": ("drn_lak@3", "drn_lak"), "matched_3v3_gnn": ("gnn_geo@3", "gnn_geo")}
    if ds == "euppbench":
        contrasts["forecast_only_3v3_gnn"] = ("gnn_geo_fcavail", "gnn_geo_fcavail")
    schemes = block_schemes(ds, Dsp.stations, Dsp, reps)
    res = {"dataset": ds, "reps": reps, "data_root": DATA.name, "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "n_stations": int(len(Dsp.stations)), "n_rows": int(len(Dsp.y)),
           "block_schemes": {k: {"n_clusters": int(v["labels"].max() + 1), "rng_seed": v["seed"],
                                 "cluster_station_counts": np.bincount(v["labels"]).tolist()} for k, v in schemes.items()},
           "contrasts": {}, "checks": []}
    for name, (Fsp, Frd) in contrasts.items():
        r = {"station": did(Dsp, Drd, Fsp, Frd)}
        for k, v in schemes.items():
            b = did(Dsp, Drd, Fsp, Frd, v["W"])
            r[k] = {kk: b[kk] for kk in ("did", "lo", "hi", "p_did_gt0", "spatial_F_minus_E_lo", "spatial_F_minus_E_hi",
                                         "random_F_minus_E_lo", "random_F_minus_E_hi")}
        res["contrasts"][name] = r
        print(ds, name, {k: (round(v["did"], 5), round(v["lo"], 5), round(v["hi"], 5)) for k, v in r.items()}, flush=True)
    # --- reproduction checks against the delivered final reports
    final = DATA / "results/stage5/eval/final"
    dec = json.loads((final / "decisions.json").read_text())[ds]
    for F in ("drn_lak", "gnn_geo"):
        got = res["contrasts"][f"primary_10v3_{F[:3]}"]["station"]
        for key in ("did", "lo", "hi", "p_did_gt0", "spatial_F_minus_E_lo", "spatial_F_minus_E_hi", "random_F_minus_E_lo", "random_F_minus_E_hi"):
            res["checks"].append({"what": f"decisions.json {ds}/{F}/{key}", "actual": got[key], "expected": dec[F][key],
                                  "passed": bool(abs(got[key] - dec[F][key]) <= 1e-12)})
    if ds == "euppbench":
        for design, D in (("spatial", Dsp), ("random", Drd)):
            pr = pd.read_csv(final / f"pairs_euppbench_{design}.csv")
            row = pr[(pr.a == "gnn_geo_fcavail") & (pr.b == "gnn_geo@3")].iloc[0]
            bs = D.boot_stat("gnn_geo_fcavail") - D.boot_stat("gnn_geo@3")
            for key, val in (("st_nested_lo", np.quantile(bs, .025)), ("st_nested_hi", np.quantile(bs, .975))):
                res["checks"].append({"what": f"pairs_euppbench_{design}.csv fcavail-vs-gnn@3 {key} (6 s.f.)", "actual": float(val),
                                      "expected": float(row[key]), "passed": bool(abs(val - row[key]) <= 5e-6 * max(1, abs(row[key])))})
        fb = DATA / "results/stage5/eval/final_supp/did_fold_block.json"
        if fb.exists():
            ref = json.loads(fb.read_text())["did"]
            for F in ("drn_lak", "gnn_geo"):
                got = res["contrasts"][f"primary_10v3_{F[:3]}"]["primary_folds_7"]
                for key in ("lo", "hi"):
                    res["checks"].append({"what": f"final_supp/did_fold_block.json {F} fold_block_{key}", "actual": got[key],
                                          "expected": ref[F][f"fold_block_{key}"], "passed": bool(abs(got[key] - ref[F][f"fold_block_{key}"]) <= 1e-12)})
    res["all_checks_passed"] = all(c["passed"] for c in res["checks"])
    res["seconds"] = round(time.time() - t0, 1)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"did_{ds}.json").write_text(json.dumps(res, indent=1))
    print(ds, "checks passed:", res["all_checks_passed"], f"{res['seconds']}s", flush=True)
    if not res["all_checks_passed"]:
        print(json.dumps([c for c in res["checks"] if not c["passed"]], indent=1))
        raise SystemExit(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=["euppbench", "german"])
    ap.add_argument("--reps", type=int, default=si.D5["primary"]["reps"])
    a = ap.parse_args()
    for ds in a.datasets:
        run(ds, a.reps)


if __name__ == "__main__":
    main()
