"""Recompute DRN/GNN coupled station/seed DiD from restored predictions.

Uses registered 2000 replicates, the original RNG order and cached duplicate seed
compositions. Writes only into this audit directory. Does not train any model.
"""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
import gc
import json
import time
import zlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import ndtr

from audit_scores import COMMIT, HERE, ORIGINAL, REPO, SOURCE, STAGE, BASE, OLD, NEW

R5 = REPO / "results/stage5"
OUT = HERE / "bootstrap_audit"
STAT = ["emos_nn1", "emos_idw", "emos_reg", "emos_regidw", "samos_lin"]
ALL = BASE + STAT + OLD + NEW
CORE = ["emos_bst_gl", "drn_lak", "gnn_geo"]
PROTO = json.loads((SOURCE / "stage5_supplementary.json").read_text())
REPS = PROTO["D_inference"]["primary"]["reps"]


def score(mu, sg, y):
    z = (y-mu)/sg
    return sg*(z*(2*ndtr(z)-1)+2*np.exp(-z*z/2)/np.sqrt(2*np.pi)-1/np.sqrt(np.pi))


def paths(ds, design, method):
    if design.startswith("part_"):
        return [R5 / ds / design / "preds/spatial"]
    original = ORIGINAL / "results" / STAGE[ds] if method in BASE+OLD else ORIGINAL / "results/stage4" / ds
    return [original / "preds" / design, R5 / ds / design / "preds" / design] if method not in STAT else [R5 / ds / design / "preds" / design]


def inventory(ds, design, method, K):
    dirs = paths(ds, design, method)
    def files(seed):
        suffix = "" if seed is None else f"_s{seed}"
        result = [next((d/f"{method}_f{k}{suffix}.npz" for d in dirs if (d/f"{method}_f{k}{suffix}.npz").exists()), None) for k in range(K)]
        return result if all(p is not None for p in result) else None
    if method in BASE+STAT:
        found = files(None)
        return {None: found} if found else {}
    found = {s: files(s) for s in range(10)}
    return {s: p for s, p in found.items() if p}


def load(files, rf):
    mu, sg = np.empty(len(rf)), np.empty(len(rf))
    for k, path in enumerate(files):
        with np.load(path, allow_pickle=False) as z:
            a, b = z["mu"].astype(np.float32), z["sigma"].astype(np.float32)
        mask = rf == k
        if len(a) == len(rf):
            a, b = a[mask], b[mask]
        assert a.shape == b.shape == (int(mask.sum()),)
        assert np.isfinite(a).all() and np.isfinite(b).all() and (b > 0).all()
        mu[mask], sg[mask] = a, b
    return mu, sg


def design_data(ds, design, meta):
    f = ORIGINAL / "results" / STAGE[ds] / f"folds_{design}.csv" if design in ("random", "spatial") else R5 / ds / "partitions" / (design[5:]+".csv")
    folds = pd.read_csv(f, index_col=0)
    rf = folds.fold.reindex(meta["station"]).to_numpy()
    assert np.isfinite(rf).all()
    station, si = np.unique(meta["station"], return_inverse=True)
    counts = np.bincount(si, minlength=len(station)).astype(float)
    W = np.random.default_rng(2026).multinomial(len(station), np.ones(len(station))/len(station), size=REPS).astype(float)
    denominator = W @ counts
    rng = np.random.default_rng(99+zlib.crc32(design.encode()) % 1000)
    result = {}
    expected = pd.read_csv(R5 / "eval/final" / f"summary_{ds}_{design}.csv").set_index("method")
    for method in ALL:
        found = inventory(ds, design, method, int(rf.max())+1)
        if not found:
            continue
        ns = 0 if list(found) == [None] else len(found)
        assert int(expected.loc[method, "n_seeds"]) == ns, (ds, design, method, ns)
        draws = rng.integers(0, ns, size=(REPS, ns)) if ns else None
        if method not in CORE:
            continue
        begin = time.monotonic()
        pairs = [load(files, rf) for files in found.values()]
        MU = np.stack([p[0] for p in pairs]); SG = np.stack([p[1] for p in pairs])
        del pairs
        y = meta["y"].astype(float)
        cr = score(MU.mean(axis=0), SG.mean(axis=0), y)
        point = float(cr.mean())
        assert abs(point-float(expected.loc[method, "crps"])) <= 5.1e-6
        if not ns:
            S = np.broadcast_to(np.bincount(si, cr, len(station)), (REPS, len(station)))
            unique_n = 1
        else:
            c = np.stack([np.bincount(d, minlength=ns) for d in draws])
            unique, inv = np.unique(c, axis=0, return_inverse=True)
            sums = np.empty((len(unique), len(station)))
            for i, cc in enumerate(unique):
                weights = cc/ns
                sums[i] = np.bincount(si, score(weights @ MU, weights @ SG, y), len(station))
                if (i+1) % 250 == 0:
                    print(f"Bootstrap {ds}/{design}/{method}: {i+1}/{len(unique)} seed compositions", flush=True)
            S = sums[inv]
            unique_n = len(unique)
        boot = (W*S).sum(axis=1)/denominator
        result[method] = {"point": point, "boot": boot, "seeds": ns}
        np.savez_compressed(OUT / f"{ds}_{design}_{method}.npz", station=station, station_counts=counts,
                            bootstrap_scores=boot, point=point, n_seeds=ns)
        print(f"Bootstrap done: {ds}/{design}/{method}, {unique_n} unique compositions, {time.monotonic()-begin:.1f}s", flush=True)
        del MU, SG, cr, S
        if ns:
            del sums
        gc.collect()
    assert set(result) == set(CORE)
    return result


def comparison(spatial, random, method):
    sp = spatial[method]["point"]-spatial["emos_bst_gl"]["point"]
    rd = random[method]["point"]-random["emos_bst_gl"]["point"]
    spb = spatial[method]["boot"]-spatial["emos_bst_gl"]["boot"]
    rdb = random[method]["boot"]-random["emos_bst_gl"]["boot"]
    boot = spb-rdb
    return {"did": sp-rd, "lo": float(np.quantile(boot,.025)), "hi": float(np.quantile(boot,.975)),
            "p_did_gt0": float((boot>0).mean()), "spatial_F_minus_E": sp, "random_F_minus_E": rd,
            "random_F_minus_E_lo": float(np.quantile(rdb,.025)), "random_F_minus_E_hi": float(np.quantile(rdb,.975)),
            "spatial_F_minus_E_lo": float(np.quantile(spb,.025)), "spatial_F_minus_E_hi": float(np.quantile(spb,.975))}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assert (HERE / "score_audit/summary.json").exists(), "Run the score audit first"
    score_result = json.loads((HERE / "score_audit/summary.json").read_text())
    assert not score_result["failures"], "Resolve original-score discrepancies before combined inference"
    expected = json.loads((R5 / "eval/final/decisions.json").read_text())
    report, checks = {}, []
    for ds, stage in STAGE.items():
        with np.load(ORIGINAL / "results" / stage / "test_meta.npz", allow_pickle=True) as z:
            meta = {k:z[k] for k in z.files}
        random = design_data(ds,"random",meta)
        designs = ["spatial"] + (["part_"+p for p in PROTO["B_partitions_euppbench"]["partitions"]] if ds=="euppbench" else [])
        report[ds] = {}
        for design in designs:
            spatial = design_data(ds,design,meta)
            report[ds][design] = {}
            for method in ["drn_lak","gnn_geo"]:
                got = comparison(spatial,random,method)
                want = expected[ds][method] if design=="spatial" else expected[ds][method]["partitions"][design]
                report[ds][design][method] = got
                for key, value in got.items():
                    if key in want:
                        delta = abs(value-want[key])
                        checks.append({"dataset":ds,"design":design,"method":method,"field":key,
                                       "actual":value,"expected":want[key],"abs_difference":delta,"passed":bool(delta<=1e-8)})
            (OUT / "progress.json").write_text(json.dumps(report,indent=2))
    failures = [x for x in checks if not x["passed"]]
    summary = {"source_commit":COMMIT,"replicates":REPS,"core_models":["drn_lak","gnn_geo"],
               "checks":len(checks),"max_abs_difference":max(x["abs_difference"] for x in checks),"failures":failures,
               "scope":"Primary DRN/GNN coupled seed/station DiD and all six additional EUPPBench partitions; not every auxiliary pair/ranking or fold bootstrap."}
    (OUT / "summary.json").write_text(json.dumps(summary,indent=2))
    (OUT / "checks.json").write_text(json.dumps(checks,indent=2))
    (OUT / "recomputed_did.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(summary,indent=2),flush=True)
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
