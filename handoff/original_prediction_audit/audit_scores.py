"""Verify restored original predictions without changing the research checkout."""
import ast
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
import numpy as np
import pandas as pd
from scipy import stats

BUNDLE = Path(__file__).resolve().parent
REPO = Path(os.environ.get("PPNN_REPO_ROOT", BUNDLE.parents[1])).resolve()
HERE = Path(os.environ.get("PPNN_AUDIT_DIR", REPO / "local_runs/original_prediction_audit")).resolve()
ORIGINAL = Path(os.environ.get("PPNN_ORIGINAL_PREDS", HERE / "originals")).resolve()
OUT = HERE / "score_audit"
SOURCE = BUNDLE / "source"
BASE = ["raw_ensemble_gauss", "emos_gl", "emos_bst_gl"]
OLD = ["nn_unk", "nn_knn", "nn_noemb", "nn_attr", "nn_hybrid", "nn_hybrid_knn", "nn_hybrid_res"]
NEW = ["drn_lak", "gnn_geo", "samos_mlp", "samos_attn"]
STAGE = {"german": "stage2", "euppbench": "stage3"}
COMMIT = "8858ddbcffb146d82030a2ba126502eb0c4784be"


def frozen_functions():
    SOURCE.mkdir(exist_ok=True)
    files = ["src/ppnn_residual/metrics.py", "scripts/stage5_inference.py", "protocols/stage5_supplementary.json"]
    for name in files:
        dest = SOURCE / Path(name).name
        if not dest.exists():
            data = subprocess.check_output(["git", "show", f"{COMMIT}:{name}"], cwd=REPO)
            dest.write_bytes(data)
    tree = ast.parse((SOURCE / "metrics.py").read_text(encoding="utf-8"))
    wanted = {"crps_gaussian_np", "pit", "summary_metrics", "newey_west_var", "diebold_mariano", "block_bootstrap_ci", "benjamini_hochberg"}
    module = ast.Module(body=[x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name in wanted], type_ignores=[])
    ns = {"np": np, "stats": stats, "math": math}
    exec(compile(module, str(SOURCE / "metrics.py"), "exec"), ns)
    return ns


FN = frozen_functions()
crps = FN["crps_gaussian_np"]


def load_prediction(ds, ft, method, seed, rf, extra_dirs=()):
    n = len(rf)
    root = ORIGINAL / "results" / STAGE[ds] if method in BASE + OLD else ORIGINAL / "results/stage4" / ds
    dirs = [root / "preds" / ft, *extra_dirs]
    mu, sg = np.empty(n), np.empty(n)
    for k in range(int(rf.max()) + 1):
        suffix = "" if seed is None else f"_s{seed}"
        name = f"{method}_f{k}{suffix}.npz"
        path = next((d / name for d in dirs if (d / name).exists()), None)
        if path is None:
            raise FileNotFoundError(name)
        with np.load(path, allow_pickle=False) as z:
            a, b = z["mu"].astype(float), z["sigma"].astype(float)
        mask = rf == k
        if len(a) == n:
            a, b = a[mask], b[mask]
        assert a.shape == b.shape == (int(mask.sum()),), str(path)
        assert np.isfinite(a).all() and np.isfinite(b).all() and (b > 0).all(), str(path)
        mu[mask], sg[mask] = a, b
    return mu, sg


def check(value, expected, label, atol=1e-11, rtol=0):
    ok = bool(np.isclose(value, expected, atol=atol, rtol=rtol, equal_nan=True))
    checks.append({"label": label, "value": float(value), "expected": float(expected),
                   "abs_difference": float(abs(value - expected)), "passed": ok})
    return ok


def seen_score(ds, method, rf, y):
    root = ORIGINAL / "results" / STAGE[ds] / "preds/random"
    values = []
    for k in range(int(rf.max()) + 1):
        mask = rf != k
        if method in BASE:
            with np.load(root / f"{method}_f{k}.npz") as z:
                mu, sg = z["mu"][mask].astype(float), z["sigma"][mask].astype(float)
        elif ds == "euppbench":
            with np.load(root / f"seen_{method}_f{k}.npz") as z:
                assert set(z["seeds"].tolist()) == set(range(10))
                mu, sg = z["sum_mu"] / 10, z["sum_sigma"] / 10
        else:
            mu, sg = np.zeros(int(mask.sum())), np.zeros(int(mask.sum()))
            for s in range(10):
                with np.load(root / f"{method}_f{k}_s{s}.npz") as z:
                    mu += z["mu"][mask].astype(float)
                    sg += z["sigma"][mask].astype(float)
            mu /= 10
            sg /= 10
        assert mu.shape == sg.shape == y[mask].shape
        values.append(crps(mu, sg, y[mask]).mean())
    return float(np.mean(values))


def check_original_tests(ds, ft, day, st, scores):
    reference = pd.read_csv(REPO / "results" / STAGE[ds] / f"significance_{ft}.csv")
    rows = []
    for _, record in reference.iterrows():
        a, b = record.a, record.b
        frame = pd.DataFrame({"station": st, "day": day, "a": scores[a], "b": scores[b]})
        d = frame.groupby("day")[["a", "b"]].mean().sort_index()
        dm = FN["diebold_mariano"](d.a.values, d.b.values, 5)
        bs = FN["block_bootstrap_ci"]((d.a-d.b).values, 7, 10000)
        stations = frame.groupby(["station", "day"])[["a", "b"]].mean().reset_index() if ds == "euppbench" else frame.sort_values(["station", "day"])
        ps, signs = [], []
        for _, g in stations.groupby("station"):
            g = g.sort_values("day")
            if len(g) < 30:
                continue
            test = FN["diebold_mariano"](g.a.values, g.b.values, 1)
            ps.append(test["p_value"])
            signs.append(np.sign(test["mean_diff"]))
        reject = FN["benjamini_hochberg"](np.array(ps))
        signs = np.array(signs)
        ca, cb = scores[a].mean(), scores[b].mean()
        values = {"crps_a": ca, "crps_b": cb, "diff_a_minus_b": ca-cb,
                  "rel_diff_pct": 100*(ca/cb-1), "dm_stat": dm["dm_stat"], "dm_p": dm["p_value"],
                  "boot_lo": bs["ci_low"], "boot_hi": bs["ci_high"], "n_stations": len(ps),
                  "frac_st_a_better_BH": float((reject & (signs < 0)).mean()),
                  "frac_st_a_worse_BH": float((reject & (signs > 0)).mean())}
        for col, value in values.items():
            check(value, record[col], f"original_tests/{ds}/{ft}/{a}-{b}/{col}", atol=1e-12, rtol=5.1e-6)
        rows.append({"dataset": ds, "folds": ft, "a": a, "b": b, "family": record.family,
                     **values, "expected_BH": record.bh_reject_q05})
    ids = [i for i, row in enumerate(rows) if row["family"] == "primary"]
    rejects = FN["benjamini_hochberg"](np.array([rows[i]["dm_p"] for i in ids]))
    for i, reject in zip(ids, rejects):
        check(int(reject), int(rows[i]["expected_BH"]), f"primary_BH/{ds}/{ft}/{rows[i]['a']}-{rows[i]['b']}", atol=0)
    return rows


checks = []


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assert (HERE / "extracted_manifest.json").exists(), "Restore must finish first"
    ref = json.loads((REPO / "provenance/derived_numbers.json").read_text(encoding="utf-8"))
    (OUT / "reference_derived_numbers.json").write_text(json.dumps(ref), encoding="utf-8")
    metadata = []
    daily_tests = []
    original_tests = []
    for ds, stage in STAGE.items():
        original_meta = ORIGINAL / "results" / stage / "test_meta.npz"
        with np.load(original_meta, allow_pickle=True) as z:
            meta = {k: z[k] for k in z.files}
        with np.load(REPO / "results" / stage / "test_meta.npz", allow_pickle=True) as z:
            assert set(meta) == set(z.files)
            for key, value in meta.items():
                eq = np.array_equal(value, z[key], equal_nan=True) if value.dtype.kind in "fc" else np.array_equal(value, z[key])
                assert eq, f"Metadata mismatch {ds}/{key}"
        metadata.append({"dataset": ds, "rows": len(meta["y"]), "all_arrays_match_checkout": True})
        st, y = meta["station"].astype(int), meta["y"].astype(float)
        day = meta.get("init", meta.get("date"))
        all_tests = []
        for ft in ("random", "spatial"):
            f = ORIGINAL / "results" / stage / f"folds_{ft}.csv"
            folds = pd.read_csv(f, index_col=0)
            assert folds.equals(pd.read_csv(REPO / "results" / stage / f"folds_{ft}.csv", index_col=0)), f"Fold mismatch {ds}/{ft}"
            rf = folds.fold.reindex(st).to_numpy()
            assert np.isfinite(rf).all()
            stored = ref["exact_scores"][f"{ds}:{ft}"]
            daily = {}
            pooled = {}
            primary_scores = {}
            for method in BASE + OLD + NEW:
                seeds = [None] if method in BASE else list(range(10 if ft == "random" and method in OLD else 3))
                sum_mu, sum_sg = np.zeros(len(y)), np.zeros(len(y))
                singles = []
                for idx, seed in enumerate(seeds, 1):
                    mu, sg = load_prediction(ds, ft, method, seed, rf)
                    sum_mu += mu
                    sum_sg += sg
                    if seed is not None:
                        singles.append(float(crps(mu, sg, y).mean()))
                    if idx not in (3, 10) and seed is not None:
                        continue
                    key = method + ("@10seeds" if idx == 10 else "")
                    mm, ss = sum_mu / idx, sum_sg / idx
                    metrics = FN["summary_metrics"](mm, ss, y)
                    actual = {"crps": metrics["crps"], "cov80": metrics["coverage_80"],
                              "spread_skill": float(np.sqrt((ss**2).mean()) / metrics["rmse"]),
                              "bias": metrics["bias_mean_minus_obs"]}
                    if singles:
                        actual.update(single_seed_mean=float(np.mean(singles)), single_seed_sd=float(np.std(singles, ddof=1)))
                    for col, value in actual.items():
                        check(value, stored[key][col], f"scores/{ds}/{ft}/{key}/{col}")
                    if singles:
                        expected = ref["single_seed_crps"][f"{ds}:{ft}"][key]
                        for s, value in enumerate(singles):
                            check(value, expected[str(s)], f"single/{ds}/{ft}/{key}/{s}")
                    row_scores = crps(mm, ss, y)
                    if method in BASE + OLD:
                        primary_scores[method] = row_scores
                    if idx <= 3:
                        daily[method] = pd.DataFrame({"day": day, "score": row_scores}).groupby("day").score.mean().sort_index()
                        pooled[method] = float(row_scores.mean())
                if ft == "random" and method in BASE + OLD and method != "raw_ensemble_gauss":
                    value = seen_score(ds, method, rf, y)
                    expected = ref["seen_vs_new_random_10seeds"][f"{ds}:{method}"]["seen"]
                    # These JSON values were copied from five-decimal summary CSVs.
                    check(value, expected, f"seen/{ds}/{method}", atol=5.1e-6)
                print(f"Scores checked: {ds}/{ft}/{method}", flush=True)
            pd.DataFrame(daily).to_csv(OUT / f"daily_crps_{ds}_{ft}.csv", float_format="%.17g")
            ref_tests = pd.read_csv(REPO / f"results/stage4/{ds}/significance_{ft}.csv")
            rows = []
            for _, record in ref_tests.iterrows():
                a, b = record.a, record.b
                dm = FN["diebold_mariano"](daily[a].values, daily[b].values, 5)
                bs = FN["block_bootstrap_ci"]((daily[a] - daily[b]).values, 7, 10000)
                values = {"crps_a": pooled[a], "crps_b": pooled[b], "diff_a_minus_b": pooled[a] - pooled[b],
                          "rel_diff_pct": 100 * (pooled[a] / pooled[b] - 1), "n_days": len(daily[a]),
                          "dm_stat": dm["dm_stat"], "dm_p": dm["p_value"], "boot_lo": bs["ci_low"], "boot_hi": bs["ci_high"]}
                for col, value in values.items():
                    check(value, record[col], f"daily/{ds}/{ft}/{a}-{b}/{col}", atol=1e-12, rtol=5.1e-6)
                rows.append({"dataset": ds, "folds": ft, "a": a, "b": b, "family": record.family,
                             **values, "expected_BH": record.bh_reject_q05})
            for family in ["baselines_descriptive"]:
                ids = [i for i, row in enumerate(rows) if row["family"] == family]
                rej = FN["benjamini_hochberg"](np.array([rows[i]["dm_p"] for i in ids]))
                for i, reject in zip(ids, rej):
                    check(int(reject), int(rows[i]["expected_BH"]), f"BH/{ds}/{ft}/{rows[i]['a']}-{rows[i]['b']}", atol=0)
            all_tests.extend(rows)
            original_tests.extend(check_original_tests(ds, ft, day, st, primary_scores))
            del daily, pooled, primary_scores
            gc.collect()
        ids = [i for i, row in enumerate(all_tests) if row["family"] == "pilot_primary"]
        rej = FN["benjamini_hochberg"](np.array([all_tests[i]["dm_p"] for i in ids]))
        for i, reject in zip(ids, rej):
            row = all_tests[i]
            check(int(reject), int(row["expected_BH"]), f"BH/{ds}/{row['folds']}/{row['a']}-{row['b']}", atol=0)
        daily_tests.extend(all_tests)
    pd.DataFrame(checks).to_csv(OUT / "checks.csv", index=False, float_format="%.17g")
    pd.DataFrame(daily_tests).to_csv(OUT / "recomputed_daily_tests.csv", index=False, float_format="%.17g")
    pd.DataFrame(original_tests).to_csv(OUT / "recomputed_stage23_tests.csv", index=False, float_format="%.17g")
    failures = [x for x in checks if not x["passed"]]
    result = {"source_commit": COMMIT, "metadata": metadata, "checks": len(checks), "failures": failures,
              "score_metric_checks": sum(x["label"].startswith("scores/") for x in checks),
              "single_seed_checks": sum(x["label"].startswith("single/") for x in checks),
              "seen_score_checks": sum(x["label"].startswith("seen/") for x in checks),
              "daily_comparisons": len(daily_tests),
              "stage23_comparisons": len(original_tests),
              "max_score_difference": max(x["abs_difference"] for x in checks if x["label"].startswith("scores/")),
              "scope": "Original main/calibration/seen scores and Stage-2/3/4 daily tests; separate audit outputs only.",
              "seen_score_tolerance": 5.1e-6,
              "seen_tolerance_reason": "Reference seen scores were copied from original summary CSVs formatted to five decimals.",
              "max_seen_difference": max(x["abs_difference"] for x in checks if x["label"].startswith("seen/"))}
    (OUT / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
