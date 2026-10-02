"""Recalculate available Stage-5 CRPS values without overwriting final reports."""
import csv
import json
from pathlib import Path

import numpy as np
from scipy.special import ndtr

from stage5_common import PARTITIONS, folds_df

ROOT = Path(__file__).resolve().parents[1]
R5 = ROOT / "results/stage5"


def score(mu, sigma, y):
    mu, sigma = np.asarray(mu, dtype=float), np.asarray(sigma, dtype=float)
    assert np.isfinite(mu).all() and np.isfinite(sigma).all() and (sigma > 0).all()
    z = (y - mu) / sigma
    return float(np.mean(sigma * (z * (2 * ndtr(z) - 1) + 2 * np.exp(-z*z/2) / np.sqrt(2*np.pi) - 1/np.sqrt(np.pi))))


def summary(ds, design):
    with (R5 / f"eval/final/summary_{ds}_{design}.csv").open(newline="") as f:
        return {row["method"]: row for row in csv.DictReader(f)}


def predictions(ds, design, method, seed, stations, ftype="spatial"):
    actual_design = ftype if design == "m1a" else design
    fdf = folds_df(ds, actual_design)
    row_fold = fdf.fold.reindex(stations).to_numpy()
    assert not np.isnan(row_fold).any()
    directory = R5 / ds / design / "preds" / ftype
    mu = np.full(len(stations), np.nan)
    sigma = mu.copy()
    for fold in range(int(fdf.fold.max()) + 1):
        suffix = "" if seed is None else f"_s{seed}"
        with np.load(directory / f"{method}_f{fold}{suffix}.npz") as z:
            mask = row_fold == fold
            a, b = z["mu"], z["sigma"]
            if len(a) == len(stations):
                a, b = a[mask], b[mask]
            assert a.shape == b.shape == (int(mask.sum()),)
            mu[mask], sigma[mask] = a, b
    return mu, sigma


def main():
    checked = []
    for ds, stage in (("euppbench", "stage3"), ("german", "stage2")):
        with np.load(ROOT / f"results/{stage}/test_meta.npz", allow_pickle=True) as meta:
            stations, y = meta["station"], meta["y"].astype(float)
        for method, row in summary(ds, "spatial").items():
            if int(row["n_seeds"]) != 10:
                continue
            expected = json.loads(row["per_seed"])
            for seed in range(3, 10):
                mu, sigma = predictions(ds, "spatial", method, seed, stations)
                actual = score(mu, sigma, y)
                difference = abs(actual - expected[str(seed)])
                assert difference <= 5.1e-6, (ds, method, seed, actual, expected[str(seed)])
                checked.append({"dataset": ds, "design": "spatial", "method": method, "seed": seed,
                                "crps": actual, "absolute_difference_from_csv": difference})
        if ds != "euppbench":
            continue
        for part in PARTITIONS:
            design = f"part_{part}"
            for method, row in summary(ds, design).items():
                seeds = range(int(row["n_seeds"])) if int(row["n_seeds"]) else [None]
                arrays = [predictions(ds, design, method, seed, stations) for seed in seeds]
                mu = np.stack([a for a, _ in arrays]).mean(axis=0)
                sigma = np.stack([b for _, b in arrays]).mean(axis=0)
                actual = score(mu, sigma, y)
                difference = abs(actual - float(row["crps"]))
                assert difference <= 5.1e-6, (design, method, actual, row["crps"])
                checked.append({"dataset": ds, "design": design, "method": method,
                                "crps": actual, "absolute_difference_from_csv": difference})
            print(f"Verified {design}", flush=True)
        for ftype, methods in (("random", ["gnn_geo_fcavail"]),
                               ("spatial", ["gnn_geo_fcavail", "samos_attn_fcavail"])):
            expected = summary(ds, ftype)
            for method in methods:
                arrays = [predictions(ds, "m1a", method, s, stations, ftype) for s in range(3)]
                actual = score(np.stack([a for a, _ in arrays]).mean(axis=0),
                               np.stack([b for _, b in arrays]).mean(axis=0), y)
                difference = abs(actual - float(expected[method]["crps"]))
                assert difference <= 5.1e-6, (ftype, method, actual)
                checked.append({"dataset": ds, "design": "m1a_" + ftype, "method": method,
                                "crps": actual, "absolute_difference_from_csv": difference})
    report = {"checked_crps_values": len(checked), "max_absolute_difference_from_csv": max(
        x["absolute_difference_from_csv"] for x in checked), "csv_rounding_tolerance": 5.1e-6,
        "scope": "Stage-5 extra seeds 3-9, all six additional partitions, and m1a. Original Stage-2/3/4 predictions are not included in the release; their full inference was not rerun.",
        "values": checked}
    (ROOT / "handoff/STAGE5_PREDICTION_AUDIT.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "values"}, indent=2))


if __name__ == "__main__":
    main()
