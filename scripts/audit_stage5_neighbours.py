"""Reconstruct held-out test neighbour changes from committed availability metadata.

No forecasts are fitted or rescored. Complete test forecast coverage is checked
against the delivered preprocessing report, and the GNN counts against run logs.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ppnn_residual.stage2_models import haversine_km


def main():
    meta = np.load(ROOT / "results/stage3/test_meta.npz")
    folds = pd.read_csv(ROOT / "results/stage3/folds_spatial.csv").sort_values("station")
    stations = folds.station.to_numpy()
    pairs = np.stack([meta["init"], meta["lead"]], axis=1)
    times, ti = np.unique(pairs, axis=0, return_inverse=True)
    si = np.searchsorted(stations, meta["station"])
    assert np.array_equal(stations[si], meta["station"])
    assert np.isfinite(meta["y"]).all()
    observed = np.zeros((len(times), len(stations)), dtype=bool)
    observed[ti, si] = True
    assert observed.sum() == len(si)  # one row per slice and station
    info = json.loads((ROOT / "results/stage5/euppbench/fcavail_extra_info.json").read_text())
    assert info["n_test_rows_missing_forecast"] == 0
    assert info["n_test_forecast_rows_missing_obs"] == 3960
    # Full 730-day x 5-lead grid, with the 3960 missing observations restored
    # as forecast-only rows by the original preprocessing run.
    assert len(times) == 730 * 5 and len(stations) == 117
    assert (~observed).sum() == 3960
    dist = haversine_km(folds.lat.values, folds.lon.values, folds.lat.values, folds.lon.values)
    graph = (dist <= 50) & ~np.eye(len(stations), dtype=bool)
    gnn_changed = ((~observed).astype(np.int32) @ graph.T.astype(np.int32)) > 0
    gnn_count = int(gnn_changed[ti, si].sum())
    expected = 0.0
    for k in range(7):
        log = json.loads((ROOT / f"results/stage5/euppbench/m1a/logs/spatial/gnn_geo_fcavail_f{k}_s0.json").read_text())
        expected += log["n_heldout_test_rows"] * log["frac_heldout_test_cases_neighbour_set_changed"]
    assert abs(gnn_count - expected) < 1e-6
    per_fold = []
    for k in range(7):
        train = np.flatnonzero(folds.fold.to_numpy() != k)
        held = np.flatnonzero(folds.fold.to_numpy() == k)
        nearest = train[np.argsort(dist[np.ix_(held, train)], axis=1)[:, :8]]
        n_changed = n_scored = 0
        for st, neighbours in zip(held, nearest):
            scored = observed[:, st]
            n_scored += int(scored.sum())
            n_changed += int((scored & (~observed[:, neighbours]).any(axis=1)).sum())
        per_fold.append({"paper_fold": k + 1, "scored": n_scored, "changed": n_changed})
    total = sum(x["scored"] for x in per_fold)
    assert total == len(si)
    result = {"description": "Metadata reconstruction of forecast-only test-neighbour changes; no retraining",
              "scored_test_rows": total, "missing_observations_with_forecasts": 3960,
              "gnn_changed": gnn_count, "gnn_fraction": gnn_count / total,
              "gnn_matches_original_run_logs": True,
              "samos_attn_spatial_changed": sum(x["changed"] for x in per_fold),
              "samos_attn_spatial_fraction": sum(x["changed"] for x in per_fold) / total,
              "samos_attn_per_fold": per_fold, "preprocessing_info": info}
    (ROOT / "handoff/STAGE5_NEIGHBOUR_AUDIT.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
