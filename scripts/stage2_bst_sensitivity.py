"""Post-hoc (documented) VALIDATION-ONLY check of the boosting iteration cap for EMOS-bst.
Fit on 2007-2014, score on 2015 (never 2016). Local: all stations; global: fold-0 training stations
of the random folds (2015 records of the same training stations)."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from ppnn_residual.boosting import BoostedEMOS  # noqa: E402
from ppnn_residual.data import load_table, feature_columns  # noqa: E402
from ppnn_residual.metrics import crps_gaussian_np  # noqa: E402

CHECK = [250, 500, 1000, 2000, 3000, 5000]
MAXIT = 5000


def crps_at(b, X, y, m):
    mu, sg = b.predict_at(X, m)
    return crps_gaussian_np(mu, sg, y).mean()


class PathBoost(BoostedEMOS):
    def fit(self, X, y):
        super().fit(X, y)
        return self


def main():
    import ppnn_residual.boosting as bm
    df = load_table(str(ROOT / "data/data_RL18.feather"))
    cols = feature_columns(df)
    loc_cols = [c for c in cols if c not in ["lat", "lon", "alt", "orog"]]
    out = {"check_iterations": CHECK}
    # keep full path: monkeypatch to retain paths
    orig = bm.BoostedEMOS.fit

    def fit_keep(self, X, y):
        r = orig(self, X, y)
        return r
    tr, va = df[df.year <= 2014], df[df.year == 2015]
    # local
    res = {m: [] for m in CHECK}; aic_m = []; w = []
    for s, g in tr.groupby("station"):
        v = va[va.station == s]
        if len(g) < 10 or len(v) == 0:
            continue
        b = Recorder(nu=0.05, maxit=MAXIT).fit(g[loc_cols].to_numpy(), g.obs.to_numpy())
        for m in CHECK:
            res[m].append(crps_at(b, v[loc_cols].to_numpy(), v.obs.to_numpy(), m) * len(v))
        aic_m.append(b.mstop); w.append(len(v))
    out["local_val2015_crps"] = {m: float(np.sum(res[m]) / np.sum(w)) for m in CHECK}
    out["local_aic_mstop_median_with_maxit5000"] = float(np.median(aic_m))
    print(out, flush=True)
    # global, fold-0 training stations
    import pandas as pd
    folds = pd.read_csv(ROOT / "results/stage2/folds_random.csv", index_col=0)["fold"].to_dict()
    trg = tr[[folds[int(s)] != 0 for s in tr.station]]
    vag = va[[folds[int(s)] != 0 for s in va.station]]
    b = Recorder(nu=0.05, maxit=MAXIT).fit(trg[cols].to_numpy(), trg.obs.to_numpy())
    out["global_fold0_val2015_crps"] = {m: float(crps_at(b, vag[cols].to_numpy(), vag.obs.to_numpy(), m)) for m in CHECK}
    out["global_aic_mstop_with_maxit5000"] = int(b.mstop)
    json.dump(out, open(ROOT / "results/stage2/logs/bst_iteration_sensitivity_val2015_standardized_y.json", "w"), indent=1)
    print(out)


class Recorder(BoostedEMOS):
    """BoostedEMOS that keeps the full coefficient path."""
    def fit(self, X, y):
        import ppnn_residual.boosting as bm
        X = np.asarray(X, np.float64); y = np.asarray(y, np.float64)
        super().fit(X, y)
        return self


if __name__ == "__main__":
    main()
