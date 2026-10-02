"""Stage 5 / m1a: forecast features of EUPPBench (slice, station) pairs whose OBSERVATION is missing (included stations
only). These rows were dropped by scripts/stage3_preprocess.py; they are needed to define neighbour availability by
forecast availability. Forecast inputs only - no observations are stored.
    python scripts/stage5_preprocess_fcavail.py --zarr /workspace/euppbench/zarr
Output: results/stage5/euppbench/fcavail_extra.npz (+ _info.json)
"""
import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from stage3_preprocess import FEATURES, build  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--zarr", required=True); a = ap.parse_args()
    out = ROOT / "results/stage5/euppbench/fcavail_extra.npz"
    if out.exists():
        print("exists", out); return
    z = np.load(ROOT / "data/stage3/stage3_euppbench.npz", allow_pickle=False)
    st = pd.read_json(io.StringIO(str(z["st_table"]))).set_index("station")
    inc = st.index[st.included].to_numpy()
    tr, te, _ = build(Path(a.zarr))
    tr = {k: v[tr["valid"] <= np.datetime64("2016-12-31")] for k, v in tr.items()}  # as stage3_preprocess
    it = FEATURES.index("t2m_mean"), FEATURES.index("t2m_sd")
    parts, info = [], {}
    for j, (name, d) in enumerate((("train", tr), ("test", te))):
        fc = np.isfinite(d["X"][:, it[0]]) & np.isfinite(d["X"][:, it[1]])
        m = fc & ~np.isfinite(d["y"]) & np.isin(d["station"], inc)
        info[f"n_{name}_forecast_rows_missing_obs"] = int(m.sum())
        info[f"n_{name}_rows_missing_forecast"] = int((~fc & np.isin(d["station"], inc)).sum())
        parts.append(dict(part=np.full(m.sum(), j, np.int8), station=d["station"][m].astype(np.int64),
                          init=d["init"][m].astype("datetime64[D]").astype(np.int64), lead=d["lead"][m].astype(np.int64),
                          X=d["X"][m].astype(np.float32)))
    cat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **cat)
    out.with_name("fcavail_extra_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
