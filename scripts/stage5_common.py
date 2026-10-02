"""Stage 5 shared helpers (protocols/stage5_supplementary.json): paths, partitions, fold tables."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
P5 = json.loads((ROOT / "protocols/stage5_supplementary.json").read_text())
R5 = Path(os.environ.get("STAGE5_ROOT", ROOT / "results/stage5"))
SMOKE = os.environ.get("STAGE5_SMOKE") == "1"  # 1 epoch, fold 0 only; outputs must go to a scratch STAGE5_ROOT
OLD_DIR = {"german": ROOT / "results/stage2", "euppbench": ROOT / "results/stage3"}
PARTITIONS = list(P5["B_partitions_euppbench"]["partitions"])


def savez_atomic(path, **arrs):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp.npz"); np.savez(tmp, **arrs); os.replace(tmp, path)


def json_atomic(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp"); tmp.write_text(json.dumps(obj, indent=1, default=float)); os.replace(tmp, path)


def design_dir(ds, design):
    """Output root of a Stage-5 design: 'random' / 'spatial' (primary folds), 'part_<pid>', 'm1a'."""
    return R5 / ds / design


def folds_df(ds, design):
    """Station -> fold table (index = station id, column 'fold') for a design."""
    if design in ("random", "spatial"):
        return pd.read_csv(OLD_DIR[ds] / f"folds_{design}.csv", index_col=0)
    if design.startswith("part_"):
        return pd.read_csv(R5 / ds / "partitions" / f"{design[5:]}.csv", index_col=0)
    raise ValueError(design)


def make_partitions():
    """Compute the EUPPBench partitions of protocol section B from station coordinates only (idempotent)."""
    from ppnn_residual.folds import spatial_folds
    st = pd.read_csv(OLD_DIR["euppbench"] / "folds_spatial.csv", index_col=0)  # same station order as the primary k-means
    out = R5 / "euppbench" / "partitions"; out.mkdir(parents=True, exist_ok=True)
    spec = P5["B_partitions_euppbench"]["partitions"]
    for pid, s in spec.items():
        if pid == "loco":
            countries = sorted(st.country.unique())
            lab = st.country.map({c: i for i, c in enumerate(countries)}).to_numpy()
        else:
            K = int(pid.split("_")[0][2:]); seed = int(pid.split("_s")[1])
            n_init = 1 if pid.startswith("km7") else 20
            f = spatial_folds(st[["lat", "lon"]], K, seed, n_init=n_init); lab = np.array([f[int(i)] for i in st.index])
            assert list(np.bincount(lab)) == s["sizes"], (pid, np.bincount(lab), s["sizes"])
        t = st.drop(columns=["fold"]).copy(); t["fold"] = lab
        t.to_csv(out / f"{pid}.csv")
    return out


def folds_dict(fdf):
    return {int(s): int(k) for s, k in fdf.fold.items()}


if __name__ == "__main__":
    print(make_partitions())
