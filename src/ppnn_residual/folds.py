"""Station-held-out fold assignment (random and spatially blocked)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def station_table(df: pd.DataFrame) -> pd.DataFrame:
    st = df.groupby("station")[["lat", "lon", "alt", "orog"]].first().sort_index()
    return st


def random_folds(stations: np.ndarray, K: int, seed: int) -> dict:
    perm = np.random.default_rng(seed).permutation(np.sort(stations))
    return {int(s): k for k, part in enumerate(np.array_split(perm, K)) for s in part}


def spatial_folds(st: pd.DataFrame, K: int, seed: int, n_init: int = 20) -> dict:
    """k-means (k-means++ init, Lloyd) on (lat, lon*cos(mean lat)); best of n_init restarts."""
    lat0 = np.radians(st["lat"].mean())
    P = np.c_[st["lat"].to_numpy(), st["lon"].to_numpy() * np.cos(lat0)]
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_init):
        C = [P[rng.integers(len(P))]]
        for _k in range(1, K):
            d2 = ((P[:, None, :] - np.array(C)[None]) ** 2).sum(-1).min(1)
            C.append(P[rng.choice(len(P), p=d2 / d2.sum())])
        C = np.array(C)
        for _it in range(300):
            lab = ((P[:, None, :] - C[None]) ** 2).sum(-1).argmin(1)
            Cn = np.array([P[lab == k].mean(0) if (lab == k).any() else C[k] for k in range(K)])
            if np.allclose(Cn, C):
                break
            C = Cn
        inertia = ((P - C[lab]) ** 2).sum()
        if best is None or inertia < best[0]:
            best = (inertia, lab)
    return {int(s): int(k) for s, k in zip(st.index, best[1])}
