"""Stage 4: dense (time slice x station) cubes for the German RL18 and EUPPBench new-station benchmarks.

A time slice is one date (German, lead 48 h) or one (initialisation, lead) pair (EUPPBench).
Missing (slice, station) entries are NaN. Test rows can be mapped back to the Stage-2/3 `test_meta.npz` order.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Cube:
    name: str
    stations: np.ndarray          # [N] station ids (sorted)
    st: pd.DataFrame              # station table indexed by id: lat, lon, alt, orog (+ lu_group)
    met_names: list
    X: np.ndarray                 # [T, N, P] float32 ensemble statistics (NaN = missing)
    y: np.ndarray                 # [T, N] float32 observations (NaN = missing)
    split: np.ndarray             # [T] 0 fit, 1 early-stopping, 2 test
    doy: np.ndarray               # [T] day of year of the valid date (1..366)
    month: np.ndarray             # [T] month of the valid date
    lead: np.ndarray              # [T] lead time (h)
    leads: list
    test_t: np.ndarray            # [R] slice index of every test row in test_meta order
    test_n: np.ndarray            # [R] station index of every test row
    test_station: np.ndarray      # [R] station id (for checks)
    i_mean: int                   # column of t2m ensemble mean
    i_sd: int                     # column of t2m ensemble sd (German: variance -> converted to sd in `ens_sd`)
    sd_is_var: bool

    def ens_sd(self):
        s = self.X[:, :, self.i_sd]
        return np.sqrt(np.clip(s, 0, None)) if self.sd_is_var else s


def _fill(T, N, t, n, vals, P=None):
    shape = (T, N) if P is None else (T, N, P)
    out = np.full(shape, np.nan, np.float32)
    out[t, n] = vals
    return out


def german_cube() -> Cube:
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from ppnn_residual.data import load_table
    df = load_table(str(ROOT / "data/data_RL18.feather"), ["sm_mean", "sm_var"])
    met = [c for c in df.columns if c not in {"date", "station", "obs", "year", "lat", "lon", "alt", "orog"}]
    st = df.groupby("station")[["lat", "lon", "alt", "orog"]].first().sort_index()
    stations = st.index.to_numpy()
    dates = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    ud = np.sort(dates.unique())
    t = np.searchsorted(ud, dates.to_numpy()); n = np.searchsorted(stations, df.station.to_numpy())
    T, N = len(ud), len(stations)
    X = _fill(T, N, t, n, df[met].to_numpy(np.float32), len(met))
    y = _fill(T, N, t, n, df.obs.to_numpy(np.float32))
    yr = pd.DatetimeIndex(ud).year.to_numpy()
    split = np.where(yr <= 2014, 0, np.where(yr == 2015, 1, 2))
    te = (df.year == 2016).to_numpy()  # stage-2 test_meta = df[year == 2016] in this order
    ts = df.station.to_numpy()[te]
    di = pd.DatetimeIndex(ud)
    return Cube("german", stations, st, met, X, y, split, di.dayofyear.to_numpy(), di.month.to_numpy(),
                np.full(T, 48), [48], t[te], n[te], ts, met.index("t2m_mean"), met.index("t2m_var"), True)


def eupp_cube() -> Cube:
    z = np.load(ROOT / "data/stage3/stage3_euppbench.npz", allow_pickle=False)
    met = [str(f) for f in z["features"]]
    st = pd.read_json(io.StringIO(str(z["st_table"]))).set_index("station")
    st = st[st.included].sort_index()
    stations = st.index.to_numpy()
    parts = []
    for p in ("train", "test"):
        keep = np.isin(z[f"{p}_station"], stations)
        init = z[f"{p}_init"][keep].astype("datetime64[D]").astype(np.int64)
        parts.append(dict(p=p, keep=keep, init=init, lead=z[f"{p}_lead"][keep].astype(np.int64),
                          valid=z[f"{p}_valid"][keep].astype("datetime64[D]"), st=z[f"{p}_station"][keep]))
    keys, tidx = [], []
    for j, q in enumerate(parts):
        k = (j * 10 ** 7 + q["init"]) * 1000 + q["lead"]
        keys.append(k)
    uk = np.unique(np.concatenate(keys))
    T, N = len(uk), len(stations)
    X = np.full((T, N, len(met)), np.nan, np.float32); y = np.full((T, N), np.nan, np.float32)
    split = np.full(T, -1); doy = np.zeros(T, int); month = np.zeros(T, int); lead = np.zeros(T, int)
    for j, q in enumerate(parts):
        t = np.searchsorted(uk, keys[j]); n = np.searchsorted(stations, q["st"])
        X[t, n] = z[f"{q['p']}_X"][q["keep"]]; y[t, n] = z[f"{q['p']}_y"][q["keep"]]
        v = pd.DatetimeIndex(q["valid"])
        doy[t] = v.dayofyear; month[t] = v.month; lead[t] = q["lead"]
        if q["p"] == "train":
            split[t] = np.where(v.year <= 2014, 0, 1)
        else:
            split[t] = 2; test_t, test_n, test_st = t, n, q["st"]
    return Cube("euppbench", stations, st[["lat", "lon", "alt", "orog", "lu_group"]], met, X, y, split, doy, month, lead,
                [24, 48, 72, 96, 120], test_t, test_n, test_st, met.index("t2m_mean"), met.index("t2m_sd"), False)


def load_cube(name):
    return german_cube() if name == "german" else eupp_cube()


def results_dir(name):
    return ROOT / ("results/stage2" if name == "german" else "results/stage3")
