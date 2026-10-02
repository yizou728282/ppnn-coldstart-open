"""Loading and year-isolated splitting of the Rasp & Lerch (2018) PPNN dataset."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

UNK = 0  # reserved embedding index for stations not seen in the fitting data


def load_table(path: str, drop_columns=("sm_mean", "sm_var")) -> pd.DataFrame:
    df = pd.read_feather(path)
    df = df.drop(columns=[c for c in drop_columns if c in df.columns])
    df["station"] = df["station"].astype(np.int64)
    df["year"] = pd.to_datetime(df["date"]).dt.year
    # the published file contains no missing obs / t2m, but be explicit
    df = df.dropna(subset=["obs", "t2m_mean", "t2m_var"]).reset_index(drop=True)
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    skip = {"date", "station", "obs", "year"}
    return [c for c in df.columns if c not in skip]


@dataclass
class Split:
    x: np.ndarray        # standardized features, float32 [n, p]
    ens_mean: np.ndarray # raw ensemble mean t2m (degC), float32
    ens_sd: np.ndarray   # raw ensemble standard deviation t2m, float32
    station: np.ndarray  # raw station id, int64
    emb_idx: np.ndarray  # embedding index (UNK for unseen stations)
    y: np.ndarray        # observation, float32
    date: np.ndarray     # datetime64


class Preprocessor:
    """Standardization + station->embedding index map, fitted on fitting years only."""

    def fit(self, df: pd.DataFrame, cols: list[str], min_station_rows: int = 10):
        self.cols = cols
        x = df[cols].to_numpy(np.float64)
        self.mean = x.mean(0)
        sd = x.std(0)
        sd[sd == 0] = 1.0
        self.sd = sd
        counts = df["station"].value_counts()
        stations = np.sort(counts.index[counts >= min_station_rows].to_numpy())  # rarer -> UNK
        self.station_to_idx = {int(s): i + 1 for i, s in enumerate(stations)}  # 0 = UNK
        self.n_emb = len(stations) + 1
        return self

    def transform(self, df: pd.DataFrame) -> Split:
        x = ((df[self.cols].to_numpy(np.float64) - self.mean) / self.sd).astype(np.float32)
        st = df["station"].to_numpy(np.int64)
        emb = np.array([self.station_to_idx.get(int(s), UNK) for s in st], dtype=np.int64)
        return Split(
            x=x,
            ens_mean=df["t2m_mean"].to_numpy(np.float32),
            ens_sd=np.sqrt(np.clip(df["t2m_var"].to_numpy(np.float64), 0, None)).astype(np.float32),
            station=st,
            emb_idx=emb,
            y=df["obs"].to_numpy(np.float32),
            date=pd.to_datetime(df["date"]).dt.tz_localize(None).to_numpy(),
        )


def year_split(df: pd.DataFrame, years) -> pd.DataFrame:
    return df[df["year"].isin(list(years))].reset_index(drop=True)
