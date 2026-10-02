"""Scores and statistical tests. All CRPS values use the closed-form Gaussian CRPS."""
from __future__ import annotations

import math

import numpy as np
from scipy import stats

try:
    import torch
except ImportError:  # metrics are usable without torch
    torch = None


def crps_gaussian_np(mu, sigma, y):
    """Closed-form CRPS of N(mu, sigma^2) (Gneiting et al. 2005)."""
    mu, sigma, y = (np.asarray(a, dtype=np.float64) for a in (mu, sigma, y))
    z = (y - mu) / sigma
    return sigma * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / math.sqrt(math.pi))


def crps_gaussian_torch(mu, sigma, y):
    z = (y - mu) / sigma
    pdf = torch.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
    cdf = 0.5 * (1 + torch.erf(z / math.sqrt(2)))
    return sigma * (z * (2 * cdf - 1) + 2 * pdf - 1 / math.sqrt(math.pi))


def pit(mu, sigma, y):
    return stats.norm.cdf((np.asarray(y, float) - mu) / sigma)


def summary_metrics(mu, sigma, y, pit_bins=10) -> dict:
    mu, sigma, y = (np.asarray(a, dtype=np.float64) for a in (mu, sigma, y))
    crps = crps_gaussian_np(mu, sigma, y)
    p = pit(mu, sigma, y)
    out = {
        "n": int(len(y)),
        "crps": float(crps.mean()),
        "mae": float(np.abs(y - mu).mean()),
        "rmse": float(np.sqrt(((y - mu) ** 2).mean())),
        "mean_sigma": float(sigma.mean()),
        "bias_mean_minus_obs": float((mu - y).mean()),
    }
    for lvl in (0.8, 0.9):
        lo, hi = (1 - lvl) / 2, 1 - (1 - lvl) / 2
        out[f"coverage_{int(lvl*100)}"] = float(((p >= lo) & (p <= hi)).mean())
        out[f"width_{int(lvl*100)}"] = float((2 * stats.norm.ppf(hi) * sigma).mean())
    hist, _ = np.histogram(p, bins=pit_bins, range=(0, 1))
    freq = hist / hist.sum()
    out["pit_hist"] = freq.tolist()
    # reliability index (Delle Monache et al. 2006): sum |f_i - 1/k|
    out["pit_reliability_index"] = float(np.abs(freq - 1 / pit_bins).sum())
    out["pit_mean"] = float(p.mean())
    out["pit_var"] = float(p.var())  # 1/12 = 0.0833 for calibrated
    return out


def newey_west_var(d: np.ndarray, lag: int) -> float:
    """HAC (Bartlett kernel) long-run variance of the mean of d."""
    d = np.asarray(d, float) - np.mean(d)
    n = len(d)
    v = d @ d / n
    for k in range(1, lag + 1):
        w = 1 - k / (lag + 1)
        v += 2 * w * (d[k:] @ d[:-k]) / n
    return v / n


def diebold_mariano(score_a: np.ndarray, score_b: np.ndarray, lag: int) -> dict:
    """DM test of H0: E[a-b]=0 on a (time-ordered) series of score differences.
    Negative statistic -> model a has lower (better) score."""
    d = np.asarray(score_a, float) - np.asarray(score_b, float)
    var = newey_west_var(d, lag)
    stat = d.mean() / math.sqrt(var) if var > 0 else float("nan")
    p = 2 * (1 - stats.norm.cdf(abs(stat)))
    return {"mean_diff": float(d.mean()), "dm_stat": float(stat), "p_value": float(p), "n": int(len(d)), "hac_lag": lag}


def block_bootstrap_ci(d: np.ndarray, block: int, reps: int, seed: int = 12345, alpha=0.05) -> dict:
    """Moving-block bootstrap CI for the mean of a time series d."""
    d = np.asarray(d, float)
    n = len(d)
    rng = np.random.default_rng(seed)
    nb = int(math.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(reps, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(reps, -1)[:, :n]
    means = d[idx].mean(1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {"ci_low": float(lo), "ci_high": float(hi), "frac_boot_mean_below_0": float((means < 0).mean()),
            "block": block, "reps": reps}


def benjamini_hochberg(pvals: np.ndarray, q: float = 0.05) -> np.ndarray:
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    thresh = q * np.arange(1, n + 1) / n
    passed = p[order] <= thresh
    k = np.max(np.nonzero(passed)[0]) + 1 if passed.any() else 0
    rej = np.zeros(n, bool)
    rej[order[:k]] = True
    return rej
