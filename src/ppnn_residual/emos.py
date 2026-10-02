"""Gaussian EMOS (Gneiting et al. 2005) fitted by minimum CRPS estimation.

mu = a + b * ens_mean,   sigma^2 = c^2 + d^2 * ens_var
(the squared parameterization keeps sigma^2 > 0; the reference R code of Rasp & Lerch
uses c + d * ens_var with an ad-hoc penalty when negative; the two coincide whenever the
optimum has c, d >= 0).

Local EMOS: one parameter set per station, each fitted separately with L-BFGS-B and the
analytic CRPS gradient. Global EMOS: a single parameter set for all stations.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import optimize, stats

_ISQPI = 1 / math.sqrt(math.pi)


def _obj_grad(q, m, v, y):
    a, b, c, d = q
    mu = a + b * m
    s = np.sqrt(c * c + d * d * v)
    z = (y - mu) / s
    Phi, phi = stats.norm.cdf(z), stats.norm.pdf(z)
    crps = s * (z * (2 * Phi - 1) + 2 * phi - _ISQPI)
    dmu = -(2 * Phi - 1)
    ds = 2 * phi - _ISQPI
    n = len(y)
    g = np.array([dmu.sum(), (dmu * m).sum(), (ds * c / s).sum(), (ds * d * v / s).sum()]) / n
    return crps.mean(), g


def _fit_one(m, v, y, x0=(0.0, 1.0, 1.0, 1.0)):
    best = None
    for start in (x0, (0.0, 1.0, 0.5, 0.5)):
        r = optimize.minimize(_obj_grad, np.array(start, float), args=(m, v, y), jac=True, method="L-BFGS-B",
                              options={"maxiter": 5000, "gtol": 1e-9, "ftol": 1e-14})
        if best is None or r.fun < best.fun:
            best = r
    return best


def fit_emos(ens_mean, ens_var, y, groups=None, min_rows=10):
    """Returns (params [G,4], group_keys, info). groups=None -> global EMOS.
    Groups with fewer than `min_rows` rows are skipped (as in the Rasp & Lerch R code) and are
    therefore treated as unseen at prediction time."""
    m = np.asarray(ens_mean, np.float64)
    v = np.asarray(ens_var, np.float64)
    y = np.asarray(y, np.float64)
    if groups is None:
        keys, inv = np.array([0]), np.zeros(len(m), int)
    else:
        keys, inv = np.unique(np.asarray(groups), return_inverse=True)
    order = np.argsort(inv, kind="stable")
    bounds = np.searchsorted(inv[order], np.arange(len(keys) + 1))
    P = np.full((len(keys), 4), np.nan)
    grads, tot, fails, n_used = [], 0.0, 0, 0
    for gi in range(len(keys)):
        idx = order[bounds[gi]:bounds[gi + 1]]
        if len(idx) < min_rows:
            continue
        n_used += len(idx)
        r = _fit_one(m[idx], v[idx], y[idx])
        P[gi] = r.x
        grads.append(np.abs(r.jac).max())
        tot += r.fun * len(idx)
        fails += int(not r.success)
    ok = ~np.isnan(P[:, 0])
    info = {"groups": int(ok.sum()), "skipped_groups_lt_min_rows": [int(k) for k in keys[~ok]],
            "max_abs_grad": float(max(grads)), "n_scipy_success_false": fails,
            "train_crps_mean_over_rows": float(tot / n_used)}
    return P[ok], keys[ok], info


def predict_emos(params, keys, ens_mean, ens_var, groups=None, fallback=None):
    """Predict (mu, sigma, unseen_mask). Rows whose group is not in keys use `fallback`
    params [4] (e.g. global EMOS)."""
    ens_mean = np.asarray(ens_mean, np.float64)
    ens_var = np.asarray(ens_var, np.float64)
    n = len(ens_mean)
    if groups is None:
        P = np.repeat(params[:1], n, axis=0)
        unseen = np.zeros(n, bool)
    else:
        lookup = {k: i for i, k in enumerate(keys)}
        idx = np.array([lookup.get(k, -1) for k in np.asarray(groups)])
        unseen = idx < 0
        if unseen.any() and fallback is None:
            raise ValueError("unseen groups but no fallback given")
        P = np.where(unseen[:, None], np.asarray(fallback)[None, :], params[np.clip(idx, 0, None)])
    mu = P[:, 0] + P[:, 1] * ens_mean
    sigma = np.sqrt(P[:, 2] ** 2 + P[:, 3] ** 2 * ens_var)
    return mu, sigma, unseen
