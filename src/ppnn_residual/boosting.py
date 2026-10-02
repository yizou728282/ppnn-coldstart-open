"""Component-wise gradient boosting for Gaussian heteroscedastic regression (EMOS-bst),
following the non-homogeneous boosting of Messner, Mayr & Zeileis (2017) as implemented in
R crch(method="boosting"): mu = X beta, log sigma = X gamma, standardized regressors,
maximum likelihood, at each iteration the location or scale coefficient giving the larger
log-likelihood gain is updated by step nu, stopping iteration chosen by AIC."""
from __future__ import annotations

import math

import numpy as np

_LOG2PI = math.log(2 * math.pi)


def _ll(y, mu, logsig):
    return float(-(0.5 * _LOG2PI + logsig + 0.5 * ((y - mu) * np.exp(-logsig)) ** 2).sum())


class BoostedEMOS:
    def __init__(self, nu=0.05, maxit=1000, stop="aic", standardize_y=True):
        self.nu, self.maxit, self.stop, self.standardize_y = nu, maxit, stop, standardize_y

    def fit(self, X, y):
        X = np.asarray(X, np.float64); y = np.asarray(y, np.float64)
        # Standardizing the response makes the algorithm invariant to the units of y. Without it the
        # location gradient (y-mu)/sigma^2 is tiny while sigma starts at sd(y) (~7 degC) and 1000
        # iterations are far from convergence (see NOTES_stage2.md). The AIC argmin is unaffected
        # (the log-likelihood only shifts by the constant n*log(sd_y)).
        self.ym, self.ys = (y.mean(), y.std()) if self.standardize_y else (0.0, 1.0)
        y = (y - self.ym) / self.ys
        self.xm = X.mean(0); sd = X.std(0); sd[sd == 0] = 1.0; self.xs = sd
        Z = np.hstack([np.ones((len(y), 1)), (X - self.xm) / self.xs])
        n, p = Z.shape
        zz = (Z ** 2).sum(0)
        beta = np.zeros(p); gamma = np.zeros(p)
        beta[0] = y.mean(); gamma[0] = math.log(y.std())
        mu = np.full(n, beta[0]); ls = np.full(n, gamma[0])
        path_b, path_g, lls = [beta.copy()], [gamma.copy()], [_ll(y, mu, ls)]
        for _ in range(self.maxit):
            inv = np.exp(-2 * ls)
            g_mu = (y - mu) * inv
            g_ls = (y - mu) ** 2 * inv - 1.0
            # constant (zero-variance) columns, e.g. a land-use class absent from the fitting stations,
            # are never selected (identical behaviour when no such column exists)
            cb = np.divide(Z.T @ g_mu, zz, out=np.zeros(p), where=zz > 0)
            cg = np.divide(Z.T @ g_ls, zz, out=np.zeros(p), where=zz > 0)
            # component with largest reduction of the residual sum of squares of the gradient
            jb = int(np.argmax(cb ** 2 * zz)); jg = int(np.argmax(cg ** 2 * zz))
            mu_new = mu + self.nu * cb[jb] * Z[:, jb]
            ls_new = ls + self.nu * cg[jg] * Z[:, jg]
            ll_b, ll_g = _ll(y, mu_new, ls), _ll(y, mu, ls_new)
            if ll_b >= ll_g:
                beta[jb] += self.nu * cb[jb]; mu = mu_new; lls.append(ll_b)
            else:
                gamma[jg] += self.nu * cg[jg]; ls = ls_new; lls.append(ll_g)
            path_b.append(beta.copy()); path_g.append(gamma.copy())
        lls = np.array(lls)
        df = np.array([(np.abs(b) > 0).sum() + (np.abs(g) > 0).sum() for b, g in zip(path_b, path_g)])
        aic = -2 * lls + 2 * df
        m = int(np.argmin(aic)) if self.stop == "aic" else len(lls) - 1
        self.mstop, self.beta, self.gamma = m, path_b[m], path_g[m]
        self.path_b, self.path_g, self.aic = path_b, path_g, aic  # full path (diagnostics)
        self.info = {"mstop": m, "maxit": self.maxit, "n_nonzero_loc": int((self.beta != 0).sum()),
                     "n_nonzero_scale": int((self.gamma != 0).sum()), "train_ll_per_obs": float(lls[m] / n)}
        return self

    def predict(self, X):
        Z = np.hstack([np.ones((len(X), 1)), (np.asarray(X, np.float64) - self.xm) / self.xs])
        return self.ym + self.ys * (Z @ self.beta), self.ys * np.exp(Z @ self.gamma)

    def predict_at(self, X, m):
        Z = np.hstack([np.ones((len(X), 1)), (np.asarray(X, np.float64) - self.xm) / self.xs])
        return self.ym + self.ys * (Z @ self.path_b[m]), self.ys * np.exp(Z @ self.path_g[m])
