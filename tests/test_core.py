import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy import integrate, optimize, stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ppnn_residual.metrics import (crps_gaussian_np, crps_gaussian_torch, diebold_mariano,  # noqa: E402
                                   benjamini_hochberg, summary_metrics, block_bootstrap_ci)
from ppnn_residual.emos import fit_emos, predict_emos  # noqa: E402
from ppnn_residual.models import DistNet  # noqa: E402
from ppnn_residual.data import Preprocessor, UNK  # noqa: E402


def test_crps_closed_form_matches_numerical_integral():
    mu, sig, y = 1.3, 2.1, -0.4
    num = integrate.quad(lambda t: (stats.norm.cdf(t, mu, sig) - (t >= y)) ** 2, -40, 40, points=[y])[0]
    assert abs(crps_gaussian_np(mu, sig, y) - num) < 1e-6
    t = crps_gaussian_torch(torch.tensor(mu), torch.tensor(sig), torch.tensor(y)).item()
    assert abs(t - num) < 1e-5


def test_summary_metrics_calibrated():
    rng = np.random.default_rng(0)
    mu = rng.normal(size=200000); sig = np.exp(rng.normal(size=200000) * 0.3)
    y = mu + sig * rng.normal(size=200000)
    m = summary_metrics(mu, sig, y)
    assert abs(m["coverage_80"] - 0.8) < 0.01 and abs(m["coverage_90"] - 0.9) < 0.01
    assert m["pit_reliability_index"] < 0.03


def test_emos_recovers_parameters_and_matches_scipy():
    rng = np.random.default_rng(1)
    n = 20000
    m = rng.normal(10, 5, n); v = rng.gamma(2, 0.5, n)
    y = 0.5 + 0.9 * m + np.sqrt(0.8 ** 2 + 1.5 ** 2 * v) * rng.normal(size=n)
    p, k, info = fit_emos(m, v, y)
    assert np.allclose(np.abs(p[0]), [0.5, 0.9, 0.8, 1.5], atol=0.1)
    f = lambda q: crps_gaussian_np(q[0] + q[1] * m, np.sqrt(q[2] ** 2 + q[3] ** 2 * v), y).mean()
    ref = optimize.minimize(f, [0, 1, 1, 1], method="Nelder-Mead", options={"maxiter": 4000, "xatol": 1e-8, "fatol": 1e-10})
    assert f(p[0]) <= ref.fun + 1e-6


def test_local_emos_equals_separate_fits_and_fallback():
    rng = np.random.default_rng(2)
    g = np.repeat([10, 20], 3000)
    m = rng.normal(0, 3, 6000); v = rng.gamma(2, 0.5, 6000)
    y = np.where(g == 10, 1 + m, -1 + 1.2 * m) + rng.normal(size=6000)
    p, k, _ = fit_emos(m, v, y, groups=g)
    for i, s in enumerate(k):
        ps, _, _ = fit_emos(m[g == s], v[g == s], y[g == s])
        assert np.allclose(np.abs(p[i]), np.abs(ps[0]), atol=1e-3)
    mu, sig, uns = predict_emos(p, k, [0.0, 0.0], [1.0, 1.0], groups=[10, 99], fallback=np.array([7.0, 1, 1, 0]))
    assert list(uns) == [False, True] and mu[1] == 7.0


@pytest.mark.parametrize("spread", ["free_abs", "free_softplus", "constrained"])
@pytest.mark.parametrize("residual", [True, False])
def test_distnet_shapes_and_positive_sigma(spread, residual):
    net = DistNet(5, 4, residual=residual, spread=spread)
    x = torch.randn(7, 5); e = torch.tensor([0, 1, 2, 3, 0, 1, 2]); m = torch.randn(7); s = torch.rand(7)
    mu, sig = net(x, e, m, s)
    assert mu.shape == (7,) and sig.shape == (7,) and (sig > 0).all()
    if residual:
        assert torch.allclose(mu, m)  # residual head starts at the ensemble mean


def test_constrained_sigma_monotone_in_spread_given_coefficients():
    net = DistNet(3, 2, spread="constrained", embedding=False)
    x = torch.zeros(3, 3); e = torch.zeros(3, dtype=torch.long); m = torch.zeros(3)
    _, sig = net(x, e, m, torch.tensor([0.1, 1.0, 3.0]))
    assert sig[0] < sig[1] < sig[2]


def test_preprocessor_unknown_station_gets_unk():
    import pandas as pd
    fit = pd.DataFrame({"station": [1, 2], "a": [0.0, 2.0], "t2m_mean": [1.0, 2.0], "t2m_var": [1.0, 1.0],
                        "obs": [1.0, 2.0], "date": pd.to_datetime(["2015-01-01", "2015-01-02"])})
    pp = Preprocessor().fit(fit, ["a"], min_station_rows=1)
    new = fit.copy(); new["station"] = [2, 3]
    s = pp.transform(new)
    assert s.emb_idx[1] == UNK and s.emb_idx[0] != UNK
    assert np.allclose(pp.transform(fit).x[:, 0], [-1, 1])


def test_dm_and_bh_and_bootstrap():
    rng = np.random.default_rng(3)
    a = rng.normal(0, 1, 400); b = a + 0.3 + rng.normal(0, 0.1, 400)
    r = diebold_mariano(a, b, lag=3)
    assert r["dm_stat"] < -10 and r["p_value"] < 1e-6
    ci = block_bootstrap_ci(a - b, 7, 2000)
    assert ci["ci_high"] < 0
    rej = benjamini_hochberg(np.array([0.001, 0.01, 0.04, 0.5]), q=0.05)
    assert list(rej) == [True, True, False, False]


def test_emos_analytic_gradient():
    from ppnn_residual.emos import _obj_grad
    rng = np.random.default_rng(5)
    m = rng.normal(size=500); v = rng.gamma(2, 0.5, 500); y = m + rng.normal(size=500)
    q = np.array([0.2, 0.9, 0.7, 1.1])
    num = optimize.approx_fprime(q, lambda qq: _obj_grad(qq, m, v, y)[0], 1e-7)
    assert np.allclose(num, _obj_grad(q, m, v, y)[1], atol=1e-5)


def test_boosted_emos_recovers_signal():
    from ppnn_residual.boosting import BoostedEMOS
    rng = np.random.default_rng(7)
    X = rng.normal(size=(20000, 6))
    y = 2 + 1.5 * X[:, 0] + np.exp(0.2 + 0.4 * X[:, 1]) * rng.normal(size=20000)
    b = BoostedEMOS(nu=0.1, maxit=600).fit(X, y)
    mu, sg = b.predict(X)
    assert np.corrcoef(mu, 2 + 1.5 * X[:, 0])[0, 1] > 0.99
    assert np.corrcoef(np.log(sg), X[:, 1])[0, 1] > 0.95
    assert abs(b.ys * b.beta[1] / b.xs[0] - 1.5) < 0.1


def test_stationnet_modes_and_knn():
    from ppnn_residual.stage2_models import StationNet, knn_idw
    x = torch.randn(6, 4); idx = torch.tensor([0, 1, 2, 3, 1, 0]); attr = torch.randn(6, 5)
    m = torch.randn(6); s = torch.rand(6)
    for mode in ["unk", "none", "attr", "hybrid"]:
        for res in [False, True]:
            net = StationNet(4, 4, 5, emb_mode=mode, residual=res, constrained_spread=res)
            mu, sg = net(x, idx, attr, m, s, training=True)
            assert mu.shape == (6,) and (sg > 0).all()
    net = StationNet(4, 4, 5, emb_mode="hybrid")
    with torch.no_grad():
        net.table.weight.normal_()
    v = net.station_vector(idx, attr)
    assert torch.allclose(v[0], net.attr_mlp(attr[:1])[0])  # UNK -> delta = 0
    ref = np.array([[1.0, 0.0], [0.0, 1.0]])
    out = knn_idw(np.array([50.0]), np.array([10.0]), np.array([100.0]), np.array([50.0, 52.0]),
                  np.array([10.0, 10.0]), np.array([100.0, 100.0]), ref, k=1)
    assert np.allclose(out, [[1.0, 0.0]])
