import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ppnn_residual.device import get_device  # noqa: E402


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


run = _load("stage3_run")
pre = _load("stage3_preprocess")


def _toy(device, n=3000, seed=0):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, 6)).astype(np.float32)
    idx = rng.integers(1, 6, n)
    attr = rng.normal(size=(6, 10))[idx].astype(np.float32)
    m = Z[:, 0] * 3; sd = np.abs(Z[:, 1]) + 0.5; y = m + rng.normal(size=n) * sd + idx * 0.3
    return run.Arrays(Z, idx, attr, m, sd, y, device)


def _train(device, var):
    hp = dict(run.PROT["nn_hyperparameters"]); hp["batch_size"] = 256
    A = _toy(device); V = _toy(device, 800, 1)
    model, hist, be, best = run.train_variant(var, A, V, 6, 6, 10, 0, hp, device, 2)
    return run.predict(model, V), hist


def test_device_autodetect():
    d = get_device("auto")
    assert d.type == ("cuda" if torch.cuda.is_available() else "cpu")
    assert get_device("cpu").type == "cpu"


@pytest.mark.parametrize("var", ["unk", "hybrid", "noemb"])
def test_cpu_training_is_bitwise_reproducible(var):
    (mu1, s1), h1 = _train(torch.device("cpu"), var)
    (mu2, s2), h2 = _train(torch.device("cpu"), var)
    assert np.array_equal(mu1, mu2) and np.array_equal(s1, s2)
    assert [h["val_crps"] for h in h1] == [h["val_crps"] for h in h2]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_gpu_matches_cpu_closely_and_is_reproducible():
    (mc, sc), _ = _train(torch.device("cpu"), "hybrid")
    (mg1, sg1), _ = _train(torch.device("cuda"), "hybrid")
    (mg2, sg2), _ = _train(torch.device("cuda"), "hybrid")
    assert np.array_equal(mg1, mg2) and np.array_equal(sg1, sg2)
    assert np.max(np.abs(mc - mg1)) < 1e-2  # same random stream; only float arithmetic differs


def test_ens_stats_and_landuse():
    a = np.arange(24, dtype=float).reshape(2, 3, 4)
    m, s = pre.ens_stats(a, 1)
    assert np.allclose(m, a.mean(1)) and np.allclose(s, a.std(1, ddof=1))
    m11, _ = pre.ens_stats(a, 1, [0, 1])
    assert np.allclose(m11, a[:, :2].mean(1))
    assert [pre.landuse_group(c) for c in (1, 12, 23, 35, 44, 48)] == [0, 1, 2, 3, 4, 4]


def test_seen_accumulator_idempotent(tmp_path):
    p = tmp_path / "seen.npz"
    run.accumulate_seen(p, 0, np.ones(3), np.ones(3))
    run.accumulate_seen(p, 0, np.ones(3), np.ones(3))
    run.accumulate_seen(p, 1, 2 * np.ones(3), np.ones(3))
    z = np.load(p)
    assert list(z["seeds"]) == [0, 1] and np.allclose(z["sum_mu"], 3)


def test_boosting_ignores_constant_columns():
    from ppnn_residual.boosting import BoostedEMOS
    rng = np.random.default_rng(0)
    X = np.c_[rng.normal(size=500), np.zeros(500), rng.normal(size=500)]
    y = 2 * X[:, 0] + rng.normal(size=500)
    b = BoostedEMOS(maxit=200).fit(X, y)
    mu, sg = b.predict(X)
    assert np.all(np.isfinite(mu)) and np.all(np.isfinite(sg)) and b.beta[2] == 0.0
