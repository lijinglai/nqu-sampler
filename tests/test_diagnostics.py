"""Tests of the ESS and R-hat diagnostics on series with known answers."""
import numpy as np

import _paths  # noqa: F401
import nqu


def test_ess_of_independent_draws():
    x = np.random.default_rng(0).standard_normal(100_000)
    e = nqu.ess(x)
    assert 0.9 * len(x) < e < 1.1 * len(x), e


def test_iact_of_ar1():
    rng = np.random.default_rng(1)
    n, a = 200_000, 0.9
    eps = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = eps[0] / np.sqrt(1 - a * a)
    for t in range(1, n):
        x[t] = a * x[t - 1] + eps[t]
    tau = nqu.integrated_autocorrelation_time(x)
    exact = (1 + a) / (1 - a)
    assert abs(tau - exact) / exact < 0.15, tau


def test_columnwise_ess_matches_single_series():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((5_000, 3))
    e = nqu.ess(X)
    assert e.shape == (3,)
    assert abs(e[1] - nqu.ess(X[:, 1])) < 1e-6


def test_split_rhat():
    rng = np.random.default_rng(3)
    good = [rng.standard_normal(10_000) for _ in range(4)]
    assert nqu.split_rhat(good) < 1.01
    bad = [rng.standard_normal(10_000) + shift for shift in (0.0, 0.0, 1.0, 1.0)]
    assert nqu.split_rhat(bad) > 1.1
