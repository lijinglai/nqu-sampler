"""The chain leaves the exact posterior invariant, whether or not the Q model is correct.

Target in d = 6: a bimodal direction z1, a Gaussian direction q whose conditional mean moves with z1, a second
Gaussian direction s and three uninformative directions. Exact moments come from one-dimensional quadrature.
"""
import numpy as np

import _paths  # noqa: F401
import nqu

D = 6
V = np.linalg.qr(np.random.default_rng(10).standard_normal((D, 3)))[0]
LAM_Q, M_Q, BETA = 20.0, 0.4, 0.5
LAM_S, M_S = 2.0, 0.3


def neg_log_lik(x):
    z = np.asarray(x) @ V
    z1, q, s = z[..., 0], z[..., 1], z[..., 2]
    return 5.0 * np.exp(-z1 ** 2) + 0.5 * LAM_Q * (q - M_Q - BETA * z1) ** 2 + 0.5 * LAM_S * (s - M_S) ** 2


def exact_moments():
    g = np.linspace(-8, 8, 160_001)
    cw = LAM_Q / (1 + LAM_Q)
    lp = -0.5 * g * g - 5.0 * np.exp(-g * g) - 0.5 * cw * (M_Q + BETA * g) ** 2
    w = np.exp(lp - lp.max())
    w /= w.sum()
    e1, e2 = float(w @ g), float(w @ g ** 2)
    eq = cw * (M_Q + BETA * e1)
    es = LAM_S * M_S / (1 + LAM_S)
    # quantities: z1, z1^2, q, s, |rho|^2
    return np.array([e1, e2, eq, es, D - 3.0])


def structure(misspecified=False):
    W_N, W_Q = V[:, :1], V[:, 1:]
    ell = np.array([-LAM_Q * M_Q, -LAM_S * M_S])
    B = np.array([[-LAM_Q * BETA], [0.0]])
    H = np.diag([LAM_Q, LAM_S])
    if misspecified:
        ell, B, H = 0.7 * ell, 0.3 * B, 1.8 * H
    return nqu.Structure(W_N=W_N, W_Q=W_Q, ell=ell, B=B, H=H)


def quantities(x):
    z = x @ V
    rho = x - V @ z
    return np.array([z[0], z[0] ** 2, z[1], z[2], rho @ rho])


def check(result):
    est = result.samples.mean(0)
    se = result.samples.std(0) / np.sqrt(nqu.ess(result.samples))
    z = (est - exact_moments()) / se
    assert np.all(np.abs(z) < 5.0), (est, exact_moments(), z)


def test_exact_model():
    sampler = nqu.NQUSampler(neg_log_lik, D, vectorized=True, seed=1)
    result = sampler.sample(n_calls=60_000, structure=structure(), x0=np.zeros(D), record=quantities)
    check(result)
    assert set(result.acceptance) == {'N', 'Q', 'U'}
    assert result.acceptance['U'] > 0.99           # Phi does not depend on rho


def test_misspecified_model_still_exact():
    sampler = nqu.NQUSampler(neg_log_lik, D, vectorized=True, seed=2)
    result = sampler.sample(n_calls=80_000, structure=structure(True), x0=np.zeros(D), record=quantities)
    check(result)


def test_full_space_baseline_runs():
    sampler = nqu.NQUSampler(neg_log_lik, D, vectorized=True, seed=3)
    result = sampler.sample(n_calls=5_000, structure=nqu.Structure.full_space(D), x0=np.zeros(D))
    assert set(result.acceptance) == {'Q'}
    assert result.samples.shape == (4_500, D)


def test_reproducible():
    a = nqu.NQUSampler(neg_log_lik, D, vectorized=True, seed=4).sample(3_000, structure=structure(), x0=np.zeros(D))
    b = nqu.NQUSampler(neg_log_lik, D, vectorized=True, seed=4).sample(3_000, structure=structure(), x0=np.zeros(D))
    assert np.array_equal(a.samples, b.samples)
