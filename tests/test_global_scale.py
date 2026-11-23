"""Global prior scale: a linear Gaussian model with theta | tau ~ N(0, tau I), tau ~ InvGamma(3, 2).

The posterior of tau is computed exactly by quadrature over log(tau), using y | tau ~ N(0, tau G G^T + sigma^2 I).
"""
import math

import numpy as np

import _paths  # noqa: F401
import nqu

D, M, SIGMA = 8, 3, 0.3
_rng = np.random.default_rng(20)
G = _rng.standard_normal((M, D))
Y = G @ (math.sqrt(0.5) * _rng.standard_normal(D)) + SIGMA * _rng.standard_normal(M)
PRIOR = nqu.InverseGamma(3.0, 2.0)


def neg_log_lik(x):
    r = Y - np.asarray(x) @ G.T
    return 0.5 * np.sum(r * r, axis=-1) / SIGMA ** 2


def exact_tau_mean():
    eta = np.linspace(-8.0, 5.0, 20_001)
    tau = np.exp(eta)
    lp = np.empty_like(tau)
    for i, t in enumerate(tau):
        C = t * G @ G.T + SIGMA ** 2 * np.eye(M)
        _, logdet = np.linalg.slogdet(C)
        lp[i] = PRIOR.logpdf(t) + eta[i] - 0.5 * logdet - 0.5 * Y @ np.linalg.solve(C, Y)
    w = np.exp(lp - lp.max())
    w /= w.sum()
    return float(w @ tau)


def exact_structure():
    """The exact class: the informed directions span range(G^T), the model is the likelihood itself."""
    W_Q = np.linalg.qr(G.T)[0]
    ell = -W_Q.T @ G.T @ Y / SIGMA ** 2
    H = W_Q.T @ G.T @ G @ W_Q / SIGMA ** 2
    return nqu.Structure(W_N=np.zeros((D, 0)), W_Q=W_Q, ell=ell, B=np.zeros((M, 0)), H=H)


def test_tau_posterior_mean():
    sampler = nqu.NQUSampler(neg_log_lik, D, scale_prior=PRIOR, vectorized=True, seed=5)
    result = sampler.sample(n_calls=60_000, structure=exact_structure(), x0=np.zeros(D))
    assert result.acceptance['H'] > 0.99           # exact class: (H) is a partially collapsed Gibbs step
    se = result.tau.std() / math.sqrt(nqu.ess(result.tau))
    z = (result.tau.mean() - exact_tau_mean()) / se
    assert abs(z) < 5.0, (result.tau.mean(), exact_tau_mean(), z)


def test_learned_structure_with_scale():
    sampler = nqu.NQUSampler(neg_log_lik, D, scale_prior=PRIOR, vectorized=True, seed=6)
    st = sampler.learn(n1=3_000, n2=2_000)
    assert st.r == 0 and st.k >= M
    result = sampler.sample(n_calls=40_000)
    se = result.tau.std() / math.sqrt(nqu.ess(result.tau))
    assert abs(result.tau.mean() - exact_tau_mean()) < 5.0 * se
