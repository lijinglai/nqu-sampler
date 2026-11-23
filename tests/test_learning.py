"""Learning recovers a bimodal and a spiky direction among Gaussian and uninformative ones."""
import os
import tempfile

import numpy as np

import _paths  # noqa: F401
import nqu

D = 30
V = np.linalg.qr(np.random.default_rng(30).standard_normal((D, 6)))[0]
LAM = np.array([100.0, 30.0, 10.0, 3.0])
M = np.array([0.5, -0.5, 0.5, -0.5])


def neg_log_lik(x):
    z = np.asarray(x) @ V
    return (5.0 * np.exp(-z[..., 0] ** 2) + np.log1p(400.0 * z[..., 1] ** 2)
            + 0.5 * np.sum(LAM * (z[..., 2:] - M) ** 2, axis=-1))


def test_recovers_blocks():
    sampler = nqu.NQUSampler(neg_log_lik, D, vectorized=True, seed=7)
    st = sampler.learn(n1=8_000, n2=6_000)
    assert st.r == 2, st.summary()
    in_n = np.sum((st.W_N.T @ V[:, :2]) ** 2, axis=0)
    assert np.all(in_n > 0.9), in_n
    in_s = np.sum((np.column_stack([st.W_N, st.W_Q]).T @ V) ** 2, axis=0)
    assert np.all(in_s > 0.9), in_s
    assert st.info['calls'] == 14_000


def test_learning_is_reproducible_and_saves():
    a = nqu.NQUSampler(neg_log_lik, D, vectorized=True, seed=8).learn(n1=4_000, n2=3_000)
    b = nqu.NQUSampler(neg_log_lik, D, vectorized=False, seed=8).learn(n1=4_000, n2=3_000)
    assert np.allclose(a.W_N, b.W_N) and np.allclose(a.W_Q, b.W_Q) and np.allclose(a.H, b.H)
    path = os.path.join(tempfile.mkdtemp(), 'structure.npz')
    a.save(path)
    c = nqu.Structure.load(path)
    assert np.array_equal(a.W_N, c.W_N) and np.array_equal(a.B, c.B) and c.info['calls'] == a.info['calls']
