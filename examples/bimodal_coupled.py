"""Example 1: a bimodal direction coupled to stiff Gaussian directions.

This is example M2 of the paper at d = 100. The likelihood depends on 12 orthonormal directions: a bimodal direction
z1, a spike z2 and ten Gaussian directions q_1, ..., q_10 whose conditional means move linearly with z1; the other 88
directions are uninformative. Moving between the two modes of z1 requires moving the stiff q_j along with it, which
the (N) update of the NQU sampler does through the learned Q model.

The script learns the structure, runs the NQU sampler, runs full-space robust adaptive Metropolis (RAM) with the same
total number of likelihood calls, and compares both with exact posterior moments from one-dimensional quadrature.

Run from the repository root:  python examples/bimodal_coupled.py
"""
import time

import numpy as np

try:
    import nqu
except ImportError:                                   # running from a source checkout without installation
    import pathlib
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
    import nqu

D = 100
V = np.linalg.qr(np.random.default_rng(1).standard_normal((D, 12)))[0]     # z1, z2, q_1..q_10
LAM = np.array([1000.0, 400.0, 200.0, 100.0, 50.0, 25.0, 10.0, 5.0, 3.0, 1.5])
M = np.array([0.5, -0.5] * 5)
BETA = 0.3
NAMES = ['z1', 'z2'] + ['q%d' % (j + 1) for j in range(10)]


def neg_log_lik(x):
    """Negative log-likelihood of one point (shape (D,)) or of many points (shape (n, D))."""
    z = np.asarray(x) @ V
    z1, z2, q = z[..., 0], z[..., 1], z[..., 2:]
    s = q - M - BETA * z1[..., None]
    return 5.0 * np.exp(-z1 ** 2) + np.log1p(400.0 * z2 ** 2) + 0.5 * np.sum(LAM * s ** 2, axis=-1)


def functionals(x):
    """Recorded quantities: the coordinates of x along the 12 informed directions."""
    return np.asarray(x) @ V


def exact_means():
    """Exact posterior means of z1, z2, q_1, ..., q_10 and of z1^2, z2^2."""
    g = np.linspace(-8.0, 8.0, 1_600_001)
    cw = LAM / (1.0 + LAM)
    lp = -0.5 * g * g - 5.0 * np.exp(-g * g) - 0.5 * np.sum(cw[:, None] * (M[:, None] + BETA * g) ** 2, axis=0)
    w = np.exp(lp - lp.max())
    w /= w.sum()
    m1, m1sq = float(w @ g), float(w @ g ** 2)
    w2 = np.exp(-0.5 * g * g - np.log1p(400.0 * g * g))
    w2 /= w2.sum()
    m2sq = float(w2 @ g ** 2)
    means = np.r_[m1, 0.0, cw * (M + BETA * m1)]
    return means, m1sq, m2sq


def report(label, result, exact):
    means, m1sq, m2sq = exact
    X = result.samples
    F = np.column_stack([X, X[:, 0] ** 2, X[:, 1] ** 2])
    ref = np.r_[means, m1sq, m2sq]
    se = F.std(0) / np.sqrt(nqu.ess(F))
    z = (F.mean(0) - ref) / se
    worst, calls = result.efficiency()
    print("\n%s: acceptance %s" % (label, ", ".join("(%s) %.2f" % kv for kv in result.acceptance.items())))
    print("  minimum ESS per 1000 calls (means and second moments of the 12 coordinates): %.3g" % worst)
    print("  calls needed for 1000 effective samples, learning included: %.3g" % calls)
    for name, j in (('E z1', 0), ('E z1^2', 12), ('E z2^2', 13), ('E q1', 2), ('E q7', 8)):
        print("  %-7s exact %8.4f   estimate %8.4f   z = %+.2f" % (name, ref[j], F[:, j].mean(), z[j]))


def main():
    exact = exact_means()
    sampler = nqu.NQUSampler(neg_log_lik, D, vectorized=True, seed=100)
    t0 = time.time()
    structure = sampler.learn(n1=20_000, n2=10_000)
    print(structure.summary())
    in_n = np.sum((structure.W_N.T @ V[:, :2]) ** 2, axis=0)
    print("  squared length of z1, z2 in the N block: %.2f, %.2f" % tuple(in_n))
    n_calls = 200_000
    result = sampler.sample(n_calls=n_calls, record=functionals)
    print("  (learning and sampling took %.0f s)" % (time.time() - t0))
    report("NQU sampler", result, exact)

    # Full-space RAM with the same total number of calls (the NQU sampler without learning).
    baseline = sampler.sample(n_calls=n_calls + structure.info['calls'], structure=nqu.Structure.full_space(D),
                              x0=np.random.default_rng(0).standard_normal(D), record=functionals)
    report("Full-space RAM (same number of calls)", baseline, exact)


if __name__ == '__main__':
    main()
