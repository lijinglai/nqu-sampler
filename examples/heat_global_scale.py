"""Example 2: backward heat conduction with an unknown prior scale.

Recover the initial condition u0(s) = sum_k sqrt(2 lam_k) x_k sin(k pi s), lam_k = (k pi)^-2, of the heat equation
u_t = u_ss on (0, 1) with zero boundary values from 30 noisy observations y_j = u(j / 31, T) + eps_j, T = 1e-3,
eps_j ~ N(0, 0.02^2). The prior is x | tau ~ N(0, tau I) with an unknown scale tau ~ InvGamma(3, 2). The forward map is
linear, so the posterior is Gaussian given tau and the exact posterior of tau follows from one-dimensional quadrature.
This is example "Heat" of the paper at d = 100. All directions are Gaussian given tau, so the learning stage selects
no non-Gaussian direction and the global scale is updated by the collapsed (H) update.

Run from the repository root:  python examples/heat_global_scale.py
"""
import math
import time

import numpy as np

try:
    import nqu
except ImportError:                                   # running from a source checkout without installation
    import pathlib
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src'))
    import nqu

D, N_OBS, T, SIGMA, TAU_TRUE = 100, 30, 1e-3, 0.02, 0.5
k = np.arange(1, D + 1)
s = np.arange(1, N_OBS + 1) / (N_OBS + 1.0)
G = (np.sqrt(2.0 * (k * np.pi) ** -2.0) * np.exp(-(k * np.pi) ** 2 * T))[None, :] * np.sin(np.pi * np.outer(s, k))
_rng = np.random.default_rng(20261014)
Y = G @ (math.sqrt(TAU_TRUE) * _rng.standard_normal(D)) + SIGMA * _rng.standard_normal(N_OBS)
PRIOR = nqu.InverseGamma(3.0, 2.0)


def neg_log_lik(x):
    """Negative log-likelihood of one point (shape (D,)) or of many points (shape (n, D))."""
    r = Y - np.asarray(x) @ G.T
    return 0.5 * np.sum(r * r, axis=-1) / SIGMA ** 2


def exact_posterior(n_coef=5):
    """Exact posterior mean and 5%/95% quantiles of tau, and posterior means of the first coefficients x_k."""
    eta = np.linspace(-6.0, 4.0, 4_001)
    tau = np.exp(eta)
    GG = G @ G.T
    lp = np.empty_like(tau)
    coef = np.empty((len(tau), n_coef))
    for i, t in enumerate(tau):
        C = t * GG + SIGMA ** 2 * np.eye(N_OBS)
        sol = np.linalg.solve(C, Y)
        lp[i] = PRIOR.logpdf(t) + eta[i] - 0.5 * np.linalg.slogdet(C)[1] - 0.5 * Y @ sol
        coef[i] = t * (G.T @ sol)[:n_coef]
    w = np.exp(lp - lp.max())
    w /= w.sum()
    cdf = np.cumsum(w)
    return dict(mean=float(w @ tau), q05=float(np.interp(0.05, cdf, tau)), q95=float(np.interp(0.95, cdf, tau)),
                coef=w @ coef)


def main():
    exact = exact_posterior()
    sampler = nqu.NQUSampler(neg_log_lik, D, scale_prior=PRIOR, vectorized=True, seed=100)
    t0 = time.time()
    structure = sampler.learn(n1=20_000, n2=10_000)
    print(structure.summary())
    result = sampler.sample(n_calls=200_000, record=lambda x: x[:5])
    print("  (learning and sampling took %.0f s)" % (time.time() - t0))
    tau = result.tau
    se = tau.std() / math.sqrt(nqu.ess(tau))
    print("\nacceptance %s" % ", ".join("(%s) %.2f" % kv for kv in result.acceptance.items()))
    print("tau: exact mean %.4f, 90%% interval %.3f-%.3f" % (exact['mean'], exact['q05'], exact['q95']))
    print("     estimate   %.4f +- %.4f, 90%% interval %.3f-%.3f; ESS per 1000 calls %.1f"
          % (tau.mean(), se, np.quantile(tau, 0.05), np.quantile(tau, 0.95), nqu.ess_per_1000_calls(tau)))
    est = result.samples.mean(0)
    se_x = result.samples.std(0) / np.sqrt(nqu.ess(result.samples))
    for j in range(5):
        print("x_%d: exact mean %+.4f, estimate %+.4f (z = %+.2f)"
              % (j + 1, exact['coef'][j], est[j], (est[j] - exact['coef'][j]) / se_x[j]))
    worst, calls = result.efficiency()
    print("calls needed for 1000 effective samples (recorded quantities and tau), learning included: %.3g" % calls)


if __name__ == '__main__':
    main()
