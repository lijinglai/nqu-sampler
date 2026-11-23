"""Convergence and efficiency diagnostics.

The effective sample size (ESS) uses Geyer's initial monotone sequence estimator of the integrated autocorrelation
time, as in the paper.
"""
import numpy as np


def _autocorrelation_columns(X):
    """Autocorrelations of the columns of ``X`` (n x p) by FFT; returns an ``(n, p)`` array (lag 0 first row)."""
    n = X.shape[0]
    Xc = X - X.mean(0)
    f = np.fft.rfft(Xc, 2 * n, axis=0)
    acov = np.fft.irfft(f * np.conj(f), axis=0)[:n] / n
    with np.errstate(invalid='ignore', divide='ignore'):
        return acov / acov[0]


def _geyer_tau(rho):
    """Integrated autocorrelation time from autocorrelations ``rho`` by Geyer's initial monotone sequence."""
    if not np.isfinite(rho[0]):
        return np.nan
    n = len(rho)
    m = (n - 1) // 2
    pairs = rho[0:2 * m:2] + rho[1:2 * m + 1:2]
    neg = np.nonzero(pairs < 0)[0]
    if len(neg):
        pairs = pairs[:neg[0]]
    tau = -1.0 + 2.0 * np.minimum.accumulate(pairs).sum()
    return max(tau, 1e-12)


def integrated_autocorrelation_time(x, block=16):
    """Integrated autocorrelation time ``1 + 2 sum_t rho_t`` of a series (1-D) or of each column (2-D)."""
    X = np.asarray(x, dtype=float)
    one = X.ndim == 1
    X = X.reshape(len(X), -1)
    out = np.empty(X.shape[1])
    for a in range(0, X.shape[1], block):
        R = _autocorrelation_columns(X[:, a:a + block])
        for j in range(R.shape[1]):
            out[a + j] = _geyer_tau(R[:, j])
    return float(out[0]) if one else out


def ess(x):
    """Effective sample size of a series (1-D) or of each column (2-D)."""
    X = np.asarray(x, dtype=float)
    tau = integrated_autocorrelation_time(X)
    return len(X) / tau


def ess_per_1000_calls(x, calls_per_sample=1):
    """ESS per 1000 likelihood calls for samples recorded every ``calls_per_sample`` calls."""
    X = np.asarray(x, dtype=float)
    return 1000.0 * ess(X) / (len(X) * calls_per_sample)


def split_rhat(chains):
    """Split potential scale reduction factor of several chains of equal length.

    ``chains`` is a list of arrays of shape ``(n,)`` or ``(n, p)``; returns a float or an array of length ``p``.
    """
    arrays = [np.asarray(c, dtype=float) for c in chains]
    one = arrays[0].ndim == 1
    arrays = [c.reshape(len(c), -1) for c in arrays]
    n = min(len(c) for c in arrays) // 2
    halves = []
    for c in arrays:
        halves += [c[:n], c[n:2 * n]]
    means = np.array([h.mean(0) for h in halves])
    within = np.mean([h.var(0, ddof=1) for h in halves], axis=0)
    between = n * means.var(0, ddof=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        rhat = np.sqrt(((n - 1) / n * within + between / n) / within)
    return float(rhat[0]) if one else rhat


def summarize(samples, names=None, reference=None, calls_per_sample=1, max_rows=None):
    """Text table of the mean, standard deviation, ESS per 1000 calls and Monte Carlo standard error of each column.

    If ``reference`` (an array of reference means) is given, the table also shows ``z = (mean - reference) / se``.
    """
    X = np.asarray(samples, dtype=float)
    X = X.reshape(len(X), -1)
    p = X.shape[1]
    names = names or ["x[%d]" % j for j in range(p)]
    e = ess(X)
    mean, sd = X.mean(0), X.std(0)
    se = sd / np.sqrt(e)
    header = "%-12s %12s %12s %14s %12s" % ("quantity", "mean", "sd", "ESS/1000 calls", "MC s.e.")
    if reference is not None:
        header += " %10s %8s" % ("reference", "z")
    lines = [header]
    rows = range(p) if max_rows is None else range(min(p, max_rows))
    for j in rows:
        line = "%-12s %12.5g %12.5g %14.4g %12.3g" % (names[j], mean[j], sd[j],
                                                       1000.0 * e[j] / (len(X) * calls_per_sample), se[j])
        if reference is not None:
            line += " %10.5g %8.2f" % (reference[j], (mean[j] - reference[j]) / se[j])
        lines.append(line)
    if max_rows is not None and p > max_rows:
        lines.append("... (%d more)" % (p - max_rows))
    return "\n".join(lines)
