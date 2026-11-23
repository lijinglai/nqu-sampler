"""Helpers: batch evaluation of the negative log-likelihood and whitening of a Gaussian prior."""
import numpy as np


def evaluate_many(neg_log_lik, X, vectorized=False, map_fn=None):
    """Evaluate the negative log-likelihood at the rows of ``X`` (n x d) and return an array of length n.

    Parameters
    ----------
    neg_log_lik:
        Function of one point ``x`` (shape ``(d,)``) returning a float; if ``vectorized`` is true it must also accept
        an array of shape ``(n, d)`` and return an array of shape ``(n,)``.
    vectorized:
        Call ``neg_log_lik`` once on the whole array.
    map_fn:
        Optional ``map``-like function used to evaluate the rows one by one, for example ``pool.map`` of a
        ``multiprocessing.Pool`` or ``concurrent.futures`` executor, to run an expensive forward model in parallel.
        Ignored if ``vectorized`` is true.
    """
    X = np.asarray(X, dtype=float)
    if vectorized:
        y = np.asarray(neg_log_lik(X), dtype=float).reshape(-1)
    else:
        rows = list(X)
        values = (map_fn or map)(neg_log_lik, rows)
        y = np.fromiter((float(v) for v in values), dtype=float, count=len(rows))
    if y.shape[0] != X.shape[0]:
        raise ValueError("neg_log_lik returned %d values for %d points" % (y.shape[0], X.shape[0]))
    if not np.all(np.isfinite(y)):
        raise ValueError("neg_log_lik returned non-finite values on the learning design")
    return y


def whiten(neg_log_lik_theta, prior_sqrt):
    """Express a negative log-likelihood of ``theta`` in whitened coordinates ``x``.

    The NQU sampler works with ``x ~ N(0, I)`` a priori. For a prior ``theta ~ N(0, C0)``, pass ``theta = C0^{1/2} x``:
    ``prior_sqrt`` is either a matrix ``C0^{1/2}`` (d x d, any square root) or a function mapping ``x`` to ``theta``.
    Returns a function of ``x`` that works on single points and on arrays of shape ``(n, d)`` if
    ``neg_log_lik_theta`` does.
    """
    if callable(prior_sqrt):
        to_theta = prior_sqrt
    else:
        S = np.asarray(prior_sqrt, dtype=float)

        def to_theta(x):
            return np.asarray(x) @ S.T

    def neg_log_lik_x(x):
        return neg_log_lik_theta(to_theta(x))

    return neg_log_lik_x
