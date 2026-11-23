"""Empirical Hermite coefficient tensors of orders 3 and 4 (used by Step 2 and by the rotation of the N basis)."""
import numpy as np


def hermite_tensors(z, w3, w4):
    """Empirical third- and fourth-order Hermite coefficient tensors.

    Parameters
    ----------
    z:
        Design points, shape ``(n, k)``, standard normal.
    w3, w4:
        Weights (usually residuals of the negative log-likelihood), shape ``(n,)``.

    Returns
    -------
    T3, T4:
        ``mean(w3 * He_3(z))`` as a ``(k, k**2)`` matrix and ``mean(w4 * He_4(z))`` as a ``(k, k**3)`` matrix, where
        ``He_3`` and ``He_4`` are the multivariate Hermite tensors, unfolded along the first index.
    """
    n, k = z.shape
    eye = np.eye(k)
    pairs = (z[:, :, None] * z[:, None, :]).reshape(n, k * k)
    m1 = (w3[:, None] * z).mean(0)
    m3 = ((w3[:, None] * z).T @ pairs / n).reshape(k, k, k)
    T3 = m3 - (np.einsum('i,jk->ijk', m1, eye) + np.einsum('j,ik->ijk', m1, eye) + np.einsum('k,ij->ijk', m1, eye))
    m0 = w4.mean()
    m2 = (w4[:, None] * z).T @ z / n
    m4 = ((w4[:, None] * pairs).T @ pairs / n).reshape(k, k, k, k)
    s6 = (np.einsum('ij,kl->ijkl', m2, eye) + np.einsum('ik,jl->ijkl', m2, eye) + np.einsum('il,jk->ijkl', m2, eye)
          + np.einsum('jk,il->ijkl', m2, eye) + np.einsum('jl,ik->ijkl', m2, eye) + np.einsum('kl,ij->ijkl', m2, eye))
    s3 = np.einsum('ij,kl->ijkl', eye, eye) + np.einsum('ik,jl->ijkl', eye, eye) + np.einsum('il,jk->ijkl', eye, eye)
    T4 = m4 - s6 + m0 * s3
    return T3.reshape(k, k * k), T4.reshape(k, k ** 3)


def cross_fitted_m34(w, e):
    """Cross-fitted matrix ``M_34`` of Step 2.

    The design is split into two halves; with ``T3^(h), T4^(h)`` the tensors of half ``h``,
    ``M_34 = (T3^(1) T3^(2)^T + T3^(2) T3^(1)^T) / (2 * 3!) + (T4^(1) T4^(2)^T + T4^(2) T4^(1)^T) / (2 * 4!)``,
    which has no positive noise bias.
    """
    h = len(w) // 2
    T3a, T4a = hermite_tensors(w[:h], e[:h], e[:h])
    T3b, T4b = hermite_tensors(w[h:], e[h:], e[h:])
    M = (T3a @ T3b.T + T3b @ T3a.T) / 12.0 + (T4a @ T4b.T + T4b @ T4a.T) / 48.0
    return 0.5 * (M + M.T)
