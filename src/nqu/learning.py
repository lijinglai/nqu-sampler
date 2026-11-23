"""Learning stage of the NQU sampler (Steps 1-3 of the paper).

Step 1 (quadratic sensing) fits a signed low-rank quadratic model to likelihood values at prior draws and returns the
informed subspace ``S``. Step 2 draws a second design from a reference Gaussian on ``S``, estimates the third- and
fourth-order Hermite tensors of the residual and selects non-Gaussian directions one at a time with a sign-flip test.
The selected directions are mapped back to ``x``, the basis of the N block is optionally rotated, and Step 3 reads the
Q model off the Step 1 model. Only Steps 1 and 2 call the likelihood (``n1 + n2`` calls).
"""
import math

import numpy as np

from .hermite import cross_fitted_m34, hermite_tensors
from .options import LearningOptions
from .structure import Structure
from .utils import evaluate_many


# ---------------------------------------------------------------------------------------------------------- Step 1
def _quadratic_part(X, L, signs):
    XL = X @ L
    return 0.5 * (XL * XL) @ signs, XL


def _scaled_gd(X, y, Qa, L, signs, max_iter, tol):
    """Scaled gradient descent with Armijo backtracking on ``f(L) = |P (y - q_L(X))|^2 / (2n)``.

    ``q_L(x) = x^T L diag(signs) L^T x / 2``; ``P`` projects out the constant and linear terms (orthonormal basis
    ``Qa``); the signs are fixed. Returns the refined ``L``, the projected residual and the number of iterations.
    """
    n = len(y)
    q, XL = _quadratic_part(X, L, signs)
    res = y - q
    res = res - Qa @ (Qa.T @ res)
    f = 0.5 * (res @ res) / n
    eta = 1.0
    it = 0
    for it in range(max_iter):
        G = -(X.T @ (res[:, None] * XL)) * signs[None, :] / n
        S = G @ np.linalg.inv(L.T @ L)
        gs = float(np.sum(G * S))
        while True:
            L_new = L - eta * S
            q_new, XL_new = _quadratic_part(X, L_new, signs)
            res_new = y - q_new
            res_new = res_new - Qa @ (Qa.T @ res_new)
            f_new = 0.5 * (res_new @ res_new) / n
            if f_new <= f - 1e-4 * eta * gs or eta < 1e-14:
                break
            eta *= 0.5
        rel = (f - f_new) / max(f, 1e-300)
        L, XL, res, f = L_new, XL_new, res_new, f_new
        eta = min(2.0 * eta, 1e4)
        if rel < tol:
            break
    return L, res, it + 1


def fit_quadratic_model(X, y, options=None, log=None):
    """Step 1: fit ``q(x) = kappa0 + g^T x + x^T L D L^T x / 2`` with signs ``D = diag(+-1)``, one rank at a time.

    The first ``validation_fraction`` of the points is held out. For the current residual, the eigenpair of largest
    modulus of the principal Hessian direction matrix gives a new column, all columns are refined by scaled gradient
    descent with fixed signs, and the constant and linear terms are refitted by least squares. The new column is kept
    only if it lowers the validation error by at least ``stop_tolerance`` (relative).

    Returns a dict with ``kappa0``, ``g``, ``L``, ``signs`` and the ``history`` of the ranks tried.
    """
    opt = options or LearningOptions()
    n = len(y)
    nv = int(round(opt.validation_fraction * n))
    Xt, yt, Xv, yv = X[nv:], y[nv:], X[:nv], y[:nv]
    nt, d = Xt.shape
    Qa, Ra = np.linalg.qr(np.column_stack([np.ones(nt), Xt]))

    def coefficients_and_validation(L, signs):
        q, _ = _quadratic_part(Xt, L, signs)
        cf = np.linalg.solve(Ra, Qa.T @ (yt - q))
        qv, _ = _quadratic_part(Xv, L, signs)
        return cf, float(np.mean((yv - cf[0] - Xv @ cf[1:] - qv) ** 2))

    L = np.zeros((d, 0))
    signs = np.zeros(0)
    cf, mse = coefficients_and_validation(L, signs)
    res = yt - Qa @ (Qa.T @ yt)
    history = [dict(rank=0, validation_mse=mse, phd_eigenvalue=0.0, iterations=0)]
    while L.shape[1] < opt.max_rank:
        M_phd = 0.5 * ((Xt.T * res) @ Xt / nt - res.mean() * np.eye(d))
        ev, vecs = np.linalg.eigh(M_phd)
        i = int(np.argmax(np.abs(ev)))
        L_new = np.column_stack([L, vecs[:, i] * math.sqrt(2.0 * abs(ev[i]))])
        signs_new = np.r_[signs, np.sign(ev[i])]
        L_new, res_new, its = _scaled_gd(Xt, yt, Qa, L_new, signs_new, opt.gd_max_iter, opt.gd_tol)
        cf_new, mse_new = coefficients_and_validation(L_new, signs_new)
        history.append(dict(rank=L_new.shape[1], validation_mse=mse_new, phd_eigenvalue=float(ev[i]), iterations=its))
        if log is not None:
            log("  Step 1, rank %2d: PHD eigenvalue %+.3f, %3d ScaledGD iterations, validation error %.4g (%.1f%% lower)"
                % (L_new.shape[1], ev[i], its, mse_new, 100.0 * (mse - mse_new) / mse))
        if (mse - mse_new) / mse < opt.stop_tolerance:
            break
        L, signs, res, cf, mse = L_new, signs_new, res_new, cf_new, mse_new
    return dict(kappa0=float(cf[0]), g=cf[1:], L=L, signs=signs, history=history)


def informed_subspace(g, L, signs, delta):
    """Orthonormal basis of the span of the eigenvectors of ``g g^T + H^2`` with eigenvalues above ``delta**2``.

    Returns ``(F_S, eigenvalues, H)`` with the eigenvalues in decreasing order and ``H = L diag(signs) L^T``.
    """
    H = (L * signs) @ L.T
    K = np.outer(g, g) + H @ H
    ev, vecs = np.linalg.eigh(K)
    keep = ev > delta ** 2
    return vecs[:, keep][:, ::-1], ev[keep][::-1], H


# ---------------------------------------------------------------------------------------------------------- Step 2
def reference_gaussian(F_S, H, g, floor):
    """Reference Gaussian of Step 2 on ``S``: precision ``A_S = I + F_S^T H F_S`` with eigenvalues floored, and mean
    ``mu_S = -A_S^{-1} F_S^T g``. Returns ``(A_half, A_mhalf, mu_S, clipped eigenvalues)``."""
    m = F_S.shape[1]
    A = np.eye(m) + F_S.T @ H @ F_S
    A = 0.5 * (A + A.T)
    w, Ua = np.linalg.eigh(A)
    w = np.maximum(w, floor)
    A_c = (Ua * w) @ Ua.T
    A_half = (Ua * np.sqrt(w)) @ Ua.T
    A_mhalf = (Ua / np.sqrt(w)) @ Ua.T
    mu_S = -np.linalg.solve(A_c, F_S.T @ g)
    return A_half, A_mhalf, mu_S, w


def select_nongaussian(w, res, rng, options, log=None):
    """Sequential selection of non-Gaussian directions in ``w`` space.

    ``w`` are the Step 2 design points (standard normal, shape ``(n2, m)``) and ``res`` the centred residuals. The
    statistic after ``j`` selections is the largest eigenvalue of ``M_34`` on the directions not yet selected; its
    null distribution is approximated by random sign flips of the residuals. Returns the selected orthonormal
    directions ``E`` (``m x r``), the test records and the eigenvalues of ``M_34``.
    """
    opt = options
    n2, m = w.shape
    M = cross_fitted_m34(w, res)
    nulls = [cross_fitted_m34(w, res * rng.choice([-1.0, 1.0], n2)) for _ in range(opt.n_flips)]
    selected, records = [], []
    basis = np.eye(m)                                  # orthonormal basis of the directions not yet selected
    while len(selected) < opt.r_max and basis.shape[1] > 0:
        ev, vecs = np.linalg.eigh(basis.T @ M @ basis)
        null_stats = np.array([np.linalg.eigvalsh(basis.T @ M0 @ basis)[-1] for M0 in nulls])
        threshold = float(np.quantile(null_stats, 1.0 - opt.level))
        p_value = float((1 + np.sum(null_stats >= ev[-1])) / (1 + len(null_stats)))
        records.append(dict(step=len(selected), statistic=float(ev[-1]), threshold=threshold,
                            null_median=float(np.median(null_stats)), p_value=p_value))
        if log is not None:
            log("  Step 2, after %d selections: statistic %.4g, threshold %.4g, p = %.3g"
                % (len(selected), ev[-1], threshold, p_value))
        if not ev[-1] > threshold:
            break
        selected.append(basis @ vecs[:, -1])
        Qs, _ = np.linalg.qr(np.column_stack(selected + [np.eye(m)]))
        basis = Qs[:, len(selected):m]
    E = np.column_stack(selected) if selected else np.zeros((m, 0))
    return E, records, np.linalg.eigvalsh(M)[::-1]


def back_map(F_S, A_mhalf, E):
    """Map the selected ``w``-space directions to ``x``: ``W_N = orth(F_S A_S^{-1/2} E)``; ``W_Q`` spans the rest of
    ``S``. The columns of ``W_N`` are the displacement directions of the selected functionals under the reference
    Gaussian, and ``c = W_Q^T x`` is an affine function of the unselected coordinates of ``w``."""
    d, m = F_S.shape
    if E.shape[1] == 0:
        W_N = np.zeros((d, 0))
    else:
        W_N, _ = np.linalg.qr(F_S @ (A_mhalf @ E))
    Qf, _ = np.linalg.qr(np.column_stack([F_S.T @ W_N, np.eye(m)]))
    return W_N, F_S @ Qf[:, W_N.shape[1]:m]


# ------------------------------------------------------------------------------------------- rotation of the N basis
def _contrast(R, T3, T4):
    J = 0.0
    for a in range(R.shape[1]):
        u = R[:, a]
        c3 = [float(np.einsum('ijk,i,j,k->', T, u, u, u)) for T in T3]
        c4 = [float(np.einsum('ijkl,i,j,k,l->', T, u, u, u, u)) for T in T4]
        J += c3[0] * c3[1] / 6.0 + c4[0] * c4[1] / 24.0
    return J


def _golden_section_min(f, lo, hi, tol):
    """Minimise a scalar function on ``[lo, hi]`` by golden-section search; returns ``(x, f(x))``."""
    g = 0.5 * (math.sqrt(5.0) - 1.0)
    a, b = lo, hi
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = f(c), f(d)
    while b - a > tol:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = f(d)
    x = 0.5 * (a + b)
    return x, f(x)


def _jacobi_rotation(T3, T4, r, n_grid, max_sweeps, tol):
    """Maximise the cross-fitted contrast ``J(R)`` over orthogonal ``R`` by Jacobi sweeps of plane rotations."""
    R = np.eye(r)
    J = J0 = _contrast(R, T3, T4)
    sweeps = 0
    grid = np.linspace(0.0, 0.5 * math.pi, n_grid, endpoint=False)
    step = grid[1] - grid[0]
    for sweeps in range(1, max_sweeps + 1):
        J_old = J
        for a in range(r - 1):
            for b in range(a + 1, r):
                def rotated(t, R=R, a=a, b=b):
                    G = np.eye(r)
                    c, s = math.cos(t), math.sin(t)
                    G[a, a] = c
                    G[b, b] = c
                    G[a, b] = -s
                    G[b, a] = s
                    return R @ G

                def f(t):
                    return -_contrast(rotated(t), T3, T4)

                values = np.array([f(t) for t in grid])
                i = int(np.argmin(values))
                t_ref, f_ref = _golden_section_min(f, grid[i] - step, grid[i] + step, 1e-10)
                t_best = t_ref if f_ref < values[i] else float(grid[i])
                J_new = -f(t_best)
                if J_new > J:
                    R = rotated(t_best)
                    J = J_new
        if J - J_old <= tol * max(abs(J), 1e-300):
            break
    return R, J0, J, sweeps


def rotate_n_basis(w, res, E, options):
    """Rotation of the selected directions that makes the third- and fourth-order tensors in the rotated
    coordinates as diagonal as possible (an ICA-type criterion, cross-fitted on the two halves of the Step 2 design).
    Returns ``(E R, info)``."""
    r = E.shape[1]
    if r < 2:
        return E, dict(sweeps=0)
    z = w @ E
    h = len(z) // 2
    T3a, T4a = hermite_tensors(z[:h], res[:h], res[:h])
    T3b, T4b = hermite_tensors(z[h:], res[h:], res[h:])
    T3 = (T3a.reshape((r,) * 3), T3b.reshape((r,) * 3))
    T4 = (T4a.reshape((r,) * 4), T4b.reshape((r,) * 4))
    R, J0, J1, sweeps = _jacobi_rotation(T3, T4, r, options.rotation_grid, options.rotation_max_sweeps,
                                         options.rotation_tol)
    return E @ R, dict(J0=float(J0), J1=float(J1), sweeps=int(sweeps), R=R.tolist())


def closest_basis(W_N, F_S, A_half, E_rotated):
    """Orthonormal basis of ``range(W_N)`` closest to the functional directions ``F_S A_S^{1/2} (E R)_j``:
    ``W_N polar(W_N^T Gamma)`` with ``Gamma`` the normalised directions."""
    C = F_S @ (A_half @ E_rotated)
    C = C / np.linalg.norm(C, axis=0)
    u, _, vt = np.linalg.svd(W_N.T @ C, full_matrices=False)
    return W_N @ (u @ vt)


# ---------------------------------------------------------------------------------------------------------- Step 3
def q_model(g, H, W_N, W_Q):
    """Step 3: the part of the Step 1 model on ``(a, c)``: ``ell = W_Q^T g``, ``B = W_Q^T H W_N``,
    ``H_Q = W_Q^T H W_Q``."""
    H_Q = W_Q.T @ H @ W_Q
    return W_Q.T @ g, W_Q.T @ H @ W_N, 0.5 * (H_Q + H_Q.T)


# ------------------------------------------------------------------------------------------------------ all steps
def learn_structure(neg_log_lik, dim, seed, options=None, vectorized=False, map_fn=None, log=None):
    """Run Steps 1-3 and return the learned :class:`~nqu.structure.Structure`.

    Parameters
    ----------
    neg_log_lik:
        Negative log-likelihood in whitened coordinates (``x ~ N(0, I)`` a priori).
    dim:
        Dimension ``d``.
    seed:
        Integer seed; the designs use the random stream ``numpy.random.default_rng([seed, 7])``.
    options:
        :class:`~nqu.options.LearningOptions`.
    vectorized, map_fn:
        How the designs are evaluated, see :func:`~nqu.utils.evaluate_many`.
    log:
        Optional function receiving progress messages (for example ``print``).
    """
    opt = options or LearningOptions()
    rng = np.random.default_rng([int(seed), 7])
    d = int(dim)
    X1 = rng.standard_normal((opt.n1, d))
    y1 = evaluate_many(neg_log_lik, X1, vectorized, map_fn)
    s1 = fit_quadratic_model(X1, y1, opt, log)
    F_S, eig_S, H = informed_subspace(s1['g'], s1['L'], s1['signs'], opt.delta)
    m = F_S.shape[1]
    info = dict(calls=opt.n1, step1_rank=int(s1['L'].shape[1]), informed_dim=int(m), step1_history=s1['history'],
                informed_eigenvalues=eig_S.tolist(), n1=opt.n1, n2=opt.n2)
    if log is not None:
        log("  Step 1: rank %d, informed subspace of dimension %d" % (s1['L'].shape[1], m))
    if m == 0:                                              # nothing informed: everything in U, Step 2 skipped
        info.update(step2_tests=[], rotation=dict(sweeps=0))
        return Structure(W_N=np.zeros((d, 0)), W_Q=np.zeros((d, 0)), ell=np.zeros(0), B=np.zeros((0, 0)),
                         H=np.zeros((0, 0)), F_S=F_S, mu_S=np.zeros(0), info=info)
    A_half, A_mhalf, mu_S, A_eig = reference_gaussian(F_S, H, s1['g'], opt.precision_floor)
    w = rng.standard_normal((opt.n2, m))
    xi = rng.standard_normal((opt.n2, d))
    xi -= (xi @ F_S) @ F_S.T
    X2 = (mu_S[None, :] + w @ A_mhalf) @ F_S.T + xi
    y2 = evaluate_many(neg_log_lik, X2, vectorized, map_fn)
    res = y2 - (s1['kappa0'] + X2 @ s1['g'] + _quadratic_part(X2, s1['L'], s1['signs'])[0])
    res = res - res.mean()
    E, tests, M_eig = select_nongaussian(w, res, rng, opt, log)
    W_N, W_Q = back_map(F_S, A_mhalf, E)
    rotation = dict(sweeps=0)
    if opt.rotate_basis and E.shape[1] >= 2:
        E_rot, rotation = rotate_n_basis(w, res, E, opt)
        W_N = closest_basis(W_N, F_S, A_half, E_rot)
    ell, B, H_Q = q_model(s1['g'], H, W_N, W_Q)
    info.update(calls=opt.n1 + opt.n2, step2_tests=tests, M34_eigenvalues=M_eig.tolist(),
                reference_precision_eigenvalues=A_eig.tolist(), mu_S_norm=float(np.linalg.norm(mu_S)),
                rotation=rotation)
    if log is not None:
        log("  learned r = %d non-Gaussian and k = %d quadratic directions" % (W_N.shape[1], W_Q.shape[1]))
    return Structure(W_N=W_N, W_Q=W_Q, ell=ell, B=B, H=H_Q, F_S=F_S, mu_S=mu_S, info=info)
