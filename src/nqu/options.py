"""Tuning constants of the NQU sampler.

The defaults are the settings used in the paper. Most users only change the learning budget ``n1`` and ``n2``.
"""
from dataclasses import dataclass


@dataclass
class LearningOptions:
    """Settings of the learning stage (Steps 1-3).

    Attributes
    ----------
    n1, n2:
        Number of likelihood calls of Step 1 (quadratic sensing) and Step 2 (selection of the non-Gaussian block).
    validation_fraction:
        Fraction of the Step 1 design held out to decide when to stop adding ranks.
    stop_tolerance:
        Step 1 stops when a new rank reduces the validation error by less than this fraction (``epsilon_0``).
    max_rank:
        Maximal rank of the quadratic model of Step 1.
    gd_max_iter, gd_tol:
        Maximal number of scaled gradient descent iterations per rank and the relative decrease that ends them.
    delta:
        Directions whose eigenvalue of ``g g^T + H^2`` is below ``delta**2`` are treated as uninformative.
    precision_floor:
        Floor on the eigenvalues of the reference precision ``A_S`` of Step 2.
    r_max:
        Maximal number of non-Gaussian directions.
    n_flips, level:
        Number of random sign flips that approximate the null distribution of the Step 2 statistic, and the level of
        the test (a direction is selected when the statistic exceeds the ``1 - level`` quantile of the flips).
    rotate_basis:
        Rotate the basis of the N block to make its third- and fourth-order Hermite tensors as diagonal as possible
        (uses the Step 2 data only, no extra calls).
    rotation_grid, rotation_max_sweeps, rotation_tol:
        Grid of plane-rotation angles, maximal number of Jacobi sweeps and relative gain that ends them.
    """

    n1: int = 30_000
    n2: int = 20_000
    validation_fraction: float = 0.2
    stop_tolerance: float = 0.02
    max_rank: int = 40
    gd_max_iter: int = 500
    gd_tol: float = 1e-9
    delta: float = 0.1
    precision_floor: float = 0.5
    r_max: int = 5
    n_flips: int = 200
    level: float = 0.01
    rotate_basis: bool = True
    rotation_grid: int = 180
    rotation_max_sweeps: int = 50
    rotation_tol: float = 1e-10


@dataclass
class SamplingOptions:
    """Settings of the sampling stage.

    Attributes
    ----------
    warmup_fraction:
        Fraction of the sampling calls used as warm-up. The step sizes of (N) and the pCN parameter of (U) are tuned
        during the warm-up and then frozen; the warm-up is not recorded.
    selection:
        Selection probabilities of the (N), (Q) and (U) updates without a global scale; with ``r = 0`` they become
        ``(0, 0.5, 0.5)``.
    selection_scale:
        Selection probabilities of (N), (Q), (U) and (H) with a global scale; with ``r = 0`` they become
        ``(0, 0.45, 0.45, 0.10)``.
    n_target, u_target, ram_target:
        Target acceptance rates of (N), of the pCN half of (U), and of the robust adaptive Metropolis (RAM) step (Q).
    ram_exponent:
        The RAM step size is ``min(1, k * n**(-ram_exponent))`` after ``n`` (Q) updates.
    epoch_length, epoch_growth, min_epoch_n, min_coord_n:
        The (N) step sizes are adjusted at the ends of epochs whose length starts at ``epoch_length`` calls and grows
        by the factor ``epoch_growth``; an adjustment needs at least ``min_epoch_n`` (N) proposals in the epoch and
        ``min_coord_n`` proposals of the coordinate.
    u_window, b_min:
        The pCN parameter of (U) is adjusted every ``u_window`` counted (U) proposals and kept in ``[b_min, 1]``.
    precision_clip:
        Without a global scale, the eigenvalues of ``I + H`` of the Q model are clipped to this interval. With a global
        scale, the negative eigenvalues of ``H`` are set to zero instead.
    slice_width, slice_max_steps:
        Initial width and maximal number of stepping-out steps of the slice sampler on ``log(tau)`` in (H).
    """

    warmup_fraction: float = 0.1
    selection: tuple = (0.4, 0.3, 0.3)
    selection_scale: tuple = (0.36, 0.27, 0.27, 0.10)
    n_target: float = 0.35
    u_target: float = 0.3
    ram_target: float = 0.234
    ram_exponent: float = 2.0 / 3.0
    epoch_length: int = 100
    epoch_growth: float = 1.15
    min_epoch_n: int = 20
    min_coord_n: int = 10
    u_window: int = 100
    b_min: float = 1e-3
    precision_clip: tuple = (0.5, 1e6)
    slice_width: float = 1.0
    slice_max_steps: int = 50
