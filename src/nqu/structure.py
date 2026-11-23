"""The learned block structure: N, Q and U directions and the Q model."""
import json
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Structure:
    """Block structure used by the sampler.

    Coordinates (whitened, ``x ~ N(0, I)`` a priori): ``a = W_N^T x`` (N block), ``c = W_Q^T x`` (Q block) and
    ``rho = P_U x`` with ``P_U = I - W_N W_N^T - W_Q W_Q^T`` (U block). The Q model approximates the negative
    log-likelihood given ``a`` by ``phi_0(a) + (ell + B a)^T c + c^T H c / 2``.

    Attributes
    ----------
    W_N:
        Orthonormal basis of the non-Gaussian directions, shape ``(d, r)``.
    W_Q:
        Orthonormal basis of the approximately quadratic directions, shape ``(d, k)``, orthogonal to ``W_N``.
    ell, B, H:
        Coefficients of the Q model, shapes ``(k,)``, ``(k, r)`` and ``(k, k)``.
    F_S, mu_S:
        Orthonormal basis of the informed subspace found by Step 1 and the mean of the Step 1 reference Gaussian on
        it (``None`` if the structure was not learned). They define the default starting point.
    info:
        Diagnostics of the learning stage (plain Python types).
    """

    W_N: np.ndarray
    W_Q: np.ndarray
    ell: np.ndarray
    B: np.ndarray
    H: np.ndarray
    F_S: np.ndarray = None
    mu_S: np.ndarray = None
    info: dict = field(default_factory=dict)

    def __post_init__(self):
        self.W_N = np.asarray(self.W_N, dtype=float)
        self.W_Q = np.asarray(self.W_Q, dtype=float)
        d = self.W_N.shape[0] if self.W_N.ndim == 2 else self.W_Q.shape[0]
        self.W_N = self.W_N.reshape(d, -1)
        self.W_Q = self.W_Q.reshape(d, -1)
        self.ell = np.asarray(self.ell, dtype=float).reshape(self.k)
        self.B = np.asarray(self.B, dtype=float).reshape(self.k, self.r)
        self.H = np.asarray(self.H, dtype=float).reshape(self.k, self.k)
        if self.F_S is not None:
            self.F_S = np.asarray(self.F_S, dtype=float).reshape(d, -1)
            self.mu_S = np.asarray(self.mu_S, dtype=float).reshape(self.F_S.shape[1])
        blocks = np.column_stack([self.W_N, self.W_Q])
        if blocks.shape[1] and not np.allclose(blocks.T @ blocks, np.eye(blocks.shape[1]), atol=1e-8):
            raise ValueError("the columns of [W_N, W_Q] must be orthonormal")

    @property
    def dim(self):
        """Dimension ``d`` of the parameter."""
        return self.W_N.shape[0]

    @property
    def r(self):
        """Number of non-Gaussian (N) directions."""
        return self.W_N.shape[1]

    @property
    def k(self):
        """Number of approximately quadratic (Q) directions."""
        return self.W_Q.shape[1]

    @classmethod
    def full_space(cls, dim):
        """All directions in the Q block with a flat Q model.

        The (Q) update is then robust adaptive Metropolis on the whole space (with the prior in the acceptance
        ratio), i.e. the NQU sampler without learning; useful as a baseline.
        """
        return cls(W_N=np.zeros((dim, 0)), W_Q=np.eye(dim), ell=np.zeros(dim), B=np.zeros((dim, 0)),
                   H=np.zeros((dim, dim)), info=dict(kind='full space'))

    def start_point(self, rng):
        """Default starting point ``F_S mu_S + P_{S-perp} xi``, ``xi ~ N(0, I)``; a prior draw if not learned."""
        xi = rng.standard_normal(self.dim)
        if self.F_S is None:
            return xi
        return self.F_S @ self.mu_S + xi - self.F_S @ (self.F_S.T @ xi)

    def summary(self):
        """Short human-readable description."""
        lines = ["NQU structure: d = %d, r = %d non-Gaussian, k = %d quadratic, %d uninformative directions"
                 % (self.dim, self.r, self.k, self.dim - self.r - self.k)]
        info = self.info
        if 'step1_rank' in info:
            lines.append("  Step 1: rank %d, informed subspace of dimension %d"
                         % (info['step1_rank'], info['informed_dim']))
        if 'step2_tests' in info:
            for t in info['step2_tests']:
                lines.append("  Step 2, selection %d: statistic %.4g, threshold %.4g -> %s"
                             % (t['step'] + 1, t['statistic'], t['threshold'],
                                'selected' if t['statistic'] > t['threshold'] else 'stop'))
        if 'rotation' in info and info['rotation'].get('sweeps'):
            rot = info['rotation']
            lines.append("  N basis rotated: contrast %.4g -> %.4g in %d sweeps" % (rot['J0'], rot['J1'], rot['sweeps']))
        if self.k:
            ev = np.linalg.eigvalsh(0.5 * (self.H + self.H.T))
            lines.append("  Q model: eigenvalues of H from %.3g to %.3g" % (ev[0], ev[-1]))
        if 'calls' in info:
            lines.append("  learning calls: %d" % info['calls'])
        return "\n".join(lines)

    def save(self, path):
        """Save to a ``.npz`` file."""
        arrays = dict(W_N=self.W_N, W_Q=self.W_Q, ell=self.ell, B=self.B, H=self.H,
                      info=np.array(json.dumps(self.info, default=_to_json)))
        if self.F_S is not None:
            arrays.update(F_S=self.F_S, mu_S=self.mu_S)
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path):
        """Load a structure saved with :meth:`save`."""
        with np.load(path, allow_pickle=False) as f:
            kw = {name: f[name] for name in ('W_N', 'W_Q', 'ell', 'B', 'H')}
            if 'F_S' in f:
                kw.update(F_S=f['F_S'], mu_S=f['mu_S'])
            kw['info'] = json.loads(str(f['info']))
        return cls(**kw)


def _to_json(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError("cannot serialise %r" % type(obj))
