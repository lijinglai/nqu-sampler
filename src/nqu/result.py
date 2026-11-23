"""Output of a sampling run."""
from dataclasses import dataclass, field

import numpy as np

from . import diagnostics


@dataclass
class SamplingResult:
    """Samples and bookkeeping of one chain.

    Attributes
    ----------
    samples:
        Recorded states (or recorded functionals), one row per recorded state, after the warm-up.
    tau:
        Recorded global scales (``None`` without a global scale).
    structure:
        The :class:`~nqu.structure.Structure` used.
    sampling_calls:
        Likelihood calls of the sampling stage (one per update plus one at the starting point).
    warmup_calls:
        Calls of the warm-up (not recorded).
    thin:
        One state is recorded every ``thin`` calls after the warm-up.
    learning_calls:
        Likelihood calls of the learning stage.
    acceptance:
        Acceptance rate of each update kind.
    tuned:
        Final (N) step sizes ``sigma``, (U) pCN parameter ``b_u`` and RAM factor ``S``.
    """

    samples: np.ndarray
    tau: np.ndarray
    structure: object
    sampling_calls: int
    warmup_calls: int
    thin: int
    learning_calls: int
    acceptance: dict
    tuned: dict = field(default_factory=dict)

    @property
    def total_calls(self):
        """All likelihood calls, learning included."""
        return self.learning_calls + self.sampling_calls

    @property
    def calls_before_window(self):
        """Calls spent before the first recorded state: learning, starting point and warm-up."""
        return self.learning_calls + 1 + self.warmup_calls

    def _values(self, include_tau=True):
        X = np.asarray(self.samples, dtype=float).reshape(len(self.samples), -1)
        if include_tau and self.tau is not None:
            X = np.column_stack([X, self.tau])
        return X

    def mean(self):
        """Posterior mean estimate of the recorded quantities."""
        return np.asarray(self.samples).mean(0)

    def std(self):
        """Posterior standard deviation estimate of the recorded quantities."""
        return np.asarray(self.samples).std(0)

    def ess(self, values=None):
        """Effective sample size of each column of ``values`` (default: the recorded quantities and ``tau``)."""
        X = self._values() if values is None else np.asarray(values, dtype=float)
        return diagnostics.ess(X)

    def ess_per_1000_calls(self, values=None):
        """ESS per 1000 likelihood calls of the recorded window."""
        X = self._values() if values is None else np.asarray(values, dtype=float)
        return diagnostics.ess_per_1000_calls(X, self.thin)

    def efficiency(self):
        """The efficiency measure of the paper: the minimum, over the means and second central moments of all recorded
        quantities (and ``tau``), of the ESS per 1000 calls, and the calls needed for 1000 effective samples (calls
        before the recorded window plus ``10**6`` divided by that minimum). Returns ``(min_ess_per_1000, calls)``."""
        X = self._values()
        F = np.column_stack([X, (X - X.mean(0)) ** 2])
        e = diagnostics.ess_per_1000_calls(F, self.thin)
        worst = float(np.nanmin(e))
        return worst, self.calls_before_window + 1e6 / worst

    def summary(self, names=None, reference=None, max_rows=12):
        """Text summary: calls, acceptance rates and a table of the recorded quantities."""
        lines = ["calls: %d learning + %d sampling (%d warm-up); %d states recorded, thin %d"
                 % (self.learning_calls, self.sampling_calls, self.warmup_calls, len(self.samples), self.thin),
                 "acceptance: " + ", ".join("(%s) %.2f" % kv for kv in self.acceptance.items())]
        X, nm = self._values(), names
        if nm is not None and self.tau is not None:
            nm = list(nm) + ['tau']
        ref = None
        if reference is not None:
            ref = np.asarray(reference, dtype=float)
        lines.append(diagnostics.summarize(X, nm, ref, self.thin, max_rows))
        worst, calls = self.efficiency()
        lines.append("minimum ESS per 1000 calls (means and second moments): %.4g; calls for 1000 ESS: %.4g"
                     % (worst, calls))
        return "\n".join(lines)
