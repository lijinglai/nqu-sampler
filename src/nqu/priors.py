"""Hyperpriors for a global prior scale.

With a global scale the prior is ``theta | tau ~ N(0, tau * C0)``, i.e. ``x | tau ~ N(0, tau * I)`` in whitened
coordinates. Any object with a method ``logpdf(tau)`` returning the log density of ``tau`` (up to a constant) can be
used as a scale prior. The learning stage uses the design ``x ~ N(0, I)``, which corresponds to ``tau = 1``; it is
therefore best to normalise the hyperprior so that ``E[tau] = 1``.
"""
import math


class InverseGamma:
    """Inverse gamma distribution with density proportional to ``tau**(-shape - 1) * exp(-scale / tau)``.

    Parameters
    ----------
    shape, scale:
        Positive shape and scale. The mean is ``scale / (shape - 1)`` for ``shape > 1``; ``InverseGamma(3, 2)``, used
        in the paper, has mean 1 and standard deviation 1.
    """

    def __init__(self, shape, scale):
        if shape <= 0 or scale <= 0:
            raise ValueError("shape and scale must be positive")
        self.shape = float(shape)
        self.scale = float(scale)

    def logpdf(self, tau):
        """Log density at ``tau > 0`` (normalised)."""
        if tau <= 0:
            return -math.inf
        a, b = self.shape, self.scale
        return a * math.log(b) - math.lgamma(a) - (a + 1.0) * math.log(tau) - b / tau

    @property
    def mean(self):
        return self.scale / (self.shape - 1.0) if self.shape > 1 else math.inf

    def __repr__(self):
        return "InverseGamma(shape=%g, scale=%g)" % (self.shape, self.scale)
