"""nqu: the NQU sampler.

Gradient-free Markov chain Monte Carlo for Bayesian inverse problems whose posteriors are approximately Gaussian in
most informed directions but strongly non-Gaussian in a few. The sampler learns non-Gaussian (N), approximately
quadratic (Q) and uninformative (U) directions from likelihood values and updates each block with a kernel suited to
it; a collapsed update handles a global prior scale.

Main entry point: :class:`NQUSampler`.
"""
from .chain import NQUChain
from .diagnostics import ess, ess_per_1000_calls, integrated_autocorrelation_time, split_rhat, summarize
from .learning import learn_structure
from .options import LearningOptions, SamplingOptions
from .priors import InverseGamma
from .result import SamplingResult
from .sampler import NQUSampler
from .structure import Structure
from .utils import evaluate_many, whiten

__version__ = "0.1.0"

__all__ = [
    "NQUSampler", "NQUChain", "Structure", "SamplingResult", "LearningOptions", "SamplingOptions", "InverseGamma",
    "learn_structure", "whiten", "evaluate_many", "ess", "ess_per_1000_calls", "integrated_autocorrelation_time",
    "split_rhat", "summarize", "__version__",
]
