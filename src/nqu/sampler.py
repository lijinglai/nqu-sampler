"""High-level interface: :class:`NQUSampler`."""
from dataclasses import replace

import numpy as np

from .chain import NQUChain
from .learning import learn_structure
from .options import LearningOptions, SamplingOptions
from .result import SamplingResult


class NQUSampler:
    """Gradient-free MCMC for Bayesian inverse problems.

    The target is ``pi(x)`` proportional to ``exp(-|x|^2 / 2 - Phi(x))`` in whitened coordinates (prior
    ``N(0, I)``), or, with a global prior scale ``tau``, ``pi(x, tau)`` proportional to
    ``p(tau) tau^(-d/2) exp(-|x|^2 / (2 tau) - Phi(x))``. The sampler first
    learns which directions are non-Gaussian (N), approximately quadratic (Q) or uninformative (U) from likelihood
    values (:meth:`learn`), and then samples with a kernel suited to each block (:meth:`sample`). The learned
    structure only shapes the proposals, so every update leaves the exact posterior invariant.

    Parameters
    ----------
    neg_log_lik:
        The negative log-likelihood ``Phi(x)``. It is called with one point of shape ``(d,)`` during sampling and, if
        ``vectorized`` is true, with arrays of shape ``(n, d)`` during learning. Use :func:`nqu.whiten` for a prior
        ``N(0, C0)`` on another parameter.
    dim:
        Dimension ``d``.
    scale_prior:
        Optional hyperprior of a global prior scale, e.g. ``nqu.InverseGamma(3, 2)``; adds the (H) update.
    vectorized:
        ``neg_log_lik`` accepts arrays of shape ``(n, d)``.
    map_fn:
        Optional ``map``-like function to evaluate the learning designs in parallel (e.g. ``pool.map``).
    seed:
        Integer seed. Learning, the starting point and each chain use separate random streams derived from it.
    learning_options, sampling_options:
        :class:`~nqu.options.LearningOptions` and :class:`~nqu.options.SamplingOptions`.
    verbose:
        Print progress of the learning stage.

    Examples
    --------
    >>> sampler = nqu.NQUSampler(neg_log_lik, dim=200, seed=1)
    >>> structure = sampler.learn(n1=30_000, n2=20_000)
    >>> result = sampler.sample(n_calls=450_000, thin=10)
    >>> print(result.summary())
    """

    def __init__(self, neg_log_lik, dim, scale_prior=None, vectorized=False, map_fn=None, seed=None,
                 learning_options=None, sampling_options=None, verbose=False):
        self.neg_log_lik = neg_log_lik
        self.dim = int(dim)
        self.scale_prior = scale_prior
        self.vectorized = bool(vectorized)
        self.map_fn = map_fn
        if seed is None:
            seed = int(np.random.SeedSequence().generate_state(1)[0])
        self.seed = int(seed)
        self.learning_options = learning_options or LearningOptions()
        self.sampling_options = sampling_options or SamplingOptions()
        self.verbose = bool(verbose)
        self.structure = None

    def learn(self, n1=None, n2=None):
        """Learn the block structure with ``n1 + n2`` likelihood calls (defaults from ``learning_options``).

        Returns the :class:`~nqu.structure.Structure`, which is also stored in ``self.structure``.
        """
        opt = self.learning_options
        if n1 is not None:
            opt = replace(opt, n1=int(n1))
        if n2 is not None:
            opt = replace(opt, n2=int(n2))
        log = print if self.verbose else None
        self.structure = learn_structure(self.neg_log_lik, self.dim, self.seed, opt, self.vectorized, self.map_fn, log)
        if self.verbose:
            print(self.structure.summary())
        return self.structure

    def start_point(self, structure=None, chain=0):
        """Default starting point of chain number ``chain``: the mean of the Step 1 reference Gaussian on the informed
        subspace plus a prior draw in its complement."""
        st = structure or self.structure
        stream = [self.seed, 11] if chain == 0 else [self.seed, 11, int(chain)]
        return st.start_point(np.random.default_rng(stream))

    def sample(self, n_calls=450_000, structure=None, x0=None, tau0=1.0, thin=1, record=None, chain=0, warmup=None):
        """Run one chain with ``n_calls`` likelihood calls (plus one at the starting point).

        Parameters
        ----------
        n_calls:
            Number of updates, each costing one call. The first ``warmup_fraction`` of them (default 10%) is a
            warm-up that tunes the step sizes and is not recorded.
        structure:
            Structure to use (default: the one learned by :meth:`learn`). ``nqu.Structure.full_space(d)`` gives
            full-space robust adaptive Metropolis.
        x0, tau0:
            Starting point and starting global scale (default: :meth:`start_point` and ``tau0 = 1``).
        thin:
            Record one state every ``thin`` calls.
        record:
            Optional function of ``x`` recorded instead of ``x``.
        chain:
            Chain number; chains with different numbers use independent random streams.
        warmup:
            Number of warm-up calls (overrides ``warmup_fraction``).

        Returns
        -------
        :class:`~nqu.result.SamplingResult`
        """
        st = structure or self.structure
        if st is None:
            raise RuntimeError("call learn() first or pass a structure")
        if st.dim != self.dim:
            raise ValueError("the structure has dimension %d, the sampler %d" % (st.dim, self.dim))
        if x0 is None:
            x0 = self.start_point(st, chain)
        rng = np.random.default_rng([self.seed, 13, int(chain)])
        ch = NQUChain(self.neg_log_lik, st, rng, x0, self.scale_prior, tau0, self.sampling_options)
        samples, taus, warm = ch.run(n_calls, warmup, thin, record)
        learning_calls = int(st.info.get('calls', 0))
        return SamplingResult(samples=samples, tau=taus, structure=st, sampling_calls=ch.calls, warmup_calls=warm,
                              thin=max(1, int(thin)), learning_calls=learning_calls,
                              acceptance=ch.acceptance_rates(),
                              tuned=dict(sigma=ch.sigma.copy(), b_u=ch.b_u, S=ch.S.copy()))
