# nqu-sampler

Gradient-free Markov chain Monte Carlo for Bayesian inverse problems whose posteriors are approximately Gaussian in
most informed directions but strongly non-Gaussian in a few.

The **NQU sampler** learns, from likelihood values only, a split of the parameter space into

* **N** – a few strongly **n**on-Gaussian directions (multiple modes, spikes, skewness),
* **Q** – informed directions in which the negative log-likelihood is approximately **q**uadratic given N,
* **U** – the remaining, approximately **u**ninformative directions,

and then samples with a kernel suited to each block. The learned structure only shapes the proposals: every update is
a Metropolis–Hastings step that leaves the exact posterior invariant and costs one likelihood evaluation. A collapsed
update handles an unknown global prior scale.

This package implements the method of the paper *Gradient-free MCMC exploiting low-dimensional Gaussian and
non-Gaussian posterior structure* (H. Wang, J. Jiang, J. Liu and J. Li). It depends only on NumPy.

## Installation

```bash
git clone <repository-url>
cd nqu-sampler
pip install .
```

Requirements: Python 3.9 or later and NumPy 1.22 or later. To run the tests with pytest, install `pip install .[test]`;
the tests also run without pytest (see [Tests](#tests)).

## Quick start

```python
import numpy as np
import nqu

d = 100
V = np.linalg.qr(np.random.default_rng(0).standard_normal((d, 3)))[0]

def neg_log_lik(x):
    """Phi(x) in whitened coordinates (x ~ N(0, I) a priori); x has shape (d,) or (n, d)."""
    z = np.asarray(x) @ V
    return (5.0 * np.exp(-z[..., 0] ** 2)                 # a bimodal direction
            + 0.5 * 100.0 * (z[..., 1] - 0.5) ** 2        # a stiff Gaussian direction
            + 0.5 * 3.0 * (z[..., 2] + 0.5) ** 2)         # a weakly informed Gaussian direction

sampler = nqu.NQUSampler(neg_log_lik, d, vectorized=True, seed=1)
structure = sampler.learn(n1=20_000, n2=10_000)          # 30 000 likelihood calls
print(structure.summary())

result = sampler.sample(n_calls=200_000, thin=10)        # 200 000 calls, one state recorded every 10
print(result.summary(max_rows=5))
posterior_mean = result.mean()
```

## The setting

The sampler works in whitened coordinates `x` with prior `N(0, I)`. The target is

```
pi(x)  proportional to  exp(-|x|^2 / 2 - Phi(x)),
```

where `Phi` is the negative log-likelihood. For a Gaussian prior `N(0, C0)` on a parameter `theta`, write
`theta = C0^{1/2} x`; the helper `nqu.whiten(neg_log_lik_theta, C0_sqrt)` returns `Phi` as a function of `x`.

**Global prior scale.** For `theta | tau ~ N(0, tau C0)` with a hyperprior `p(tau)`, pass `scale_prior`, for example
`nqu.InverseGamma(3, 2)`. The target is then
`pi(x, tau)` proportional to `p(tau) tau^(-d/2) exp(-|x|^2 / (2 tau) - Phi(x))`, and the recorded samples include
`tau`. Learning uses the design `x ~ N(0, I)`, i.e. `tau = 1`, so it is best to normalise the hyperprior to
`E[tau] = 1`. Any object with a method `logpdf(tau)` can serve as a hyperprior.

**Other non-Gaussian priors** that are the image of `N(0, I)` under a known map `T` (for example a Laplace prior by
componentwise quantile transforms) are handled by sampling `x` with `Phi(T(x))`.

## How the method works

### Learning (`n1 + n2` likelihood calls, done once)

1. **Step 1 – quadratic sensing.** `Phi` is evaluated at `n1` prior draws and a signed low-rank quadratic model
   `q(x) = kappa0 + g^T x + x^T H x / 2` is fitted one rank at a time (principal Hessian directions to initialise a
   rank, scaled gradient descent to refine, a held-out validation set to stop). The informed subspace `S` is spanned by
   the eigenvectors of `g g^T + H^2` with eigenvalues above `delta^2`.
2. **Step 2 – non-Gaussian directions.** `Phi` is evaluated at `n2` points drawn from a reference Gaussian on `S`
   built from the Step 1 model. The third- and fourth-order Hermite coefficient tensors of the residual
   `Phi - q` are estimated on two halves of the design and combined into a cross-fitted matrix `M_34`; directions are
   selected one at a time while the leading eigenvalue exceeds a high quantile of random sign flips of the residual.
   The selected directions are mapped back to `x` (columns of `W_N`); the rest of `S` forms `W_Q`.
3. **Rotation of the N basis.** With two or more N directions, the basis of the N block is rotated (Jacobi sweeps,
   an ICA-type criterion on the same data) so that each coordinate carries one non-Gaussian feature.
4. **Step 3 – Q model.** The Q model `Phi ~ phi_0(a) + (ell + B a)^T c + c^T H c / 2`, with `a = W_N^T x` and
   `c = W_Q^T x`, is read off the Step 1 model. No further calls are needed.

### Sampling (one likelihood call per update)

A random-scan Metropolis-within-Gibbs chain on `(a, u, rho)`, where `u` standardises `c` around the conditional mean of
the Q model and `rho = P_U x`:

* **(N)** a random walk on one coordinate of `a`; the Q block follows its learned conditional mean, so the N block
  does not have to wait for it;
* **(Q)** robust adaptive Metropolis (Vihola, 2012) on the Q block;
* **(U)** with probability 1/2 a refresh of `rho` from its conditional prior, otherwise a pCN step;
* **(H)** (with a global scale) `log(tau)` is updated by slice sampling on its conditional density given `a` under the
  Q model, then `c` and `rho` are redrawn, followed by a Metropolis–Hastings correction with the model residual.

The updates are chosen with probabilities 0.4, 0.3, 0.3 for (N), (Q), (U) (0.36, 0.27, 0.27, 0.10 with (H)). The first
10% of the sampling calls are a warm-up that tunes the (N) step sizes (target acceptance 0.35) and the (U) pCN
parameter (target 0.3); they are then frozen and the warm-up is discarded. RAM adapts throughout.

Errors in the learned structure affect efficiency, never the invariant distribution. For likelihoods with exactly the
learned structure, the spectral gap of the sampler does not depend on the dimension.

## Interface

### `nqu.NQUSampler`

```python
nqu.NQUSampler(neg_log_lik, dim, scale_prior=None, vectorized=False, map_fn=None, seed=None,
               learning_options=None, sampling_options=None, verbose=False)
```

| Argument | Meaning |
|---|---|
| `neg_log_lik` | `Phi(x)`. Called with one point of shape `(d,)` while sampling; if `vectorized=True`, also with arrays of shape `(n, d)` while learning. |
| `dim` | Dimension `d`. |
| `scale_prior` | Hyperprior of a global prior scale (object with `logpdf(tau)`, e.g. `nqu.InverseGamma(3, 2)`); adds the (H) update. |
| `vectorized` | `neg_log_lik` accepts arrays of shape `(n, d)` and returns shape `(n,)`. |
| `map_fn` | A `map`-like function (e.g. `multiprocessing.Pool(...).map`) used to evaluate the learning designs point by point in parallel. |
| `seed` | Integer seed. Learning, the starting point and each chain use separate streams derived from it. |
| `learning_options`, `sampling_options` | `nqu.LearningOptions`, `nqu.SamplingOptions` (defaults: the settings of the paper). |
| `verbose` | Print the progress of the learning stage. |

Methods:

* `learn(n1=None, n2=None)` – run Steps 1–3 with `n1 + n2` calls and return a `Structure` (also stored as
  `sampler.structure`).
* `sample(n_calls=450_000, structure=None, x0=None, tau0=1.0, thin=1, record=None, chain=0, warmup=None)` – run one
  chain with `n_calls` updates (plus one call at the starting point) and return a `SamplingResult`. `record` is an
  optional function of `x` that is stored instead of `x` (useful in high dimension); `chain` selects an independent
  random stream; `structure=nqu.Structure.full_space(d)` gives full-space robust adaptive Metropolis, a useful baseline.
* `start_point(structure=None, chain=0)` – the default starting point.

### `nqu.Structure`

Attributes `W_N` (`d x r`), `W_Q` (`d x k`), `ell`, `B`, `H` (the Q model), `F_S`, `mu_S` (informed subspace and
reference mean of Step 1) and `info` (learning diagnostics: Step 1 history, Step 2 test statistics and thresholds,
calls). Properties `dim`, `r`, `k`. Methods `summary()`, `save(path)`, `Structure.load(path)`,
`Structure.full_space(dim)` and `start_point(rng)`. A structure can also be built by hand, for example from prior
knowledge of the problem.

### `nqu.SamplingResult`

Attributes `samples` (recorded states after the warm-up), `tau` (recorded global scales or `None`), `acceptance`
(acceptance rate of each update), `tuned` (final step sizes), `learning_calls`, `sampling_calls`, `warmup_calls`,
`thin` and `structure`. Methods:

* `mean()`, `std()`;
* `ess(values=None)`, `ess_per_1000_calls(values=None)` – effective sample sizes of the recorded quantities (and
  `tau`) or of any array `values` computed from the samples;
* `efficiency()` – the measure used in the paper: the minimum ESS per 1000 calls over the means and second central
  moments of the recorded quantities, and the calls needed for 1000 effective samples, learning included;
* `summary(names=None, reference=None, max_rows=12)` – a text report.

### Options

`nqu.LearningOptions`: `n1`, `n2`, `validation_fraction`, `stop_tolerance`, `max_rank`, `delta`, `precision_floor`,
`r_max`, `n_flips`, `level`, `rotate_basis`, ... `nqu.SamplingOptions`: `warmup_fraction`, `selection`,
`selection_scale`, target acceptance rates, epoch lengths of the (N) tuning, `precision_clip`, slice sampler settings,
... See the docstrings for every field; the defaults reproduce the paper.

### Diagnostics and helpers

* `nqu.ess(x)`, `nqu.ess_per_1000_calls(x, calls_per_sample=1)`, `nqu.integrated_autocorrelation_time(x)` – Geyer's
  initial monotone sequence estimator, for a series or column by column;
* `nqu.split_rhat(chains)` – split potential scale reduction factor of several chains;
* `nqu.summarize(samples, names=None, reference=None)` – a text table;
* `nqu.whiten(neg_log_lik_theta, prior_sqrt)` – express a likelihood of `theta ~ N(0, C0)` in whitened coordinates;
* `nqu.learn_structure(...)` and `nqu.NQUChain(...)` – the lower-level building blocks used by `NQUSampler`.

## Examples

Both examples have exact reference values, so the output shows whether the estimates are correct. Run them from the
repository root; each takes one to a few minutes.

**`examples/bimodal_coupled.py`** – example M2 of the paper at `d = 100`: a bimodal direction, a spike and ten stiff
Gaussian directions whose conditional means move with the bimodal one. The script learns the structure, samples, and
compares the NQU sampler with full-space robust adaptive Metropolis using the same total number of likelihood calls.
Typical output (abridged):

```
NQU structure: d = 100, r = 2 non-Gaussian, k = 11 quadratic, 87 uninformative directions
NQU sampler: acceptance (N) 0.35, (Q) 0.24, (U) 0.60
  calls needed for 1000 effective samples, learning included: 4.02e+05
  E z1    exact  -0.0723   estimate  -0.0551   z = +0.80
  E z1^2  exact   1.9237   estimate   1.9186   z = -0.37
Full-space RAM (same number of calls): acceptance (Q) 0.24
  calls needed for 1000 effective samples, learning included: 8.69e+06
```

**`examples/heat_global_scale.py`** – example "Heat" of the paper at `d = 100`: recovering the initial condition of the
heat equation from 30 noisy observations, with an unknown prior scale `tau ~ InvGamma(3, 2)`. All directions are
Gaussian given `tau`; the learning stage selects no non-Gaussian direction and `tau` is updated by the collapsed (H)
update. The script compares the posterior of `tau` and of the first coefficients with the exact values. Typical output
(abridged):

```
NQU structure: d = 100, r = 0 non-Gaussian, k = 17 quadratic, 83 uninformative directions
acceptance (Q) 0.24, (U) 0.96, (H) 0.96
tau: exact mean 0.8985, 90% interval 0.474-1.558
     estimate   0.9014 +- 0.0039, 90% interval 0.477-1.561; ESS per 1000 calls 45.6
```

**Validation.** On examples M2 and M4 of the paper (`d = 200`, the paper's settings), this package learns the same
structures as the code used for the paper (differences below `1e-8`) and samples with the same efficiency: over eight
seeds of M2, the median number of calls needed for 1000 effective samples is `2.24 x 10^5`, against `2.25 x 10^5`.

## Practical advice

* **Budget.** The paper uses `n1 = 30 000`, `n2 = 20 000` and `4.5 x 10^5` sampling calls for `d = 200`. Step 1 needs
  enough points to estimate a quadratic model in dimension `d`; if `structure.summary()` reports far fewer informed
  directions than expected, increase `n1`.
* **Expensive forward models.** The learning designs are independent points: pass `vectorized=True` or a parallel
  `map_fn`. The chain itself is sequential.
* **Memory.** Recording every state of a long chain in high dimension is expensive; use `thin` or `record`.
* **Checks.** Look at `structure.summary()`, the acceptance rates (the (N) step sizes target 0.35), the ESS, and run
  several chains with different seeds (`nqu.split_rhat`).
* **Limitations.** Learning assumes that the higher-order part of the likelihood is not dominant along the N and Q
  directions; the Step 2 design is centred at the Step 1 estimate, so non-Gaussian directions can go undetected when
  the posterior is concentrated far from it; the Q model is affine in the N coordinates, so strongly curved coupling is
  followed only approximately. In all these cases the sampler remains exact but becomes less efficient.

## Tests

```bash
python tests/run_tests.py      # without pytest
pytest tests                   # with pytest
```

The tests check the diagnostics on series with known answers, check that the chain leaves the exact posterior invariant
with a correct and with a deliberately wrong Q model, check the global scale against an exact posterior, and check that
learning recovers a bimodal and a spiky direction.

## Citing

If you use this package, please cite the paper (see `CITATION.cff`):

> H. Wang, J. Jiang, J. Liu and J. Li. Gradient-free MCMC exploiting low-dimensional Gaussian and non-Gaussian
> posterior structure. 2026.

## License

MIT, see `LICENSE`.
