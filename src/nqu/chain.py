"""The Markov chain of the NQU sampler for a frozen structure.

Coordinates: ``a = W_N^T x`` (N), ``c = W_Q^T x`` (Q), ``rho = P_U x`` (U); with a global scale the state also holds
``tau`` and ``x | tau ~ N(0, tau I)`` a priori. With ``V_tau = (I / tau + H)^{-1}`` and ``mu_tau(a) = -V_tau (ell + B a)``
(``tau = 1`` without a global scale), the updates are

* (N): one coordinate of ``a`` moves by a Gaussian random walk; ``c`` is shifted with the conditional mean of the Q
  model, ``c' = c - V_tau B (a' - a)``, so that ``u = V_tau^{-1/2} (c - mu_tau(a))`` and ``rho`` are unchanged;
* (Q): robust adaptive Metropolis (Vihola 2012) in ``u``: ``c' = c + V_tau^{1/2} S xi``;
* (U): with probability 1/2 a refresh ``rho' ~ N(0, tau P_U)``, otherwise a pCN step towards that distribution;
* (H): ``log(tau)`` by slice sampling on its conditional density given ``a`` under the Q model (no call), then ``c``
  and ``rho`` redrawn from their Q-model conditionals; a Metropolis-Hastings correction with the model residual.

Every update leaves the exact posterior invariant and calls the negative log-likelihood once, at the proposal.
"""
import math

import numpy as np

from .options import SamplingOptions

KINDS = ('N', 'Q', 'U', 'H')


class NQUChain:
    """One chain of the NQU sampler.

    Parameters
    ----------
    neg_log_lik:
        Negative log-likelihood of one point ``x`` (shape ``(d,)``), in whitened coordinates.
    structure:
        :class:`~nqu.structure.Structure`.
    rng:
        ``numpy.random.Generator``.
    x0:
        Starting point.
    scale_prior:
        Optional hyperprior of a global scale (object with ``logpdf(tau)``); enables the (H) update.
    tau0:
        Starting value of the global scale.
    options:
        :class:`~nqu.options.SamplingOptions`.
    """

    def __init__(self, neg_log_lik, structure, rng, x0, scale_prior=None, tau0=1.0, options=None):
        self.f = neg_log_lik
        self.opt = options or SamplingOptions()
        self.rng = rng
        self.scale_prior = scale_prior
        st = structure
        self.d, self.r, self.k = st.dim, st.r, st.k
        self.W_N, self.W_Q = st.W_N, st.W_Q
        self.ell, self.B = st.ell, st.B
        if self.k:
            h, Qh = np.linalg.eigh(0.5 * (st.H + st.H.T))
        else:
            h, Qh = np.zeros(0), np.zeros((0, 0))
        if scale_prior is None:                      # eigenvalues of I + H clipped to precision_clip
            lo, hi = self.opt.precision_clip
            h = np.clip(1.0 + h, lo, hi) - 1.0
        else:                                        # negative eigenvalues set to zero: I / tau + H is positive definite
            h = np.maximum(h, 0.0)
        self.h, self.Qh = h, Qh
        self.H_used = (Qh * h) @ Qh.T
        self.u_dim = self.d - self.r - self.k
        self.x = np.array(x0, dtype=float).reshape(self.d).copy()
        self.phi = float(self.f(self.x))
        if not math.isfinite(self.phi):
            raise ValueError("the negative log-likelihood is not finite at the starting point")
        self.calls = 1
        self._set_tau(float(tau0) if scale_prior is not None else 1.0)
        self.sigma = np.ones(self.r)                 # step sizes of (N)
        self.b_u = 1.0                               # pCN parameter of (U)
        self.S = (2.38 / math.sqrt(self.k)) * np.eye(self.k) if self.k else np.zeros((0, 0))
        self.n_q = 0
        self.accepted = dict.fromkeys(KINDS, 0)
        self.proposed = dict.fromkeys(KINDS, 0)

    # ------------------------------------------------------------------------------------------------ helpers
    def selection_probabilities(self):
        """Probabilities of (N), (Q), (U), (H). With ``r = 0`` the share of (N) is split equally between (Q) and (U);
        updates of empty blocks are removed and the others renormalised."""
        o = self.opt
        p = np.array(list(o.selection) + [0.0] if self.scale_prior is None else list(o.selection_scale), dtype=float)
        if self.r == 0:
            p[1] += 0.5 * p[0]
            p[2] += 0.5 * p[0]
            p[0] = 0.0
        if self.k == 0:
            p[1] = 0.0
        if self.u_dim == 0:
            p[2] = 0.0
        if p.sum() <= 0:
            raise ValueError("no update is available for this structure")
        return p / p.sum()

    def _derived(self, tau):
        dv = 1.0 / (1.0 / tau + self.h)
        return (self.Qh * dv) @ self.Qh.T, (self.Qh * np.sqrt(dv)) @ self.Qh.T

    def _set_tau(self, tau):
        self.tau = tau
        self.V, self.V_half = self._derived(tau)

    def _split(self, x):
        a = self.W_N.T @ x
        c = self.W_Q.T @ x
        return a, c, x - self.W_N @ a - self.W_Q @ c

    def _u_draw(self, tau):
        xi = self.rng.standard_normal(self.d)
        return math.sqrt(tau) * (xi - self.W_N @ (self.W_N.T @ xi) - self.W_Q @ (self.W_Q.T @ xi))

    def _log_phat(self, eta, a):
        """Log density of ``eta = log(tau)`` given ``a`` when the likelihood is replaced by the Q model."""
        tau = math.exp(eta)
        dv = 1.0 / (1.0 / tau + self.h)
        bq = self.Qh.T @ (self.ell + self.B @ a)
        log_zq = 0.5 * (float(np.log(dv).sum()) - self.k * eta) + 0.5 * float((dv * bq * bq).sum())
        return (self.scale_prior.logpdf(tau) + eta - 0.5 * self.r * eta - 0.5 * float(a @ a) / tau + log_zq)

    def _slice(self, eta0, a):
        """One slice-sampling update of ``eta`` (stepping out and shrinkage, Neal 2003)."""
        rng, w, m = self.rng, self.opt.slice_width, self.opt.slice_max_steps
        log_y = self._log_phat(eta0, a) + math.log(rng.random())
        left = eta0 - w * rng.random()
        right = left + w
        j = int(m * rng.random())
        kk = m - 1 - j
        while j > 0 and self._log_phat(left, a) > log_y:
            left -= w
            j -= 1
        while kk > 0 and self._log_phat(right, a) > log_y:
            right += w
            kk -= 1
        while True:
            eta1 = left + (right - left) * rng.random()
            if self._log_phat(eta1, a) > log_y:
                return eta1
            if eta1 < eta0:
                left = eta1
            else:
                right = eta1

    # ------------------------------------------------------------------------------------------------ updates
    def step(self, kind):
        """Apply one update of the given kind (``'N'``, ``'Q'``, ``'U'`` or ``'H'``); returns ``(accepted, extra)``
        where ``extra`` is the moved coordinate for (N) and whether the step counts for tuning ``b_u`` for (U)."""
        rng, x, tau = self.rng, self.x, self.tau
        a, c, rho = self._split(x)
        extra = None
        tau_new = tau
        if kind == 'N':
            j = int(rng.integers(self.r)) if self.r > 1 else 0
            a_new = a.copy()
            a_new[j] += self.sigma[j] * rng.standard_normal()
            c_new = c - self.V @ (self.B @ (a_new - a))
            x_new = self.W_N @ a_new + self.W_Q @ c_new + rho
            phi_new = float(self.f(x_new))
            log_ratio = (-0.5 * (a_new @ a_new + c_new @ c_new) / tau - phi_new
                         + 0.5 * (a @ a + c @ c) / tau + self.phi)
            extra = j
        elif kind == 'Q':
            xi = rng.standard_normal(self.k)
            c_new = c + self.V_half @ (self.S @ xi)
            x_new = x + self.W_Q @ (c_new - c)
            phi_new = float(self.f(x_new))
            log_ratio = -0.5 * (c_new @ c_new) / tau + 0.5 * (c @ c) / tau - phi_new + self.phi
        elif kind == 'U':
            rho_tilde = self._u_draw(tau)
            local = rng.random() < 0.5
            b = self.b_u if local else 1.0
            rho_new = math.sqrt(1.0 - b * b) * rho + b * rho_tilde if b < 1.0 else rho_tilde
            x_new = x - rho + rho_new
            phi_new = float(self.f(x_new))
            log_ratio = self.phi - phi_new
            extra = (b < 1.0) or (self.b_u == 1.0)
        elif kind == 'H':
            tau_new = math.exp(self._slice(math.log(tau), a))
            V_new, V_half_new = self._derived(tau_new)
            b = self.ell + self.B @ a
            c_new = -V_new @ b + V_half_new @ rng.standard_normal(self.k)
            x_new = self.W_N @ a + self.W_Q @ c_new + self._u_draw(tau_new)
            phi_new = float(self.f(x_new))
            r_old = self.phi - float(b @ c) - 0.5 * float(c @ self.H_used @ c)
            r_new = phi_new - float(b @ c_new) - 0.5 * float(c_new @ self.H_used @ c_new)
            log_ratio = r_old - r_new
        else:
            raise ValueError("unknown update %r" % (kind,))
        accepted = (not math.isnan(log_ratio)) and math.log(rng.random()) < log_ratio
        if kind == 'Q':                              # RAM update of the proposal shape S
            self.n_q += 1
            eta = min(1.0, self.k * self.n_q ** (-self.opt.ram_exponent))
            alpha = math.exp(min(0.0, log_ratio)) if not math.isnan(log_ratio) else 0.0
            M = self.S @ (np.eye(self.k) + eta * (alpha - self.opt.ram_target) * np.outer(xi, xi) / float(xi @ xi)) \
                @ self.S.T
            self.S = np.linalg.cholesky(0.5 * (M + M.T))
        if accepted:
            self.x, self.phi = x_new, phi_new
            if kind == 'H':
                self._set_tau(tau_new)
        self.calls += 1
        self.proposed[kind] += 1
        self.accepted[kind] += int(accepted)
        return accepted, extra

    # ------------------------------------------------------------------------------------------------ run
    def run(self, n_calls, warmup=None, thin=1, record=None):
        """Run ``n_calls`` updates. The first ``warmup`` calls (default ``warmup_fraction * n_calls``) tune the (N)
        step sizes and the (U) pCN parameter and are not recorded; afterwards every ``thin``-th state is recorded.

        Parameters
        ----------
        record:
            Optional function of ``x`` whose value is recorded instead of ``x`` (for example a few functionals of a
            high-dimensional parameter).

        Returns
        -------
        samples, taus, warmup:
            Recorded values (one row per recorded state), the recorded global scales (``None`` without a global
            scale) and the number of warm-up calls.
        """
        o, rng = self.opt, self.rng
        n_calls = int(n_calls)
        warmup = int(round(o.warmup_fraction * n_calls)) if warmup is None else int(warmup)
        thin = max(1, int(thin))
        probs = self.selection_probabilities()
        kinds = [kind for kind, p in zip(KINDS, probs) if p > 0]
        p = probs[probs > 0]
        n_keep = max(0, (n_calls - warmup) // thin)
        first = np.asarray(record(self.x) if record is not None else self.x, dtype=float)
        samples = np.empty((n_keep,) + first.shape)
        taus = np.empty(n_keep) if self.scale_prior is not None else None
        i = 0
        epoch_len = next_end = o.epoch_length
        n_prop = 0
        per_coord = {}
        u_acc = u_n = 0
        for t in range(1, n_calls + 1):
            kind = kinds[rng.choice(len(kinds), p=p)]
            accepted, extra = self.step(kind)
            if t <= warmup:
                if kind == 'N':
                    n_prop += 1
                    cnt = per_coord.setdefault(extra, [0, 0])
                    cnt[0] += accepted
                    cnt[1] += 1
                elif kind == 'U' and extra:
                    u_acc += accepted
                    u_n += 1
                    if u_n >= o.u_window:
                        log_b = math.log(self.b_u) + u_acc / u_n - o.u_target
                        self.b_u = math.exp(min(0.0, max(math.log(o.b_min), log_b)))
                        u_acc = u_n = 0
            elif (t - warmup) % thin == 0 and i < n_keep:
                samples[i] = record(self.x) if record is not None else self.x
                if taus is not None:
                    taus[i] = self.tau
                i += 1
            if t == next_end:
                if t <= warmup and n_prop >= o.min_epoch_n:
                    for j, (a_j, n_j) in per_coord.items():
                        if n_j >= o.min_coord_n:
                            self.sigma[j] *= math.exp(a_j / n_j - o.n_target)
                n_prop = 0
                per_coord = {}
                epoch_len = int(math.ceil(epoch_len * o.epoch_growth))
                next_end = t + epoch_len
        return samples[:i], (taus[:i] if taus is not None else None), warmup

    def acceptance_rates(self):
        """Acceptance rate of each update kind that was used."""
        return {kind: self.accepted[kind] / self.proposed[kind] for kind in KINDS if self.proposed[kind]}
