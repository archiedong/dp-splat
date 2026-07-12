"""NIW conjugate updates and expected Gaussian log-density identities.

Implements brief §3.4 eqs. (1) and (4) exactly (Bishop PRML §10.2 identities), plus the
NIW expectation terms of the ELBO (§3.5): E_q[log p(mu, Lambda)] and E_q[log q(mu, Lambda)].

Parameterization: we store the Inverse-Wishart scale Psi (brief notation); the Wishart scale
on the precision is W = Psi^{-1} and is never stored — log|W| = -log|Psi| and all quadratic
forms are computed via Cholesky solves against Psi (ridge only inside solves, per brief §10.3).

All functions are vectorized over a leading component axis K and are jit-safe. One modality
per call; the caller (cavi.py) sums over modalities.
"""

from typing import NamedTuple

import jax.numpy as jnp
from jax import vmap
from jax.scipy.linalg import cho_solve, solve_triangular
from jax.scipy.special import digamma, gammaln

# Ridge used ONLY inside Cholesky factorizations (never written back into stored
# parameters, brief §10.3). Scale-aware: eps(dtype) * mean diagonal scale of Psi, so it is
# invisible at working precision yet still guards near-singular Psi_k from tiny N_k.


class NIW(NamedTuple):
    """NIW parameters for K components (or a single prior broadcast to K).

    m:     (K, D)     mean
    kappa: (K,)       mean-precision scale
    Psi:   (K, D, D)  Inverse-Wishart scale
    nu:    (K,)       degrees of freedom
    """

    m: jnp.ndarray
    kappa: jnp.ndarray
    Psi: jnp.ndarray
    nu: jnp.ndarray

    @property
    def dim(self) -> int:
        return self.m.shape[-1]


def make_prior(m0, kappa0, Psi0, nu0, K: int) -> NIW:
    """Broadcast prior hyperparameters to a K-component NIW container.

    A scalar Psi0 means Psi0 * I (a bare broadcast would produce a singular
    constant matrix — rejected by the equation-check review 2026-07-06).
    """
    m0 = jnp.asarray(m0)
    D = m0.shape[-1]
    Psi0 = jnp.asarray(Psi0)
    if Psi0.ndim == 0:
        Psi0 = Psi0 * jnp.eye(D)
    if Psi0.shape[-2:] != (D, D):
        raise ValueError(f"Psi0 must be scalar or (..., {D}, {D}), got {Psi0.shape}")
    return NIW(
        m=jnp.broadcast_to(m0, (K, D)),
        kappa=jnp.full((K,), kappa0),
        Psi=jnp.broadcast_to(Psi0, (K, D, D)),
        nu=jnp.full((K,), nu0),
    )


def soft_stats(x: jnp.ndarray, r: jnp.ndarray):
    """Soft counts and weighted moments (brief §3.4 preamble).

    x: (N, D) data (one modality); r: (N, K) responsibilities.
    Returns Nk (K,), xbar (K, D), S (K, D, D) with S the *centered* weighted scatter.

    The Nk == 0 guard divides by max(Nk, tiny) only; this is exact, not an approximation:
    every use of xbar downstream is multiplied by Nk (eq. 1), so the guarded value never
    contributes when Nk == 0.
    """
    Nk = r.sum(axis=0)  # (K,)
    weighted_sum = r.T @ x  # (K, D)
    xbar = weighted_sum / jnp.maximum(Nk, 1e-32)[:, None]
    # S = sum_n r_nk x x^T - Nk xbar xbar^T  (centered scatter)
    xxT = jnp.einsum("nk,ni,nj->kij", r, x, x)
    S = xxT - Nk[:, None, None] * jnp.einsum("ki,kj->kij", xbar, xbar)
    return Nk, xbar, S


def posterior_update(prior: NIW, Nk, xbar, S) -> NIW:
    """Brief eq. (1) — exact conjugate NIW update, vectorized over K."""
    kappa = prior.kappa + Nk
    m = (prior.kappa[:, None] * prior.m + Nk[:, None] * xbar) / kappa[:, None]
    nu = prior.nu + Nk
    dev = xbar - prior.m  # (K, D)
    coef = prior.kappa * Nk / (prior.kappa + Nk)  # (K,)
    Psi = prior.Psi + S + coef[:, None, None] * jnp.einsum("ki,kj->kij", dev, dev)
    return NIW(m=m, kappa=kappa, Psi=Psi, nu=nu)


def _chol(Psi):
    """Cholesky of Psi with a scale-aware ridge inside the factorization only."""
    D = Psi.shape[-1]
    scale = jnp.trace(Psi, axis1=-2, axis2=-1) / D
    ridge = jnp.finfo(Psi.dtype).eps * scale
    return jnp.linalg.cholesky(Psi + ridge[..., None, None] * jnp.eye(D))


def logdet_psi(q: NIW) -> jnp.ndarray:
    """log|Psi_k|, (K,)."""
    L = _chol(q.Psi)
    return 2.0 * jnp.log(jnp.diagonal(L, axis1=-2, axis2=-1)).sum(-1)


def expected_logdet_precision(q: NIW) -> jnp.ndarray:
    """Brief eq. (4a): E[log|Lambda_k|] = sum_i psi((nu+1-i)/2) + D log 2 + log|W_k|, (K,)."""
    D = q.dim
    i = jnp.arange(1, D + 1)
    dig = digamma((q.nu[:, None] + 1.0 - i[None, :]) / 2.0).sum(-1)
    return dig + D * jnp.log(2.0) - logdet_psi(q)


def expected_mahalanobis(q: NIW, x: jnp.ndarray) -> jnp.ndarray:
    """Brief eq. (4b): E[(x - mu_k)^T Lambda_k (x - mu_k)] for all n, k -> (N, K).

    = D/kappa_k + nu_k (x - m_k)^T Psi_k^{-1} (x - m_k).
    """
    D = q.dim
    L = _chol(q.Psi)  # (K, D, D)

    def per_k(Lk, mk):
        y = solve_triangular(Lk, (x - mk).T, lower=True)  # (D, N)
        return (y**2).sum(0)  # (N,)

    quad = vmap(per_k)(L, q.m)  # (K, N)
    return (D / q.kappa)[None, :] + q.nu[None, :] * quad.T


def expected_gauss_loglik(q: NIW, x: jnp.ndarray) -> jnp.ndarray:
    """Per-modality inner term of brief eq. (5): E[log N(x_n | mu_k, Lambda_k^{-1})] -> (N, K)."""
    D = q.dim
    return (
        0.5 * expected_logdet_precision(q)[None, :]
        - 0.5 * D * jnp.log(2.0 * jnp.pi)
        - 0.5 * expected_mahalanobis(q, x)
    )


def _log_wishart_B(logdet_W, nu, D):
    """log B(W, nu) of the Wishart normalizer (Bishop B.79), given log|W|."""
    i = jnp.arange(1, D + 1)
    mvlgamma = (D * (D - 1) / 4.0) * jnp.log(jnp.pi) + gammaln(
        (nu[..., None] + 1.0 - i) / 2.0
    ).sum(-1)
    return -(nu / 2.0) * logdet_W - (nu * D / 2.0) * jnp.log(2.0) - mvlgamma


def _trace_psi0_wk(prior: NIW, q: NIW) -> jnp.ndarray:
    """tr(Psi0 Psi_k^{-1}) per component, (K,)."""

    def per_k(Lk, Psi0k):
        return jnp.trace(cho_solve((Lk, True), Psi0k))

    return vmap(per_k)(_chol(q.Psi), prior.Psi)


def expected_sigma(q: NIW) -> jnp.ndarray:
    """Posterior expected covariance E[Sigma_k] = Psi_k / (nu_k - D - 1), (K, D, D).
    Used by the §3.7 rendering map (splat shape) — matches VBGS's expected_sigma."""
    return q.Psi / (q.nu - q.dim - 1.0)[:, None, None]


def expected_log_prior(q: NIW, prior: NIW) -> jnp.ndarray:
    """E_q[log p(mu_k, Lambda_k)] under the NIW prior, (K,).  ELBO term E[log p(theta)].

    p(mu | Lambda) = N(m0, (kappa0 Lambda)^{-1});  p(Lambda) = Wishart(W0 = Psi0^{-1}, nu0).
    """
    D = q.dim
    elogdet = expected_logdet_precision(q)
    # E[(m_k - m0)^T Lambda (m_k - m0)] via eq. (4b) at x = m0-row:
    dev = q.m - prior.m  # (K, D)
    L = _chol(q.Psi)

    def per_k(Lk, dk):
        y = solve_triangular(Lk, dk, lower=True)
        return (y**2).sum()

    quad = vmap(per_k)(L, dev)  # (K,)
    e_maha_m0 = D / q.kappa + q.nu * quad

    log_p_mu = (
        0.5 * D * jnp.log(prior.kappa / (2.0 * jnp.pi))
        + 0.5 * elogdet
        - 0.5 * prior.kappa * e_maha_m0
    )
    logdet_W0 = -logdet_psi(prior)
    log_p_lam = (
        _log_wishart_B(logdet_W0, prior.nu, D)
        + 0.5 * (prior.nu - D - 1.0) * elogdet
        - 0.5 * q.nu * _trace_psi0_wk(prior, q)
    )
    return log_p_mu + log_p_lam


def expected_log_q(q: NIW) -> jnp.ndarray:
    """E_q[log q(mu_k, Lambda_k)] (negative NIW entropy), (K,).  ELBO term E[log q(theta)]."""
    D = q.dim
    elogdet = expected_logdet_precision(q)
    log_q_mu = 0.5 * D * jnp.log(q.kappa / (2.0 * jnp.pi)) + 0.5 * elogdet - 0.5 * D
    logdet_Wk = -logdet_psi(q)
    log_q_lam = (
        _log_wishart_B(logdet_Wk, q.nu, D)
        + 0.5 * (q.nu - D - 1.0) * elogdet
        - 0.5 * q.nu * D
    )
    return log_q_mu + log_q_lam
