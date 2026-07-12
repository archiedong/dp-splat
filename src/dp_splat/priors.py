"""Weight priors: truncated stick-breaking DP, sparse Dirichlet, symmetric Dirichlet.

Implements brief §3.2 Variants A/B and §3.4 eqs. (2), (3), (6) exactly, plus the weight-prior
ELBO terms of §3.5. The switch is `weight_prior in {"dp", "sparse_dir", "dir"}`; `sparse_dir`
and `dir` share the Dirichlet code path and differ only in e0 (reproducing
VBGS requires e0 = 1/K, their shipped default).

DP truncation convention (brief §3.2): q(v_k) = Beta(gamma_k1, gamma_k2) for k = 1..T-1 and
v_T := 1, so arrays gamma1/gamma2 have length T-1 while E[log pi] has length T.
"""

from typing import NamedTuple, Optional

import jax.numpy as jnp
from jax.scipy.special import betaln, digamma, gammaln


class StickBreakingPosterior(NamedTuple):
    """q(v) Beta parameters, (T-1,) each; and q(alpha) Gamma(w1, w2) if alpha is learned."""

    gamma1: jnp.ndarray
    gamma2: jnp.ndarray
    w1: Optional[jnp.ndarray] = None  # None <=> alpha fixed
    w2: Optional[jnp.ndarray] = None


class DirichletPosterior(NamedTuple):
    """q(pi) = Dir(alpha_post), (T,)."""

    alpha_post: jnp.ndarray


# ---------------------------------------------------------------------------
# Variant A: truncated stick-breaking DP
# ---------------------------------------------------------------------------


def dp_update(Nk: jnp.ndarray, e_alpha) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Brief eq. (2): gamma_k1 = 1 + N_k, gamma_k2 = E[alpha] + sum_{j>k} N_j, for k<T."""
    T = Nk.shape[0]
    tail = jnp.cumsum(Nk[::-1])[::-1]  # tail[k] = sum_{j>=k} N_j
    gamma1 = 1.0 + Nk[: T - 1]
    gamma2 = e_alpha + tail[1:]  # sum_{j=k+1}^{T} N_j
    return gamma1, gamma2


def beta_expectations(gamma1, gamma2):
    """Brief eq. (3) first line: (E[log v_k], E[log(1 - v_k)]), each (T-1,)."""
    dg_sum = digamma(gamma1 + gamma2)
    return digamma(gamma1) - dg_sum, digamma(gamma2) - dg_sum


def dp_elogpi(gamma1, gamma2) -> jnp.ndarray:
    """Brief eq. (3): E[log pi_k] = E[log v_k] + sum_{j<k} E[log(1-v_j)], (T,) with v_T := 1."""
    elogv, elog1mv = beta_expectations(gamma1, gamma2)
    elogv_full = jnp.concatenate([elogv, jnp.zeros(1)])  # E[log v_T] = 0
    prefix = jnp.concatenate([jnp.zeros(1), jnp.cumsum(elog1mv)])  # sum_{j<k}
    return elogv_full + prefix


def dp_expected_pi(gamma1, gamma2) -> jnp.ndarray:
    """Brief §3.7: E[pi_k] = (g1/(g1+g2)) prod_{j<k} (g2/(g1+g2)), (T,) with v_T := 1."""
    frac = gamma1 / (gamma1 + gamma2)
    one_minus = gamma2 / (gamma1 + gamma2)
    frac_full = jnp.concatenate([frac, jnp.ones(1)])
    prefix = jnp.concatenate([jnp.ones(1), jnp.cumprod(one_minus)])
    return frac_full * prefix


def alpha_update(elog1mv: jnp.ndarray, a0, b0):
    """Brief eq. (6): w1 = a0 + T - 1, w2 = b0 - sum_k E[log(1 - v_k)]."""
    T_minus_1 = elog1mv.shape[0]
    w1 = a0 + T_minus_1
    w2 = b0 - elog1mv.sum()
    return jnp.asarray(w1, dtype=elog1mv.dtype), w2


def gamma_expectations(w1, w2):
    """(E[alpha], E[log alpha]) under Gamma(w1, w2) (shape/rate)."""
    return w1 / w2, digamma(w1) - jnp.log(w2)


# --- ELBO pieces (Variant A) ---


def dp_elbo_log_p_v(gamma1, gamma2, e_alpha, e_log_alpha) -> jnp.ndarray:
    """E[log p(v | alpha)] = sum_k ( E[log alpha] + (E[alpha]-1) E[log(1-v_k)] ).

    p(v_k) = Beta(1, alpha) = alpha (1-v_k)^{alpha-1}; expectation over independent
    q(v_k) and q(alpha). For fixed alpha pass e_alpha=alpha, e_log_alpha=log(alpha).
    """
    _, elog1mv = beta_expectations(gamma1, gamma2)
    return (e_log_alpha + (e_alpha - 1.0) * elog1mv).sum()


def dp_elbo_log_q_v(gamma1, gamma2) -> jnp.ndarray:
    """E[log q(v)] = sum_k ( (g1-1)E[log v] + (g2-1)E[log(1-v)] - log B(g1, g2) )."""
    elogv, elog1mv = beta_expectations(gamma1, gamma2)
    return ((gamma1 - 1.0) * elogv + (gamma2 - 1.0) * elog1mv - betaln(gamma1, gamma2)).sum()


def elbo_log_p_alpha(w1, w2, a0, b0) -> jnp.ndarray:
    """E[log p(alpha)] under Gamma(a0, b0) prior (brief §3.2, flag learn_alpha)."""
    e_alpha, e_log_alpha = gamma_expectations(w1, w2)
    return a0 * jnp.log(b0) - gammaln(a0) + (a0 - 1.0) * e_log_alpha - b0 * e_alpha


def elbo_log_q_alpha(w1, w2) -> jnp.ndarray:
    """E[log q(alpha)] = -H[Gamma(w1, w2)]."""
    e_alpha, e_log_alpha = gamma_expectations(w1, w2)
    return w1 * jnp.log(w2) - gammaln(w1) + (w1 - 1.0) * e_log_alpha - w2 * e_alpha


# ---------------------------------------------------------------------------
# Variant B / baseline: (sparse) symmetric Dirichlet — mirrors VBGS
# ---------------------------------------------------------------------------


def dir_update(Nk: jnp.ndarray, e0) -> jnp.ndarray:
    """Standard conjugate Dirichlet update: alpha_post = e0 + N_k, (T,)."""
    return e0 + Nk


def dir_elogpi(alpha_post: jnp.ndarray) -> jnp.ndarray:
    """E[log pi_k] = psi(alpha_k) - psi(sum_j alpha_j)  (brief §3.4 note after eq. 3)."""
    return digamma(alpha_post) - digamma(alpha_post.sum())


def dir_expected_pi(alpha_post: jnp.ndarray) -> jnp.ndarray:
    return alpha_post / alpha_post.sum()


def dir_elbo_log_p_pi(alpha_post: jnp.ndarray, e0) -> jnp.ndarray:
    """E[log p(pi)] under symmetric Dir(e0)."""
    T = alpha_post.shape[0]
    elogpi = dir_elogpi(alpha_post)
    return gammaln(T * e0) - T * gammaln(e0) + (e0 - 1.0) * elogpi.sum()


def dir_elbo_log_q_pi(alpha_post: jnp.ndarray) -> jnp.ndarray:
    """E[log q(pi)] = -H[Dir(alpha_post)]."""
    elogpi = dir_elogpi(alpha_post)
    return (
        gammaln(alpha_post.sum())
        - gammaln(alpha_post).sum()
        + ((alpha_post - 1.0) * elogpi).sum()
    )
