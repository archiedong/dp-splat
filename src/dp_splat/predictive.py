"""Posterior predictive under the mean-field posterior: mixture-of-Students density,
held-out log-likelihood (F3), and conditional color moments Var[c | s] (F5).

Everything here is the standard NIW posterior-predictive / mixture-conditioning algebra;
the complete derivation is given in the paper's appendix, and every formula is verified
against Monte Carlo / numerical integration in tests/test_predictive.py. Summary:

  Per component k and modality m, integrating (mu, Sigma) over q = NIW(m_k, kappa_k, Psi_k,
  nu_k) gives the multivariate Student-t
      p(x* | k) = St(x*; m_k, S_k, eta_k),   S_k = Psi_k (kappa_k + 1) / (kappa_k eta_k),
      eta_k = nu_k - D + 1,
  and by mean-field independence the exact predictive is the mixture
      p_hat(x*) = sum_k E[pi_k] * St_s(x*_s | k) * St_c(x*_c | k).
  Conditioning the mixture on the spatial coordinate s*:
      w_k(s*) proportional to E[pi_k] St_s(s* | k),
      E[c | s*]   = sum_k w_k m_{k,c},
      Cov[c | s*] = sum_k w_k (C_k + m_{k,c} m_{k,c}^T) - E[c|s*] E[c|s*]^T,
  with C_k = S_{k,c} * eta_{k,c} / (eta_{k,c} - 2) the Student-t covariance (eta > 2).
"""

from typing import NamedTuple

import jax.numpy as jnp
from jax import vmap
from jax.scipy.linalg import solve_triangular
from jax.scipy.special import gammaln

from . import niw as _niw
from .cavi import Config, State
from .prune import expected_pi


class StudentT(NamedTuple):
    """Multivariate Student-t parameters per component: loc (K,D), scale (K,D,D), dof (K,)."""

    loc: jnp.ndarray
    scale: jnp.ndarray
    dof: jnp.ndarray


def niw_predictive(q: _niw.NIW) -> StudentT:
    """NIW posterior predictive: St(m, Psi (kappa+1)/(kappa (nu-D+1)), nu-D+1)."""
    D = q.dim
    eta = q.nu - D + 1.0
    coef = (q.kappa + 1.0) / (q.kappa * eta)
    return StudentT(loc=q.m, scale=coef[:, None, None] * q.Psi, dof=eta)


def student_logpdf(st: StudentT, x: jnp.ndarray) -> jnp.ndarray:
    """log St(x; loc, scale, dof) for all points/components -> (N, K)."""
    D = st.loc.shape[-1]
    L = _niw._chol(st.scale)  # scale-aware-ridge Cholesky, same policy as niw.py (§10.3)
    logdet = 2.0 * jnp.log(jnp.diagonal(L, axis1=-2, axis2=-1)).sum(-1)  # (K,)

    def per_k(Lk, mk):
        y = solve_triangular(Lk, (x - mk).T, lower=True)
        return (y**2).sum(0)  # (N,)

    quad = vmap(per_k)(L, st.loc).T  # (N, K)
    eta = st.dof
    return (
        gammaln((eta + D) / 2.0)
        - gammaln(eta / 2.0)
        - 0.5 * D * jnp.log(eta * jnp.pi)
        - 0.5 * logdet
        - 0.5 * (eta + D) * jnp.log1p(quad / eta)
    )


def heldout_loglik(state: State, cfg: Config, xs: jnp.ndarray, xc: jnp.ndarray) -> jnp.ndarray:
    """Mean per-point held-out log predictive density log p_hat(x*) (F3 metric)."""
    from jax.scipy.special import logsumexp

    logw = jnp.log(expected_pi(state, cfg) + 1e-300)
    ll = (
        logw[None, :]
        + student_logpdf(niw_predictive(state.spatial), xs)
        + student_logpdf(niw_predictive(state.color), xc)
    )
    return logsumexp(ll, axis=1).mean()


def conditional_color_moments(state: State, cfg: Config, xs: jnp.ndarray):
    """E[c | s*] and Cov[c | s*] at spatial query points xs (N, Ds).

    Returns (mean (N, Dc), cov (N, Dc, Dc)). Requires eta_c > 2 for finite covariance —
    under the default prior nu0 = D + 2 this always holds (eta_c = nu0 + N_k − Dc + 1 =
    N_k + 3 > 2 for every N_k >= 0); a runtime guard covers non-default nu0_offset.
    """
    from jax.scipy.special import logsumexp

    st_s = niw_predictive(state.spatial)
    st_c = niw_predictive(state.color)
    if bool(jnp.any(st_c.dof <= 2.0)):
        raise ValueError(
            "color predictive dof <= 2 (nu0_offset too small?): Student-t covariance "
            "undefined — Cov[c|s] would be negative/infinite"
        )
    Dc = state.color.dim

    logw = jnp.log(expected_pi(state, cfg) + 1e-300)[None, :] + student_logpdf(st_s, xs)
    w = jnp.exp(logw - logsumexp(logw, axis=1, keepdims=True))  # (N, K)

    mkc = st_c.loc  # (K, Dc)
    Ck = st_c.scale * (st_c.dof / (st_c.dof - 2.0))[:, None, None]  # (K, Dc, Dc)

    mean = w @ mkc  # (N, Dc)
    # law-of-total-variance form: sum_k w_k C_k + sum_k w_k (m_kc - mean)(m_kc - mean)^T.
    # Algebraically identical to second-moment-minus-outer-product but PSD by construction
    # (the subtraction form can lose PSD in floating point when between-component
    # spread dominates).
    within = jnp.einsum("nk,kij->nij", w, Ck)
    dev = mkc[None, :, :] - mean[:, None, :]  # (N, K, Dc)
    between = jnp.einsum("nk,nki,nkj->nij", w, dev, dev)
    return mean, within + between


def predictive_color_variance(state: State, cfg: Config, xs: jnp.ndarray) -> jnp.ndarray:
    """Total predictive color variance tr(Cov[c | s*]) per query point, (N,) — F5 statistic."""
    _, cov = conditional_color_moments(state, cfg, xs)
    return jnp.trace(cov, axis1=-2, axis2=-1)
