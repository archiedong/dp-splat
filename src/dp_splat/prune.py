"""Effective-K estimators, pruning thresholds, posterior expected weights (brief §3.7)."""

import jax.numpy as jnp

from . import priors as _pr
from .cavi import Config, State


def soft_counts(state: State) -> jnp.ndarray:
    """N_k = sum_n r_nk from the last E-step."""
    if state.r is None:
        raise ValueError("state has no responsibilities; run cavi_step first")
    return state.r.sum(axis=0)


def expected_pi(state: State, cfg: Config) -> jnp.ndarray:
    """Posterior expected mixture weights (brief §3.7 first bullet)."""
    if cfg.weight_prior == "dp":
        w = state.weights
        return _pr.dp_expected_pi(w.gamma1, w.gamma2)
    return _pr.dir_expected_pi(state.weights.alpha_post)


def effective_k(state: State, n_min: float = 1.0) -> int:
    """K_hat = #{k : N_k > n_min} (brief §3.7; default n_min = 1; report sensitivity
    to n_min in {0.5, 1, 2, 5} in experiments)."""
    return int((soft_counts(state) > n_min).sum())


def entropy_effective_k(state: State, cfg: Config) -> float:
    """exp(-sum pi~_k log pi~_k) with pi~ the normalized posterior expected weights."""
    p = expected_pi(state, cfg)
    p = p / p.sum()
    h = -jnp.where(p > 0, p * jnp.log(jnp.where(p > 0, p, 1.0)), 0.0).sum()
    return float(jnp.exp(h))


# Var[c | s] lives in dp_splat.predictive (mixture-of-Students conditioning; see the
# paper's appendix for the derivation).
