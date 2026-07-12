"""Full-batch CAVI loop and ELBO for DP-Splat (brief §3.3–§3.5, implemented exactly).

Model state = two NIW posteriors (spatial, color — brief §3.1's independent modalities given a
shared assignment z) + a weight posterior chosen by `weight_prior in {"dp", "sparse_dir", "dir"}`
+ optionally q(alpha) = Gamma(w1, w2) (flag `learn_alpha`, Variant A only).

The CAVI cycle is brief eq. (7): (5) responsibilities -> (1) NIW -> (2) weights -> (6) alpha,
iterated until relative ELBO change < tol. The ELBO (§3.5) is implemented one named function
per term so each expectation can be tested against Monte Carlo (tests/test_elbo_mc.py).

Ground rule 5: full-batch CAVI must be monotone in the ELBO; any violation is a bug, never
something to clip away.
"""

import dataclasses
from typing import NamedTuple, Optional, Union

import numpy as np
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from . import niw as _niw
from . import priors as _pr


@dataclasses.dataclass(frozen=True)
class Config:
    weight_prior: str = "dp"  # "dp" | "sparse_dir" | "dir"
    T: int = 20  # truncation level / number of components
    alpha: float = 1.0  # DP concentration (fixed unless learn_alpha)
    learn_alpha: bool = False
    a0: float = 1.0  # Gamma prior on alpha (brief §3.2 defaults)
    b0: float = 1.0
    e0: float = 1.0  # Dirichlet concentration ("sparse_dir": e0 << 1; VBGS repro: 1/T)
    kappa0: float = 1e-3  # NIW defaults per brief §3.2
    nu0_offset: float = 2.0  # nu0 = D + nu0_offset
    max_iters: int = 200
    tol: float = 1e-6  # relative ELBO change (brief §3.4 item 7)
    # Q5 compat flag: freeze the color NIW's Wishart part (Psi, nu) at the prior, mirroring
    # VBGS's fixed_precision=True (their n/inv_u read-backs are pinned to the prior — see
    # third_party/vbgs/vbgs/vi/conjugate/mvn.py:210-260). kappa/m still update; this is exact
    # coordinate ascent in the restricted family. Used by tests/test_vbgs_regression.py.
    fixed_color_precision: bool = False

    def __post_init__(self):
        if self.weight_prior not in ("dp", "sparse_dir", "dir"):
            raise ValueError(f"unknown weight_prior {self.weight_prior!r}")


Weights = Union[_pr.StickBreakingPosterior, _pr.DirichletPosterior]


class State(NamedTuple):
    spatial: _niw.NIW
    color: _niw.NIW
    spatial_prior: _niw.NIW
    color_prior: _niw.NIW
    weights: Weights
    r: Optional[jnp.ndarray]  # (N, T) responsibilities of the last E-step


# ---------------------------------------------------------------------------
# Initialization (brief §3.4 item 7; all choices logged in docstrings)
# ---------------------------------------------------------------------------


def _kmeanspp_seeds(rng: np.random.Generator, x: np.ndarray, K: int) -> np.ndarray:
    """k-means++ seeding (seeding only, no Lloyd iterations) on the given points."""
    n = x.shape[0]
    seeds = [x[rng.integers(n)]]
    d2 = np.full(n, np.inf)
    for _ in range(K - 1):
        d2 = np.minimum(d2, ((x - seeds[-1]) ** 2).sum(-1))
        p = d2 / d2.sum() if d2.sum() > 0 else np.full(n, 1.0 / n)
        seeds.append(x[rng.choice(n, p=p)])
    return np.stack(seeds)


def default_niw_prior(x: jnp.ndarray, K: int, kappa0: float, nu0_offset: float) -> _niw.NIW:
    """Brief §3.2 defaults: m0 = data centroid, Psi0 = scale^2 * I * (nu0 - D - 1),
    kappa0 weak, nu0 = D + 2 (via nu0_offset)."""
    D = x.shape[1]
    nu0 = D + nu0_offset
    m0 = x.mean(0)
    scale2 = float(jnp.mean(x.var(0)))
    Psi0 = scale2 * jnp.eye(D) * (nu0 - D - 1.0)
    return _niw.make_prior(m0, kappa0, Psi0, nu0, K)


def init_state(seed: int, xs: jnp.ndarray, xc: jnp.ndarray, cfg: Config,
               subsample: int = 4096) -> State:
    """Initialize per brief §3.4 item 7: means by k-means++ on a subsample; NIW posteriors
    start at the prior except for the seeded means; weight posterior starts at its prior
    (N_k = 0). Responsibilities are produced by the first E-step of the loop."""
    rng = np.random.default_rng(seed)
    T = cfg.T
    n = xs.shape[0]
    idx = rng.choice(n, size=min(subsample, n), replace=False)
    joint = np.concatenate([np.asarray(xs)[idx], np.asarray(xc)[idx]], axis=1)
    # whiten each feature before seeding so modalities weigh comparably
    joint = (joint - joint.mean(0)) / (joint.std(0) + 1e-12)
    seeds = _kmeanspp_seeds(rng, joint, T)
    # nearest original points to the whitened seeds -> unwhitened modality means
    d2 = ((joint[None, :, :] - seeds[:, None, :]) ** 2).sum(-1)  # (T, |idx|)
    chosen = idx[np.argmin(d2, axis=1)]

    sp = default_niw_prior(xs, T, cfg.kappa0, cfg.nu0_offset)
    cp = default_niw_prior(xc, T, cfg.kappa0, cfg.nu0_offset)
    spatial = sp._replace(m=jnp.asarray(np.asarray(xs)[chosen]))
    color = cp._replace(m=jnp.asarray(np.asarray(xc)[chosen]))

    if cfg.weight_prior == "dp":
        zeros = jnp.zeros(T)
        g1, g2 = _pr.dp_update(zeros, cfg.alpha)
        w1 = w2 = None
        if cfg.learn_alpha:
            _, elog1mv = _pr.beta_expectations(g1, g2)
            w1, w2 = _pr.alpha_update(elog1mv, cfg.a0, cfg.b0)
        weights: Weights = _pr.StickBreakingPosterior(g1, g2, w1, w2)
    else:
        weights = _pr.DirichletPosterior(_pr.dir_update(jnp.zeros(T), cfg.e0))
    return State(spatial, color, sp, cp, weights, None)


# ---------------------------------------------------------------------------
# E-step and M-step (brief eqs. 5, 1, 2, 6)
# ---------------------------------------------------------------------------


def _e_alpha_pair(state: State, cfg: Config):
    """(E[alpha], E[log alpha]) — from q(alpha) if learned, else the fixed value."""
    w = state.weights
    if cfg.learn_alpha and isinstance(w, _pr.StickBreakingPosterior) and w.w1 is not None:
        return _pr.gamma_expectations(w.w1, w.w2)
    return jnp.asarray(cfg.alpha), jnp.log(jnp.asarray(cfg.alpha))


def elogpi(state: State, cfg: Config) -> jnp.ndarray:
    if cfg.weight_prior == "dp":
        w = state.weights
        return _pr.dp_elogpi(w.gamma1, w.gamma2)
    return _pr.dir_elogpi(state.weights.alpha_post)


def responsibilities(state: State, xs, xc, cfg: Config) -> jnp.ndarray:
    """Brief eq. (5), computed with log-sum-exp."""
    log_rho = (
        elogpi(state, cfg)[None, :]
        + _niw.expected_gauss_loglik(state.spatial, xs)
        + _niw.expected_gauss_loglik(state.color, xc)
    )
    return jnp.exp(log_rho - logsumexp(log_rho, axis=1, keepdims=True))


def cavi_step(state: State, xs, xc, cfg: Config) -> State:
    """One full CAVI cycle in the order of brief eq. (7): (5) -> (1) -> (2) -> (6)."""
    r = responsibilities(state, xs, xc, cfg)  # (5)

    Nk_s, xbar_s, S_s = _niw.soft_stats(xs, r)
    Nk_c, xbar_c, S_c = _niw.soft_stats(xc, r)
    spatial = _niw.posterior_update(state.spatial_prior, Nk_s, xbar_s, S_s)  # (1)
    color = _niw.posterior_update(state.color_prior, Nk_c, xbar_c, S_c)
    if cfg.fixed_color_precision:  # Q5 compat: Wishart part pinned to prior (VBGS delta)
        color = color._replace(Psi=state.color_prior.Psi, nu=state.color_prior.nu)

    Nk = Nk_s  # same responsibilities, same counts
    if cfg.weight_prior == "dp":
        e_alpha, _ = _e_alpha_pair(state, cfg)
        g1, g2 = _pr.dp_update(Nk, e_alpha)  # (2)
        w1 = w2 = None
        if cfg.learn_alpha:
            _, elog1mv = _pr.beta_expectations(g1, g2)
            w1, w2 = _pr.alpha_update(elog1mv, cfg.a0, cfg.b0)  # (6)
        weights: Weights = _pr.StickBreakingPosterior(g1, g2, w1, w2)
    else:
        weights = _pr.DirichletPosterior(_pr.dir_update(Nk, cfg.e0))

    return State(spatial, color, state.spatial_prior, state.color_prior, weights, r)


# ---------------------------------------------------------------------------
# ELBO (brief §3.5) — one named function per term
# ---------------------------------------------------------------------------


def elbo_expected_loglik(state: State, xs, xc) -> jnp.ndarray:
    """E[log p(X | Z, theta)] = sum_{n,k} r_nk sum_m E[log N(x_{n,m} | mu_km, Lambda_km^-1)]."""
    ell = _niw.expected_gauss_loglik(state.spatial, xs) + _niw.expected_gauss_loglik(
        state.color, xc
    )
    return (state.r * ell).sum()


def elbo_log_p_z(state: State, cfg: Config) -> jnp.ndarray:
    """E[log p(Z | v)] = sum_{n,k} r_nk E[log pi_k]."""
    return (state.r * elogpi(state, cfg)[None, :]).sum()


def elbo_log_p_weights(state: State, cfg: Config) -> jnp.ndarray:
    """E[log p(v | alpha)] (Variant A) or E[log p(pi)] (Dirichlet variants)."""
    if cfg.weight_prior == "dp":
        w = state.weights
        e_alpha, e_log_alpha = _e_alpha_pair(state, cfg)
        return _pr.dp_elbo_log_p_v(w.gamma1, w.gamma2, e_alpha, e_log_alpha)
    return _pr.dir_elbo_log_p_pi(state.weights.alpha_post, cfg.e0)


def elbo_log_p_theta(state: State) -> jnp.ndarray:
    """E[log p(theta)] = sum_k sum_m E[log NIW(mu_km, Lambda_km)]."""
    return (
        _niw.expected_log_prior(state.spatial, state.spatial_prior).sum()
        + _niw.expected_log_prior(state.color, state.color_prior).sum()
    )


def elbo_log_p_alpha(state: State, cfg: Config) -> jnp.ndarray:
    """E[log p(alpha)] — present only when alpha is learned (Variant A)."""
    if cfg.weight_prior == "dp" and cfg.learn_alpha:
        w = state.weights
        return _pr.elbo_log_p_alpha(w.w1, w.w2, cfg.a0, cfg.b0)
    return jnp.asarray(0.0)


def elbo_log_q_z(state: State) -> jnp.ndarray:
    """E[log q(Z)] = sum_{n,k} r_nk log r_nk (0 log 0 := 0)."""
    r = state.r
    return jnp.where(r > 0, r * jnp.log(jnp.where(r > 0, r, 1.0)), 0.0).sum()


def elbo_log_q_weights(state: State, cfg: Config) -> jnp.ndarray:
    if cfg.weight_prior == "dp":
        w = state.weights
        return _pr.dp_elbo_log_q_v(w.gamma1, w.gamma2)
    return _pr.dir_elbo_log_q_pi(state.weights.alpha_post)


def elbo_log_q_theta(state: State) -> jnp.ndarray:
    return _niw.expected_log_q(state.spatial).sum() + _niw.expected_log_q(state.color).sum()


def elbo_log_q_alpha(state: State, cfg: Config) -> jnp.ndarray:
    if cfg.weight_prior == "dp" and cfg.learn_alpha:
        w = state.weights
        return _pr.elbo_log_q_alpha(w.w1, w.w2)
    return jnp.asarray(0.0)


def elbo(state: State, xs, xc, cfg: Config) -> jnp.ndarray:
    """Brief §3.5, term by term. Requires state.r from the E-step of the same cycle."""
    if state.r is None:
        raise ValueError("ELBO needs responsibilities; run cavi_step first")
    return (
        elbo_expected_loglik(state, xs, xc)
        + elbo_log_p_z(state, cfg)
        + elbo_log_p_weights(state, cfg)
        + elbo_log_p_theta(state)
        + elbo_log_p_alpha(state, cfg)
        - elbo_log_q_z(state)
        - elbo_log_q_weights(state, cfg)
        - elbo_log_q_theta(state)
        - elbo_log_q_alpha(state, cfg)
    )


# ---------------------------------------------------------------------------
# Fit loop (brief eq. 7)
# ---------------------------------------------------------------------------


def fit(seed: int, xs, xc, cfg: Config, verbose: bool = False):
    """Full-batch CAVI until relative ELBO change < cfg.tol or cfg.max_iters.

    Returns (state, elbo_history). Convergence measures |ΔL| / |L| between cycles.
    """
    xs = jnp.asarray(xs)
    xc = jnp.asarray(xc)
    state = init_state(seed, xs, xc, cfg)
    history = []
    prev = -jnp.inf
    for it in range(cfg.max_iters):
        state = cavi_step(state, xs, xc, cfg)
        L = float(elbo(state, xs, xc, cfg))
        history.append(L)
        if verbose:
            print(f"  iter {it:4d}  elbo {L:.6f}")
        if it > 0 and abs(L - prev) < cfg.tol * abs(prev):
            break
        prev = L
    return state, history
