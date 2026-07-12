"""Natural-gradient SVI variant (brief §3.6; Hoffman et al. 2013).

Global parameters lambda in {stick-breaking gammas, NIW naturals, (w1, w2)} follow
natural-gradient steps: given minibatch B, intermediate estimates lambda_hat use sufficient
statistics scaled by N/|B|, then

    lambda_t = (1 - rho_t) lambda_{t-1} + rho_t lambda_hat,   rho_t = (t + tau0)^(-kappa_sched).

The NIW blend happens in natural-parameter space (convert, blend, convert back), as the brief
requires. NIW naturals (linear in the sufficient statistics (N_k, sum r x, sum r xx^T)):

    n1 = kappa,  n2 = kappa m,  n3 = nu,  n4 = Psi + kappa m m^T,

with the conjugate update n1 = kappa0 + N_k, n2 = kappa0 m0 + sum r x, n3 = nu0 + N_k,
n4 = Psi0 + kappa0 m0 m0^T + sum r x x^T (algebraically identical to brief eq. (1) — the
equivalence is exercised by the rho=1, |B|=N test against cavi_step).

Acceptance (brief §3.6): SVI with |B| = N and rho_t = 1 must reproduce full-batch CAVI to
numerical tolerance — tests/test_svi.py.

State semantics after svi_step: the returned State.r holds the minibatch responsibilities
SCALED by N/|B|, so prune.soft_counts(state) is an unbiased estimate of the full-data
N_k = sum_n r_nk and effective_k / to_render_params use the brief's §3.7 semantics directly
(unscaled r would make K_hat batch-scale). The ELBO still requires a full
E-step (cavi_step) first, as before.

Numerical note: niw_from_natural recovers Psi by subtracting the rank-one kappa*m*m^T term —
catastrophically cancellative in float32 for uncentered data. Run SVI in float64 (all tests
and experiments do) or on centered/normalized coordinates; see LIMITATIONS.md N6.
"""

import dataclasses

import numpy as np
import jax.numpy as jnp

from . import niw as _niw
from . import priors as _pr
from .cavi import Config, State, init_state, responsibilities


@dataclasses.dataclass(frozen=True)
class SVIConfig:
    batch_size: int = 2**16  # brief §3.6 default
    tau0: float = 64.0
    kappa_sched: float = 0.7  # in (0.5, 1]
    n_steps: int = 200


def niw_to_natural(q: _niw.NIW):
    n1 = q.kappa
    n2 = q.kappa[:, None] * q.m
    n3 = q.nu
    n4 = q.Psi + q.kappa[:, None, None] * jnp.einsum("ki,kj->kij", q.m, q.m)
    return n1, n2, n3, n4


def niw_from_natural(n1, n2, n3, n4) -> _niw.NIW:
    m = n2 / n1[:, None]
    Psi = n4 - n1[:, None, None] * jnp.einsum("ki,kj->kij", m, m)
    return _niw.NIW(m=m, kappa=n1, Psi=Psi, nu=n3)


def _niw_intermediate(prior: _niw.NIW, Nk, sum_x, sum_xxT):
    """lambda_hat for the NIW: prior naturals + (scaled) sufficient statistics."""
    p1, p2, p3, p4 = niw_to_natural(prior)
    return p1 + Nk, p2 + sum_x, p3 + Nk, p4 + sum_xxT


def _blend_niw(q: _niw.NIW, hat, rho) -> _niw.NIW:
    cur = niw_to_natural(q)
    return niw_from_natural(*[(1.0 - rho) * c + rho * h for c, h in zip(cur, hat)])


def svi_step(state: State, xs, xc, cfg: Config, idx, scale: float, rho: float) -> State:
    """One natural-gradient step on minibatch xs[idx], xc[idx] with statistics scale N/|B|."""
    xs_b, xc_b = xs[idx], xc[idx]
    r = responsibilities(state, xs_b, xc_b, cfg)  # local step (eq. 5 on the batch)

    Nk = scale * r.sum(0)
    new_niw = {}
    for name, x_b, q, prior in (
        ("spatial", xs_b, state.spatial, state.spatial_prior),
        ("color", xc_b, state.color, state.color_prior),
    ):
        sum_x = scale * (r.T @ x_b)
        sum_xxT = scale * jnp.einsum("nk,ni,nj->kij", r, x_b, x_b)
        hat = _niw_intermediate(prior, Nk, sum_x, sum_xxT)
        new_niw[name] = _blend_niw(q, hat, rho)

    if cfg.weight_prior == "dp":
        w = state.weights
        if cfg.learn_alpha and w.w1 is not None:
            e_alpha, _ = _pr.gamma_expectations(w.w1, w.w2)
        else:
            e_alpha = jnp.asarray(cfg.alpha)
        g1_hat, g2_hat = _pr.dp_update(Nk, e_alpha)
        g1 = (1.0 - rho) * w.gamma1 + rho * g1_hat
        g2 = (1.0 - rho) * w.gamma2 + rho * g2_hat
        w1 = w2 = None
        if cfg.learn_alpha:
            _, elog1mv = _pr.beta_expectations(g1, g2)
            w1_hat, w2_hat = _pr.alpha_update(elog1mv, cfg.a0, cfg.b0)
            w1_old = w.w1 if w.w1 is not None else w1_hat
            w2_old = w.w2 if w.w2 is not None else w2_hat
            w1 = (1.0 - rho) * w1_old + rho * w1_hat
            w2 = (1.0 - rho) * w2_old + rho * w2_hat
        weights = _pr.StickBreakingPosterior(g1, g2, w1, w2)
    else:
        hat = _pr.dir_update(Nk, cfg.e0)
        weights = _pr.DirichletPosterior(
            (1.0 - rho) * state.weights.alpha_post + rho * hat
        )

    return State(new_niw["spatial"], new_niw["color"], state.spatial_prior,
                 state.color_prior, weights, scale * r)


def fit_svi(seed: int, xs, xc, cfg: Config, svi: SVIConfig, verbose: bool = False):
    """SVI loop. Returns (state, []) — ELBO tracking on full data is the caller's choice
    (it costs a full E-step; experiments evaluate it periodically)."""
    xs = jnp.asarray(xs)
    xc = jnp.asarray(xc)
    N = xs.shape[0]
    B = min(svi.batch_size, N)
    scale = N / B
    rng = np.random.default_rng(seed)
    state = init_state(seed, xs, xc, cfg)
    for t in range(svi.n_steps):
        rho = (t + 1 + svi.tau0) ** (-svi.kappa_sched)
        idx = jnp.asarray(rng.choice(N, size=B, replace=False))
        state = svi_step(state, xs, xc, cfg, idx, scale, rho)
        if verbose and t % 20 == 0:
            print(f"  svi step {t}, rho={rho:.4f}")
    return state, []
