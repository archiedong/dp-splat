"""Rendering-parameter map (brief §3.7): posterior -> (mu, cov, alpha) for the VBGS renderer.

Maps E_q[pi_k] and posterior expected covariances to splat weight/shape exactly as the VBGS
codebase does (CODEMAP: model.denormalize builds joint mean [spatial | color] and blockdiag
covariance; render_img weights components by alpha and renders E[color | uv]). Components with
N_k <= n_min are dropped by zeroing their weight (brief §3.7).

This module contains only the parameter mapping; feeding the result to
third_party/vbgs's render_img (pure JAX, 2D) is done by the caller, keeping src/ free of
vbgs imports (VBGS stays an unmodified external dependency).
"""

import jax.numpy as jnp

from . import niw as _niw
from .cavi import Config, State
from .prune import expected_pi, soft_counts


def to_render_params(state: State, cfg: Config, data_params=None, n_min: float = 1.0):
    """Returns (mu (T, Ds+Dc), cov (T, Ds+Dc, Ds+Dc), alpha (T,)).

    data_params: optional VBGS normalize_data dict {"offset", "stdevs"} over the joint
    [spatial | color] dimensions — if given, parameters are mapped back to data units
    (mu -> mu * std + offset, cov -> cov ∘ outer(std, std)) so they can be fed straight
    to vbgs.render.image.render_img, which works in pixel coordinates.
    """
    T = state.spatial.m.shape[0]
    Ds, Dc = state.spatial.dim, state.color.dim
    mu = jnp.concatenate([state.spatial.m, state.color.m], axis=1)
    cov = jnp.zeros((T, Ds + Dc, Ds + Dc))
    cov = cov.at[:, :Ds, :Ds].set(_niw.expected_sigma(state.spatial))
    cov = cov.at[:, Ds:, Ds:].set(_niw.expected_sigma(state.color))

    alpha = expected_pi(state, cfg) * (soft_counts(state) > n_min)

    if data_params is not None:
        std = jnp.asarray(data_params["stdevs"]).reshape(-1)
        off = jnp.asarray(data_params["offset"]).reshape(-1)
        mu = mu * std[None, :] + off[None, :]
        cov = cov * (std[:, None] * std[None, :])[None, :, :]
    return mu, cov, alpha
