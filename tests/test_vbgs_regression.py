"""Appendix B regression (with Q4's e0 = 1/K and Q5's fixed_color_precision): starting from
VBGS's OWN initialized model, one step of our CAVI must reproduce one step of VBGS's fit_gmm
to numerical tolerance — the cross-implementation guard for every refactor.

Runs VBGS's actual code (third_party/vbgs, used unmodified per Q1); skipped if absent.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
VBGS = REPO / "third_party" / "vbgs"
if not VBGS.exists():  # pragma: no cover
    pytest.skip("third_party/vbgs not present", allow_module_level=True)
sys.path.insert(0, str(VBGS))
sys.path.insert(0, str(VBGS / "scripts"))

import jax.numpy as jnp
import jax.random as jr

from vbgs.data.image import image_to_data
from vbgs.data.utils import normalize_data
from vbgs.model.train import fit_gmm
from vbgs.model.utils import random_mean_init
from model_image import get_image_model

from dp_splat import cavi, niw, priors as pr

K = 8


def _their_model_and_data(seed=0):
    rng = np.random.default_rng(seed)
    img = jnp.asarray(rng.uniform(size=(16, 16, 3)))
    x, _ = normalize_data(image_to_data(img))
    key = jr.PRNGKey(seed)
    key, sk = jr.split(key)
    mean_init = random_mean_init(sk, x, component_shape=(K,), event_shape=(5, 1),
                                 init_random=True, add_noise=False)
    model = get_image_model(key, n_components=K, mean_init=mean_init, beta=0.0,
                            learning_rate=1.0, dof_offset=1.0, position_scale=None)
    return model, x


def _sq(a):
    """Drop VBGS's trailing event dim: (K, D, 1) -> (K, D), (K,) stays."""
    a = np.asarray(a)
    return a[..., 0] if a.ndim == 3 and a.shape[-1] == 1 else a


def _state_from_vbgs(model):
    """Mirror VBGS's initialized model as a dp_splat State (dir weights, e0 = their prior)."""

    def to_niw(mvn, fixed):
        # posterior read-backs (their properties honor fixed_precision pinning)
        q = niw.NIW(m=jnp.asarray(_sq(mvn.mean)),
                    kappa=jnp.asarray(np.asarray(mvn.kappa).reshape(-1)),
                    Psi=jnp.asarray(np.asarray(mvn.inv_u).reshape(K, *mvn.inv_u.shape[-2:])),
                    nu=jnp.asarray(np.asarray(mvn.n).reshape(-1)))
        p = niw.NIW(m=jnp.asarray(_sq(mvn.prior_mean)) * jnp.ones((K, 1)),
                    kappa=jnp.asarray(np.asarray(mvn.prior_kappa).reshape(-1)) * jnp.ones(K),
                    Psi=jnp.broadcast_to(
                        jnp.asarray(np.asarray(mvn.prior_inv_u).reshape(-1, *mvn.prior_inv_u.shape[-2:]))[0]
                        if np.asarray(mvn.prior_inv_u).reshape(-1, *mvn.prior_inv_u.shape[-2:]).shape[0] == 1
                        else jnp.asarray(np.asarray(mvn.prior_inv_u).reshape(K, *mvn.prior_inv_u.shape[-2:])),
                        (K, *mvn.prior_inv_u.shape[-2:])),
                    nu=jnp.asarray(np.asarray(mvn.prior_n).reshape(-1)) * jnp.ones(K))
        return q, p

    spatial, spatial_prior = to_niw(model.mixture.likelihood, fixed=False)
    color, color_prior = to_niw(model.delta, fixed=True)
    weights = pr.DirichletPosterior(jnp.asarray(np.asarray(model.mixture.prior.alpha).reshape(-1)))
    return cavi.State(spatial, color, spatial_prior, color_prior, weights, None)


def test_one_step_matches_vbgs():
    model, x = _their_model_and_data()
    e0 = float(np.asarray(model.mixture.prior.prior_alpha).reshape(-1)[0])
    np.testing.assert_allclose(e0, 1.0 / K, rtol=1e-12)  # Q4: shipped default is 1/K

    state = _state_from_vbgs(model)
    cfg = cavi.Config(weight_prior="dir", T=K, e0=e0, fixed_color_precision=True)

    xs = jnp.asarray(np.asarray(x)[:, :2])
    xc = jnp.asarray(np.asarray(x)[:, 2:])
    ours = cavi.cavi_step(state, xs, xc, cfg)

    import copy
    theirs = fit_gmm(copy.deepcopy(model), model, x)

    # weights: their alpha = prior + N_k must equal our e0 + N_k
    np.testing.assert_allclose(
        np.asarray(ours.weights.alpha_post),
        np.asarray(theirs.mixture.prior.alpha).reshape(-1), rtol=1e-8, atol=1e-10)

    # spatial NIW posterior
    np.testing.assert_allclose(np.asarray(ours.spatial.m), _sq(theirs.mixture.likelihood.mean),
                               rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(np.asarray(ours.spatial.kappa),
                               np.asarray(theirs.mixture.likelihood.kappa).reshape(-1),
                               rtol=1e-8)
    np.testing.assert_allclose(np.asarray(ours.spatial.nu),
                               np.asarray(theirs.mixture.likelihood.n).reshape(-1), rtol=1e-8)
    np.testing.assert_allclose(
        np.asarray(ours.spatial.Psi),
        np.asarray(theirs.mixture.likelihood.inv_u).reshape(K, 2, 2), rtol=1e-6, atol=1e-8)

    # color: mean part updates, Wishart part pinned (fixed_precision)
    np.testing.assert_allclose(np.asarray(ours.color.m), _sq(theirs.delta.mean),
                               rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(np.asarray(ours.color.kappa),
                               np.asarray(theirs.delta.kappa).reshape(-1), rtol=1e-8)
    np.testing.assert_allclose(np.asarray(ours.color.nu),
                               np.asarray(theirs.delta.n).reshape(-1), rtol=1e-8)
