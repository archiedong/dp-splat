"""Brief §3.6 acceptance: SVI with |B| = N and rho_t = 1 reproduces full-batch CAVI to
numerical tolerance; plus a convergence smoke test at real minibatch settings."""

import numpy as np
import jax.numpy as jnp
import pytest

from dp_splat import cavi, svi as _svi
from dp_splat.cavi import elbo


def _data(seed=0, N=400):
    rng = np.random.default_rng(seed)
    means_s = np.array([[-4.0, 0.0], [4.0, 0.0], [0.0, 5.0]])
    means_c = np.array([[0.9, 0.1, 0.1], [0.1, 0.9, 0.1], [0.1, 0.1, 0.9]])
    z = rng.integers(0, 3, size=N)
    xs = means_s[z] + 0.5 * rng.normal(size=(N, 2))
    xc = np.clip(means_c[z] + 0.05 * rng.normal(size=(N, 3)), 0, 1)
    return jnp.asarray(xs), jnp.asarray(xc)


@pytest.mark.parametrize("weight_prior,learn_alpha",
                         [("dp", False), ("dp", True), ("sparse_dir", False)])
def test_full_batch_rho1_equals_cavi(weight_prior, learn_alpha):
    xs, xc = _data()
    N = xs.shape[0]
    cfg = cavi.Config(weight_prior=weight_prior, T=9, alpha=1.0, learn_alpha=learn_alpha,
                      e0=0.05, max_iters=1)

    state_c = cavi.init_state(0, xs, xc, cfg)
    state_s = cavi.init_state(0, xs, xc, cfg)
    idx = jnp.arange(N)
    for _ in range(8):
        state_c = cavi.cavi_step(state_c, xs, xc, cfg)
        state_s = _svi.svi_step(state_s, xs, xc, cfg, idx, scale=1.0, rho=1.0)

    np.testing.assert_allclose(np.asarray(state_s.spatial.m),
                               np.asarray(state_c.spatial.m), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(np.asarray(state_s.spatial.Psi),
                               np.asarray(state_c.spatial.Psi), rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(np.asarray(state_s.color.Psi),
                               np.asarray(state_c.color.Psi), rtol=1e-7, atol=1e-9)
    if weight_prior == "dp":
        np.testing.assert_allclose(np.asarray(state_s.weights.gamma1),
                                   np.asarray(state_c.weights.gamma1), rtol=1e-9)
        np.testing.assert_allclose(np.asarray(state_s.weights.gamma2),
                                   np.asarray(state_c.weights.gamma2), rtol=1e-9)
        if learn_alpha:
            np.testing.assert_allclose(float(state_s.weights.w2),
                                       float(state_c.weights.w2), rtol=1e-9)
    else:
        np.testing.assert_allclose(np.asarray(state_s.weights.alpha_post),
                                   np.asarray(state_c.weights.alpha_post), rtol=1e-9)

    L_c = float(elbo(state_c, xs, xc, cfg))
    L_s = float(elbo(state_s, xs, xc, cfg))
    np.testing.assert_allclose(L_s, L_c, rtol=1e-9)


def test_svi_minibatch_reaches_near_cavi_elbo():
    xs, xc = _data(N=2000)
    cfg = cavi.Config(weight_prior="dp", T=9, alpha=1.0, max_iters=100)

    state_c, hist = cavi.fit(0, xs, xc, cfg)
    L_cavi = hist[-1]

    svi_cfg = _svi.SVIConfig(batch_size=256, tau0=64.0, kappa_sched=0.7, n_steps=400)
    state_s, _ = _svi.fit_svi(0, xs, xc, cfg, svi_cfg)
    state_s = cavi.cavi_step(state_s, xs, xc, cfg)  # one full E-step for a valid ELBO
    L_svi = float(elbo(state_s, xs, xc, cfg))

    # SVI is stochastic; require it lands within 2% of the CAVI ELBO on this easy problem
    assert L_svi > L_cavi - 0.02 * abs(L_cavi), f"SVI {L_svi:.2f} vs CAVI {L_cavi:.2f}"
