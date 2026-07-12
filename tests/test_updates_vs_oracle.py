"""Ground rule 4: every update equation vs the brute-force NumPy oracle on tiny problems
(N <= 200, T <= 10). oracle_numpy.py was written independently from the brief's equations
only — it must never import from src/."""

import numpy as np
import jax.numpy as jnp
import pytest

import oracle_numpy as oracle
from dp_splat import niw, priors as pr
from dp_splat.cavi import Config, State, cavi_step, responsibilities


def _random_problem(seed, N=120, T=7, D=3):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(N, D))
    r = np.exp(rng.normal(size=(N, T)))
    r = r / r.sum(1, keepdims=True)
    m0 = rng.normal(size=D)
    kappa0 = 10 ** rng.uniform(-3, 0)
    A = rng.normal(size=(D, D))
    Psi0 = A @ A.T + D * np.eye(D)
    nu0 = D + 2.0 + rng.uniform(0, 3)
    return rng, x, r, m0, kappa0, Psi0, nu0


@pytest.mark.parametrize("seed", range(5))
def test_soft_stats(seed):
    _, x, r, *_ = _random_problem(seed)
    Nk_o, xbar_o, S_o = oracle.soft_stats(x, r)
    Nk_j, xbar_j, S_j = niw.soft_stats(jnp.asarray(x), jnp.asarray(r))
    np.testing.assert_allclose(np.asarray(Nk_j), Nk_o, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(xbar_j), xbar_o, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(np.asarray(S_j), S_o, rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize("seed", range(5))
def test_niw_posterior_update(seed):
    _, x, r, m0, kappa0, Psi0, nu0 = _random_problem(seed)
    Nk, xbar, S = oracle.soft_stats(x, r)
    o = oracle.niw_update(m0, kappa0, Psi0, nu0, Nk, xbar, S)
    T = r.shape[1]
    prior = niw.make_prior(m0, kappa0, Psi0, nu0, T)
    q = niw.posterior_update(prior, jnp.asarray(Nk), jnp.asarray(xbar), jnp.asarray(S))
    np.testing.assert_allclose(np.asarray(q.kappa), o["kappa"], rtol=1e-12)
    np.testing.assert_allclose(np.asarray(q.nu), o["nu"], rtol=1e-12)
    np.testing.assert_allclose(np.asarray(q.m), o["m"], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(np.asarray(q.Psi), o["Psi"], rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize("seed", range(5))
def test_expected_identities_eq4(seed):
    _, x, r, m0, kappa0, Psi0, nu0 = _random_problem(seed)
    Nk, xbar, S = oracle.soft_stats(x, r)
    o = oracle.niw_update(m0, kappa0, Psi0, nu0, Nk, xbar, S)
    q = niw.NIW(jnp.asarray(o["m"]), jnp.asarray(o["kappa"]), jnp.asarray(o["Psi"]),
                jnp.asarray(o["nu"]))
    np.testing.assert_allclose(
        np.asarray(niw.expected_logdet_precision(q)),
        oracle.expected_logdet_precision(o["Psi"], o["nu"]),
        rtol=1e-9,
    )
    np.testing.assert_allclose(
        np.asarray(niw.expected_mahalanobis(q, jnp.asarray(x))),
        oracle.expected_mahalanobis(o["m"], o["kappa"], o["Psi"], o["nu"], x),
        rtol=1e-8, atol=1e-10,
    )
    np.testing.assert_allclose(
        np.asarray(niw.expected_gauss_loglik(q, jnp.asarray(x))),
        oracle.expected_gauss_loglik(o["m"], o["kappa"], o["Psi"], o["nu"], x),
        rtol=1e-8, atol=1e-10,
    )


@pytest.mark.parametrize("seed", range(5))
def test_weight_updates(seed):
    rng = np.random.default_rng(100 + seed)
    T = 8
    Nk = rng.uniform(0, 40, size=T)
    e_alpha = rng.uniform(0.1, 5.0)
    g1_o, g2_o = oracle.dp_update(Nk, e_alpha)
    g1_j, g2_j = pr.dp_update(jnp.asarray(Nk), e_alpha)
    np.testing.assert_allclose(np.asarray(g1_j), g1_o, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(g2_j), g2_o, rtol=1e-12)

    np.testing.assert_allclose(
        np.asarray(pr.dp_elogpi(g1_j, g2_j)), oracle.dp_elogpi(g1_o, g2_o), rtol=1e-10
    )
    np.testing.assert_allclose(
        np.asarray(pr.dp_expected_pi(g1_j, g2_j)), oracle.dp_expected_pi(g1_o, g2_o),
        rtol=1e-10,
    )

    e0 = rng.uniform(0.001, 1.0)
    np.testing.assert_allclose(
        np.asarray(pr.dir_elogpi(pr.dir_update(jnp.asarray(Nk), e0))),
        oracle.dir_elogpi(oracle.dir_update(Nk, e0)),
        rtol=1e-10,
    )

    elogv_o, elog1mv_o = oracle.beta_expectations(g1_o, g2_o)
    elogv_j, elog1mv_j = pr.beta_expectations(g1_j, g2_j)
    np.testing.assert_allclose(np.asarray(elogv_j), elogv_o, rtol=1e-10)
    np.testing.assert_allclose(np.asarray(elog1mv_j), elog1mv_o, rtol=1e-10)

    w1_o, w2_o = oracle.alpha_update(elog1mv_o, 1.0, 1.0)
    w1_j, w2_j = pr.alpha_update(elog1mv_j, 1.0, 1.0)
    np.testing.assert_allclose(float(w1_j), w1_o, rtol=1e-12)
    np.testing.assert_allclose(float(w2_j), w2_o, rtol=1e-12)


@pytest.mark.parametrize("weight_prior", ["dp", "sparse_dir", "dir"])
def test_full_cavi_step_vs_oracle_chain(weight_prior):
    """One full cavi_step reproduced end-to-end with the oracle: (5) -> (1) -> (2)."""
    seed = 42
    rng = np.random.default_rng(seed)
    N, T, Ds, Dc = 150, 6, 2, 3
    xs = rng.normal(size=(N, Ds))
    xc = rng.normal(size=(N, Dc))
    e0 = 0.05 if weight_prior == "sparse_dir" else 1.0
    cfg = Config(weight_prior=weight_prior, T=T, alpha=1.3, e0=e0)

    # build a state with random-but-valid posteriors
    def rand_niw(D):
        m0 = rng.normal(size=D)
        A = rng.normal(size=(D, D))
        Psi0 = A @ A.T + D * np.eye(D)
        prior = niw.make_prior(m0, 0.5, Psi0, D + 3.0, T)
        post = prior._replace(m=jnp.asarray(rng.normal(size=(T, D))))
        return prior, post

    sp, spost = rand_niw(Ds)
    cp, cpost = rand_niw(Dc)
    if weight_prior == "dp":
        g1 = jnp.asarray(rng.uniform(1, 20, size=T - 1))
        g2 = jnp.asarray(rng.uniform(1, 20, size=T - 1))
        weights = pr.StickBreakingPosterior(g1, g2, None, None)
        elogpi_o = oracle.dp_elogpi(np.asarray(g1), np.asarray(g2))
    else:
        ap = jnp.asarray(rng.uniform(0.5, 30, size=T))
        weights = pr.DirichletPosterior(ap)
        elogpi_o = oracle.dir_elogpi(np.asarray(ap))
    state = State(spost, cpost, sp, cp, weights, None)

    # oracle chain
    ell_s = oracle.expected_gauss_loglik(
        np.asarray(spost.m), np.asarray(spost.kappa), np.asarray(spost.Psi),
        np.asarray(spost.nu), xs)
    ell_c = oracle.expected_gauss_loglik(
        np.asarray(cpost.m), np.asarray(cpost.kappa), np.asarray(cpost.Psi),
        np.asarray(cpost.nu), xc)
    r_o = oracle.responsibilities(elogpi_o, [ell_s, ell_c])

    r_j = responsibilities(state, jnp.asarray(xs), jnp.asarray(xc), cfg)
    np.testing.assert_allclose(np.asarray(r_j), r_o, rtol=1e-8, atol=1e-12)

    new = cavi_step(state, jnp.asarray(xs), jnp.asarray(xc), cfg)

    Nk_o, xbar_o, S_o = oracle.soft_stats(xs, r_o)
    o_s = oracle.niw_update(np.asarray(sp.m[0]), float(sp.kappa[0]), np.asarray(sp.Psi[0]),
                            float(sp.nu[0]), Nk_o, xbar_o, S_o)
    np.testing.assert_allclose(np.asarray(new.spatial.m), o_s["m"], rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(np.asarray(new.spatial.Psi), o_s["Psi"], rtol=1e-7, atol=1e-9)

    if weight_prior == "dp":
        g1_o, g2_o = oracle.dp_update(Nk_o, cfg.alpha)
        np.testing.assert_allclose(np.asarray(new.weights.gamma1), g1_o, rtol=1e-10)
        np.testing.assert_allclose(np.asarray(new.weights.gamma2), g2_o, rtol=1e-10)
    else:
        np.testing.assert_allclose(
            np.asarray(new.weights.alpha_post), oracle.dir_update(Nk_o, e0), rtol=1e-10
        )
