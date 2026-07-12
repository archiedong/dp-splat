"""MC / numerical verification of the mixture-of-Students predictive (paper Eq. 7).

(1) per-component Student-t density == E_q[N(x | mu, Sigma)] (MC over q);
(2) mixture predictive p_hat(x) == E_q[mixture density] (MC);
(3) end-to-end: conditional color moments at s* == empirical moments of samples from the
    predictive joint restricted to a small spatial ball around s*.
"""

import numpy as np
import jax.numpy as jnp
from scipy.stats import wishart

from dp_splat import cavi, predictive as pred
from dp_splat.prune import expected_pi

S = 30_000


def _fitted_state(seed=0, N=60, T=4):
    rng = np.random.default_rng(seed)
    xs = rng.normal(size=(N, 2)) + rng.integers(0, 2, size=(N, 1)) * 3.0
    xc = np.clip(rng.uniform(0.2, 0.8, size=(N, 3)) + 0.1 * rng.normal(size=(N, 3)), 0, 1)
    cfg = cavi.Config(weight_prior="dp", T=T, alpha=1.0, max_iters=5)
    state = cavi.init_state(seed, jnp.asarray(xs), jnp.asarray(xc), cfg)
    for _ in range(5):
        state = cavi.cavi_step(state, jnp.asarray(xs), jnp.asarray(xc), cfg)
    return state, cfg


def _sample_niw(rng, q, k, n):
    D = q.dim
    W = np.linalg.inv(np.asarray(q.Psi[k]))
    lam = wishart.rvs(df=float(q.nu[k]), scale=W, size=n, random_state=rng).reshape(n, D, D)
    L = np.linalg.cholesky(float(q.kappa[k]) * lam)
    z = rng.normal(size=(n, D))
    mu = np.asarray(q.m[k]) + np.linalg.solve(np.transpose(L, (0, 2, 1)), z[..., None])[..., 0]
    return mu, lam


def _gauss_pdf(x, mu, lam):
    D = x.shape[-1]
    dev = x[None, :] - mu
    quad = np.einsum("si,sij,sj->s", dev, lam, dev)
    _, logdet = np.linalg.slogdet(lam)
    return np.exp(0.5 * logdet - 0.5 * D * np.log(2 * np.pi) - 0.5 * quad)


def test_component_predictive_is_expected_gaussian():
    """Two-layer check: (a) student_logpdf == scipy.multivariate_t (exact, all components);
    (b) the NIW -> Student mapping == MC E_q[N(x|mu,Sigma)] (200k samples, 6 sigma — the
    integrand is heavy-tailed at small dof, so the criterion must be generously powered)."""
    from scipy.stats import multivariate_t

    state, cfg = _fitted_state()
    rng = np.random.default_rng(1)
    for q in (state.spatial, state.color):
        st = pred.niw_predictive(q)
        x = np.asarray(q.m) + 0.5 * rng.normal(size=q.m.shape)  # one probe per component
        analytic = np.exp(np.asarray(pred.student_logpdf(st, jnp.asarray(x))))
        for k in range(x.shape[0]):
            ref = multivariate_t.pdf(x[k], loc=np.asarray(st.loc[k]),
                                     shape=np.asarray(st.scale[k]), df=float(st.dof[k]))
            np.testing.assert_allclose(analytic[k, k], ref, rtol=1e-10)

        n_mc = 200_000
        for k in range(min(2, x.shape[0])):  # MC on the lowest-index (smallest-dof) comps
            mu, lam = _sample_niw(rng, q, k, n_mc)
            vals = _gauss_pdf(x[k], mu, lam)
            se = vals.std(ddof=1) / np.sqrt(n_mc)
            assert abs(analytic[k, k] - vals.mean()) < 6 * se + 1e-12, (
                f"component {k}: St {analytic[k, k]:.6g} vs MC {vals.mean():.6g} ± {se:.2g}"
            )


def test_heldout_loglik_is_expected_mixture_density():
    state, cfg = _fitted_state()
    rng = np.random.default_rng(2)
    xs_q = np.asarray(state.spatial.m)[:2] + 0.3
    xc_q = np.asarray(state.color.m)[:2]
    analytic = float(pred.heldout_loglik(state, cfg, jnp.asarray(xs_q), jnp.asarray(xc_q)))

    epi = np.asarray(expected_pi(state, cfg))
    T = epi.shape[0]
    dens = np.zeros((2, S))
    for k in range(T):
        mu_s, lam_s = _sample_niw(rng, state.spatial, k, S)
        mu_c, lam_c = _sample_niw(rng, state.color, k, S)
        for i in range(2):
            dens[i] += epi[k] * _gauss_pdf(xs_q[i], mu_s, lam_s) * _gauss_pdf(xc_q[i], mu_c, lam_c)
    mc = np.log(dens.mean(axis=1)).mean()
    se = np.abs(np.log(dens.mean(1) + 1e-300) - np.log(dens.mean(1) + dens.std(1) / np.sqrt(S))).max()
    assert abs(analytic - mc) < 5 * se + 5e-3, f"{analytic:.4f} vs MC {mc:.4f}"


def test_conditional_moments_against_ball_sampling():
    state, cfg = _fitted_state()
    rng = np.random.default_rng(3)
    epi = np.asarray(expected_pi(state, cfg))
    T = epi.shape[0]

    n = 4_000_000
    ks = rng.choice(T, p=epi / epi.sum(), size=n)
    s_all = np.empty((n, 2))
    c_all = np.empty((n, 3))
    for k in range(T):
        idx = np.where(ks == k)[0]
        if idx.size == 0:
            continue
        mu_s, lam_s = _sample_niw(rng, state.spatial, k, idx.size)
        z = rng.normal(size=(idx.size, 2))
        s_all[idx] = mu_s + np.linalg.solve(
            np.transpose(np.linalg.cholesky(lam_s), (0, 2, 1)), z[..., None]
        )[..., 0]
        mu_c, lam_c = _sample_niw(rng, state.color, k, idx.size)
        zc = rng.normal(size=(idx.size, 3))
        c_all[idx] = mu_c + np.linalg.solve(
            np.transpose(np.linalg.cholesky(lam_c), (0, 2, 1)), zc[..., None]
        )[..., 0]

    s_star = np.asarray(state.spatial.m[0])
    r = 0.15
    sel = np.linalg.norm(s_all - s_star, axis=1) < r
    assert sel.sum() > 5_000, f"ball too empty ({sel.sum()}) — enlarge r"
    emp_mean = c_all[sel].mean(0)
    emp_var = np.trace(np.cov(c_all[sel].T))

    mean, cov = pred.conditional_color_moments(state, cfg, jnp.asarray(s_star[None, :]))
    an_mean = np.asarray(mean[0])
    an_var = float(np.trace(np.asarray(cov[0])))

    np.testing.assert_allclose(an_mean, emp_mean, atol=0.05)
    np.testing.assert_allclose(an_var, emp_var, rtol=0.15)


def test_conditional_cov_full_matrix_and_psd():
    """Pin the FULL covariance matrix (incl. off-diagonals) against the
    direct mixture-moment formula, and confirm PSD."""
    state, cfg = _fitted_state()
    from dp_splat.prune import expected_pi

    xs_q = jnp.asarray(np.asarray(state.spatial.m)[:3] + 0.2)
    mean, cov = pred.conditional_color_moments(state, cfg, xs_q)

    st_s = pred.niw_predictive(state.spatial)
    st_c = pred.niw_predictive(state.color)
    logw = np.log(np.asarray(expected_pi(state, cfg)))[None, :] + np.asarray(
        pred.student_logpdf(st_s, xs_q))
    w = np.exp(logw - logw.max(1, keepdims=True))
    w /= w.sum(1, keepdims=True)
    mkc = np.asarray(st_c.loc)
    Ck = np.asarray(st_c.scale) * (np.asarray(st_c.dof) / (np.asarray(st_c.dof) - 2.0))[:, None, None]
    for n in range(3):
        m_ref = w[n] @ mkc
        second = np.einsum("k,kij->ij", w[n], Ck + np.einsum("ki,kj->kij", mkc, mkc))
        c_ref = second - np.outer(m_ref, m_ref)
        np.testing.assert_allclose(np.asarray(mean[n]), m_ref, rtol=1e-8)
        np.testing.assert_allclose(np.asarray(cov[n]), c_ref, rtol=1e-7, atol=1e-10)
        eig = np.linalg.eigvalsh(np.asarray(cov[n]))
        assert eig.min() >= -1e-12, f"cov not PSD: min eig {eig.min()}"


def test_dof_guard_raises():
    """The eta_c <= 2 guard must fire for tiny nu0_offset."""
    import pytest as _pytest
    from dp_splat import cavi as _cavi

    rng = np.random.default_rng(0)
    xs = rng.normal(size=(30, 2)); xc = rng.uniform(size=(30, 3))
    cfg = _cavi.Config(weight_prior="dp", T=3, nu0_offset=-1.5, max_iters=1)
    state = _cavi.init_state(0, jnp.asarray(xs), jnp.asarray(xc), cfg)
    # at the prior state nu = nu0 = Dc + nu0_offset = 1.5, so eta_c = nu0 - Dc + 1 = -0.5 <= 2
    with _pytest.raises(ValueError, match="dof <= 2"):
        pred.conditional_color_moments(state, cfg, jnp.asarray(xs[:2]))
