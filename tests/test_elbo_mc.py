"""Brief §3.5 acceptance test: every ELBO expectation term vs a Monte Carlo estimate from q.

Sampling and reference densities use scipy.stats (wishart, beta, gamma, dirichlet) — an
implementation path independent of the analytic identities in src/. Acceptance = agreement
within MC error (5 standard errors + tiny relative slack).

scipy Wishart(df, scale) has E[X] = df * scale, matching Bishop's W(Lambda | W, nu) with
scale = W = Psi^{-1} — the parameterization documented in src/dp_splat/niw.py.
"""

import numpy as np
import jax.numpy as jnp
from scipy.stats import beta as beta_dist, dirichlet, gamma as gamma_dist, wishart

from dp_splat import cavi, niw, priors as pr

S_SCALAR = 100_000  # brief: 1e5 samples for the cheap scalar terms
S_MATRIX = 20_000  # heavy Wishart terms use SE-based tolerance


def _fitted_state(weight_prior="dp", learn_alpha=True, seed=0, N=40, T=4):
    rng = np.random.default_rng(seed)
    xs = rng.normal(size=(N, 2)) + rng.integers(0, 3, size=(N, 1)) * 2.0
    xc = rng.uniform(0, 1, size=(N, 3))
    cfg = cavi.Config(weight_prior=weight_prior, T=T, alpha=1.5, learn_alpha=learn_alpha,
                      e0=0.1, max_iters=3)
    state = cavi.init_state(seed, jnp.asarray(xs), jnp.asarray(xc), cfg)
    for _ in range(3):
        state = cavi.cavi_step(state, jnp.asarray(xs), jnp.asarray(xc), cfg)
    return state, cfg, xs, xc


def _assert_mc(analytic, samples, label):
    mc, se = samples.mean(), samples.std(ddof=1) / np.sqrt(len(samples))
    tol = 5 * se + 1e-8 * abs(analytic)
    assert abs(analytic - mc) < tol, (
        f"{label}: analytic {analytic:.6f} vs MC {mc:.6f} ± {se:.6f} (tol {tol:.6f})"
    )


def _sample_niw(rng, q: niw.NIW, k: int, S: int):
    """S samples of (mu, Lambda) from q for component k."""
    D = q.dim
    W = np.linalg.inv(np.asarray(q.Psi[k]))
    lam = wishart.rvs(df=float(q.nu[k]), scale=W, size=S, random_state=rng)
    lam = lam.reshape(S, D, D)
    L = np.linalg.cholesky(float(q.kappa[k]) * lam)  # (S,D,D); cov = (kappa*Lam)^{-1}
    z = rng.normal(size=(S, D))
    mu = np.asarray(q.m[k]) + np.linalg.solve(np.transpose(L, (0, 2, 1)), z[..., None])[..., 0]
    return mu, lam


def _wishart_logpdf(lam, df, W):
    p = lam.shape[-1]
    return wishart.logpdf(np.transpose(lam, (1, 2, 0)), df=df, scale=W) if p > 1 else (
        wishart.logpdf(lam[:, 0, 0], df=df, scale=W[0, 0])
    )


def test_niw_terms_p_theta_and_q_theta():
    state, cfg, _, _ = _fitted_state()
    rng = np.random.default_rng(10)
    for name, q, prior in (
        ("spatial", state.spatial, state.spatial_prior),
        ("color", state.color, state.color_prior),
    ):
        an_p = np.asarray(niw.expected_log_prior(q, prior))
        an_q = np.asarray(niw.expected_log_q(q))
        for k in range(cfg.T):
            mu, lam = _sample_niw(rng, q, k, S_MATRIX)
            m0, k0 = np.asarray(prior.m[k]), float(prior.kappa[k])
            W0 = np.linalg.inv(np.asarray(prior.Psi[k]))
            # log p(mu | Lambda) = log N(mu | m0, (k0 Lam)^{-1}) evaluated at sampled mu
            dev = mu - m0
            quad = np.einsum("si,sij,sj->s", dev, k0 * lam, dev)
            _, logdet = np.linalg.slogdet(k0 * lam)
            lp = 0.5 * logdet - 0.5 * q.dim * np.log(2 * np.pi) - 0.5 * quad
            lp += _wishart_logpdf(lam, float(prior.nu[k]), W0)
            _assert_mc(an_p[k], lp, f"E[log p(theta)] {name} k={k}")

            mk, kk = np.asarray(q.m[k]), float(q.kappa[k])
            Wk = np.linalg.inv(np.asarray(q.Psi[k]))
            devq = mu - mk
            quadq = np.einsum("si,sij,sj->s", devq, kk * lam, devq)
            _, logdetq = np.linalg.slogdet(kk * lam)
            lq = 0.5 * logdetq - 0.5 * q.dim * np.log(2 * np.pi) - 0.5 * quadq
            lq += _wishart_logpdf(lam, float(q.nu[k]), Wk)
            _assert_mc(an_q[k], lq, f"E[log q(theta)] {name} k={k}")


def test_expected_loglik_term():
    state, cfg, xs, xc = _fitted_state()
    rng = np.random.default_rng(11)
    an = float(cavi.elbo_expected_loglik(state, jnp.asarray(xs), jnp.asarray(xc)))
    r = np.asarray(state.r)
    totals = np.zeros(S_MATRIX)
    for q, x in ((state.spatial, xs), (state.color, xc)):
        for k in range(cfg.T):
            mu, lam = _sample_niw(rng, q, k, S_MATRIX)
            dev = x[:, None, :] - mu[None, :, :]  # (N,S,D)
            quad = np.einsum("nsi,sij,nsj->ns", dev, lam, dev)
            _, logdet = np.linalg.slogdet(lam)
            lp = 0.5 * logdet[None, :] - 0.5 * q.dim * np.log(2 * np.pi) - 0.5 * quad
            totals += r[:, k] @ lp  # (S,)
    _assert_mc(an, totals, "E[log p(X|Z,theta)]")


def test_stick_terms_p_z_p_v_q_v():
    state, cfg, xs, xc = _fitted_state()
    rng = np.random.default_rng(12)
    w = state.weights
    g1, g2 = np.asarray(w.gamma1), np.asarray(w.gamma2)
    T = cfg.T
    v = beta_dist.rvs(g1, g2, size=(S_SCALAR, T - 1), random_state=rng)
    logpi = np.concatenate(
        [np.log(v), np.zeros((S_SCALAR, 1))], axis=1
    ) + np.concatenate(
        [np.zeros((S_SCALAR, 1)), np.cumsum(np.log1p(-v), axis=1)], axis=1
    )

    an_pz = float(cavi.elbo_log_p_z(state, cfg))
    _assert_mc(an_pz, logpi @ np.asarray(state.r).sum(0), "E[log p(Z|v)]")

    a = gamma_dist.rvs(float(w.w1), scale=1.0 / float(w.w2), size=S_SCALAR, random_state=rng)
    an_pv = float(cavi.elbo_log_p_weights(state, cfg))
    samples_pv = (np.log(a)[:, None] + (a[:, None] - 1.0) * np.log1p(-v)).sum(1)
    _assert_mc(an_pv, samples_pv, "E[log p(v|alpha)]")

    an_qv = float(cavi.elbo_log_q_weights(state, cfg))
    samples_qv = beta_dist.logpdf(v, g1, g2).sum(1)
    _assert_mc(an_qv, samples_qv, "E[log q(v)]")

    an_pa = float(cavi.elbo_log_p_alpha(state, cfg))
    _assert_mc(an_pa, gamma_dist.logpdf(a, cfg.a0, scale=1.0 / cfg.b0), "E[log p(alpha)]")

    an_qa = float(cavi.elbo_log_q_alpha(state, cfg))
    _assert_mc(an_qa, gamma_dist.logpdf(a, float(w.w1), scale=1.0 / float(w.w2)),
               "E[log q(alpha)]")


def test_dirichlet_terms():
    state, cfg, xs, xc = _fitted_state(weight_prior="sparse_dir", learn_alpha=False, seed=1)
    rng = np.random.default_rng(13)
    ap = np.asarray(state.weights.alpha_post)
    pis = dirichlet.rvs(ap, size=S_SCALAR, random_state=rng)
    pis = np.clip(pis, 1e-300, None)

    an_pz = float(cavi.elbo_log_p_z(state, cfg))
    _assert_mc(an_pz, np.log(pis) @ np.asarray(state.r).sum(0), "E[log p(Z|pi)] (dir)")

    an_ppi = float(cavi.elbo_log_p_weights(state, cfg))
    e0 = np.full_like(ap, cfg.e0)
    _assert_mc(an_ppi, dirichlet.logpdf(pis.T, e0), "E[log p(pi)] (dir)")

    an_qpi = float(cavi.elbo_log_q_weights(state, cfg))
    _assert_mc(an_qpi, dirichlet.logpdf(pis.T, ap), "E[log q(pi)] (dir)")
