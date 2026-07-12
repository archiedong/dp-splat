"""Appendix B sanity identities (brief).

The VBGS numeric regression (with e0 = 1/K, VBGS's shipped default) needs the
baseline-comparison harness and is tracked separately — not in this file.
"""

import numpy as np
import jax.numpy as jnp
from scipy.integrate import quad
from scipy.stats import beta as beta_dist

from dp_splat import priors as pr


def test_beta_expectations_match_numerical_integration():
    """Appendix B: for Beta(a,b), E[log v] = psi(a) - psi(a+b), E[log(1-v)] = psi(b) - psi(a+b).
    Checked against direct numerical integration (independent of digamma)."""
    rng = np.random.default_rng(0)
    for _ in range(5):
        a, b = rng.uniform(0.5, 20.0, size=2)
        elogv, elog1mv = pr.beta_expectations(jnp.array([a]), jnp.array([b]))
        num_elogv = quad(lambda v: np.log(v) * beta_dist.pdf(v, a, b), 0, 1)[0]
        num_elog1mv = quad(lambda v: np.log1p(-v) * beta_dist.pdf(v, a, b), 0, 1)[0]
        assert abs(float(elogv[0]) - num_elogv) < 1e-8
        assert abs(float(elog1mv[0]) - num_elog1mv) < 1e-8


def test_stick_weights_sum_to_one_with_vT_equal_one():
    """Appendix B: sum_k E-weights <= 1 with equality when v_T := 1 (our convention)."""
    rng = np.random.default_rng(1)
    for T in (2, 5, 10):
        g1 = jnp.asarray(rng.uniform(0.5, 50.0, size=T - 1))
        g2 = jnp.asarray(rng.uniform(0.5, 50.0, size=T - 1))
        epi = pr.dp_expected_pi(g1, g2)
        assert epi.shape == (T,)
        assert np.all(np.asarray(epi) >= 0)
        np.testing.assert_allclose(float(epi.sum()), 1.0, rtol=0, atol=1e-12)


def test_dp_elogpi_is_log_of_valid_subprobability():
    rng = np.random.default_rng(2)
    g1 = jnp.asarray(rng.uniform(0.5, 50.0, size=9))
    g2 = jnp.asarray(rng.uniform(0.5, 50.0, size=9))
    elogpi = pr.dp_elogpi(g1, g2)
    assert elogpi.shape == (10,)
    assert np.all(np.asarray(elogpi) < 0)
    # Jensen: E[log pi_k] <= log E[pi_k]
    assert np.all(np.asarray(elogpi) <= np.log(np.asarray(pr.dp_expected_pi(g1, g2))) + 1e-12)


def test_dirichlet_expected_pi_normalized():
    rng = np.random.default_rng(3)
    a = jnp.asarray(rng.uniform(0.01, 5.0, size=8))
    np.testing.assert_allclose(float(pr.dir_expected_pi(a).sum()), 1.0, atol=1e-12)
    # E[log pi] under Dirichlet is also < log E[pi]
    assert np.all(np.asarray(pr.dir_elogpi(a)) < np.log(np.asarray(pr.dir_expected_pi(a))))


def test_alpha_update_matches_eq6():
    """Brief eq. (6) literal check."""
    elog1mv = jnp.asarray([-0.3, -0.7, -0.1])
    w1, w2 = pr.alpha_update(elog1mv, a0=1.0, b0=1.0)
    assert float(w1) == 1.0 + 3  # a0 + T - 1 with T-1 = 3
    np.testing.assert_allclose(float(w2), 1.0 + 1.1, atol=1e-12)
