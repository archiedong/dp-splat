"""Ground rule 5 / brief §3.5: full-batch CAVI must monotonically increase the ELBO —
200 iterations x 20 seeds, all weight-prior variants. A violation is an implementation
(or reported math) bug, never something to clip.

Also a light K-recovery smoke check (full experiment grid is experiments/, Phase 1 F2).
"""

import numpy as np
import jax.numpy as jnp
import pytest

from dp_splat import cavi, prune

N_SEEDS = 20
N_ITERS = 200


def _synthetic(seed, N=200, K_true=3):
    """Well-separated colored 2D GMM (brief §5 Phase 1 synthetic design, tiny version)."""
    rng = np.random.default_rng(seed)
    means_s = np.array([[-4.0, 0.0], [4.0, 0.0], [0.0, 5.0]])[:K_true]
    means_c = np.array([[0.9, 0.1, 0.1], [0.1, 0.9, 0.1], [0.1, 0.1, 0.9]])[:K_true]
    z = rng.integers(0, K_true, size=N)
    xs = means_s[z] + 0.5 * rng.normal(size=(N, 2))
    xc = np.clip(means_c[z] + 0.05 * rng.normal(size=(N, 3)), 0, 1)
    return xs, xc


CONFIGS = {
    "dp": dict(weight_prior="dp", alpha=1.0, learn_alpha=False),
    "dp+learn_alpha": dict(weight_prior="dp", alpha=1.0, learn_alpha=True),
    "sparse_dir": dict(weight_prior="sparse_dir", e0=0.01),
    "dir": dict(weight_prior="dir", e0=1.0),
}


@pytest.mark.parametrize("name", CONFIGS)
def test_elbo_monotone(name):
    violations = []
    for seed in range(N_SEEDS):
        xs, xc = _synthetic(seed)
        cfg = cavi.Config(T=10, max_iters=N_ITERS, tol=0.0, **CONFIGS[name])
        _, hist = cavi.fit(seed, xs, xc, cfg)
        h = np.asarray(hist)
        diffs = np.diff(h)
        bad = np.where(diffs < -1e-10 * np.abs(h[:-1]) - 1e-12)[0]
        if bad.size:
            violations.append((seed, int(bad[0]), float(diffs[bad[0]])))
    assert not violations, f"non-monotone ELBO ({name}): {violations[:5]}"


def test_khat_recovers_ktrue_dp():
    """Smoke version of Phase 1 acceptance: K_hat within +/-1 of K_true = 3 for
    well-separated synthetics at T ~ 3*K_true (dp variant)."""
    hits = 0
    for seed in range(5):
        xs, xc = _synthetic(seed, N=400)
        cfg = cavi.Config(weight_prior="dp", T=9, alpha=1.0, max_iters=150)
        state, _ = cavi.fit(seed, xs, xc, cfg)
        khat = prune.effective_k(state, n_min=1.0)
        hits += int(abs(khat - 3) <= 1)
    assert hits >= 4, f"K_hat recovery failed: {hits}/5 seeds within +/-1 of K_true"
