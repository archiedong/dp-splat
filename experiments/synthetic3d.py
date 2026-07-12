"""Colored 3D point-cloud GMMs with known structure (brief §5 Phase 2.1)."""

import colorsys

import numpy as np


def colored_gmm_3d(seed: int, K_true: int, N: int, sep: float = 6.0, sigma: float = 1.0,
                   color_noise: float = 0.03, weights: str = "uniform"):
    """xs (N,3), xc (N,3), z (N,), plus the true mixing measure (means_s, means_c, w_true).

    Spatial means on a 3D grid with spacing sep*sigma; colors on the HSV wheel; anisotropic
    per-component covariances (random rotation, axis scales in [0.5, 1.5]*sigma).
    """
    rng = np.random.default_rng(seed)
    side = int(np.ceil(K_true ** (1 / 3)))
    grid = np.array(
        [(i, j, k) for i in range(side) for j in range(side) for k in range(side)][:K_true],
        float,
    )
    means_s = grid * sep * sigma
    means_c = np.array(
        [colorsys.hsv_to_rgb(k / K_true, 0.85, 0.5 + 0.5 * ((k % 2) == 0)) for k in range(K_true)]
    )
    covs = []
    for _ in range(K_true):
        Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        scales = sigma * rng.uniform(0.5, 1.5, size=3)
        covs.append(Q @ np.diag(scales**2) @ Q.T)
    covs = np.stack(covs)

    if weights == "uniform":
        w = np.full(K_true, 1.0 / K_true)
    else:  # geometric decay — unequal cluster sizes
        w = 0.7 ** np.arange(K_true)
        w /= w.sum()

    z = rng.choice(K_true, p=w, size=N)
    eps = rng.normal(size=(N, 3))
    L = np.linalg.cholesky(covs)  # (K,3,3)
    xs = means_s[z] + np.einsum("nij,nj->ni", L[z], eps)
    xc = np.clip(means_c[z] + color_noise * rng.normal(size=(N, 3)), 0.0, 1.0)
    return xs, xc, z, (means_s, means_c, w)
