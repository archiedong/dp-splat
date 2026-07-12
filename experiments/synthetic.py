"""Synthetic colored GMM generators for Phase 1/2 experiments (brief §5 Phase 1.2).

Well-separated 2D spatial mixtures with distinct colors; K_true up to ~36. Separation is
`sep` sigma units between neighboring means on a sqrt(K) x sqrt(K) grid.
"""

import colorsys

import numpy as np


def colored_gmm_2d(seed: int, K_true: int, N: int, sep: float = 6.0, sigma: float = 1.0,
                   color_noise: float = 0.03):
    """Returns xs (N,2), xc (N,3), z (N,) from a well-separated colored 2D GMM.

    Spatial means on a grid with spacing sep*sigma; colors evenly spaced on the HSV wheel
    (distinct for K_true <= ~36); uniform mixing weights.
    """
    rng = np.random.default_rng(seed)
    side = int(np.ceil(np.sqrt(K_true)))
    grid = np.array([(i, j) for i in range(side) for j in range(side)][:K_true], float)
    means_s = grid * sep * sigma
    means_c = np.array(
        [colorsys.hsv_to_rgb(k / K_true, 0.85, 0.5 + 0.5 * ((k % 2) == 0)) for k in range(K_true)]
    )
    z = rng.integers(0, K_true, size=N)
    xs = means_s[z] + sigma * rng.normal(size=(N, 2))
    xc = np.clip(means_c[z] + color_noise * rng.normal(size=(N, 3)), 0.0, 1.0)
    return xs, xc, z
