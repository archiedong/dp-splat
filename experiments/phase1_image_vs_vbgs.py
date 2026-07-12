"""Phase 1 acceptance: natural-image PSNR, DP-Splat vs fixed-K VBGS at matched effective budget.

Per image (tiny-imagenet valid, VBGS's own data pipeline and pure-JAX renderer for BOTH
models — brief §3.7 forbids inventing a new rendering map):
  1. DP-Splat (weight_prior=dp, truncation T = VBGS's demo budget 2000) -> K_hat, PSNR_dp
  2. VBGS at its demo K=2000                                            -> PSNR, n_used
  3. VBGS at fixed K = K_hat (matched effective budget)                 -> PSNR_matched
Acceptance (brief §5 Phase 1): PSNR_dp within 0.5 dB of PSNR_matched.

Run: ~/.venvs/dp-splat/bin/python experiments/phase1_image_vs_vbgs.py [n_images] [T] [alpha]
Outputs experiments/out/phase1_image_vs_vbgs_a{alpha}.json + figures/phase1_image_panel_a{alpha}.png
"""

import copy
import json
import sys
import time
from itertools import islice
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "third_party" / "vbgs"))
sys.path.insert(0, str(REPO / "third_party" / "vbgs" / "scripts"))

import vbgs  # noqa: F401  (enables jax x64 globally)
import datasets as ds
import jax.numpy as jnp
import jax.random as jr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vbgs.data.image import image_to_data
from vbgs.data.utils import normalize_data
from vbgs.metrics import calc_psnr
from vbgs.model.train import fit_gmm
from vbgs.model.utils import random_mean_init
from vbgs.render.image import render_img

from model_image import get_image_model

from dp_splat import cavi, prune
from dp_splat.render import to_render_params

SEED = 0


def fit_vbgs(key, img, n_components):
    """VBGS demo pipeline, imagenet.yaml defaults (as in Phase 0)."""
    data = image_to_data(img)
    x, data_params = normalize_data(data)
    key, subkey = jr.split(key)
    mean_init = random_mean_init(subkey, x, component_shape=(n_components,),
                                 event_shape=(5, 1), init_random=True, add_noise=False)
    model = get_image_model(key, n_components=n_components, mean_init=mean_init, beta=0.0,
                            learning_rate=1.0, dof_offset=1.0, position_scale=None)
    initial_model = copy.deepcopy(model)
    model = fit_gmm(initial_model, model, x)
    mu, si = model.denormalize(data_params)
    rendered = render_img(mu, si, model.prior.alpha, img.shape[:2])
    psnr = float(calc_psnr(np.asarray(img, np.float32), rendered.clip(0, 1.0)))
    n_used = int((model.prior.alpha > model.prior.prior_alpha.min()).sum())
    return psnr, n_used, rendered


def fit_ours(img, T, alpha, seed, weight_prior="dp", e0=1.0):
    """Fit our CAVI (dp or converged fixed-K dir) and render with the VBGS renderer."""
    data = image_to_data(img)
    x, data_params = normalize_data(data)
    xs, xc = jnp.asarray(x[:, :2]), jnp.asarray(x[:, 2:])
    cfg = cavi.Config(weight_prior=weight_prior, T=T, alpha=alpha, e0=e0,
                      max_iters=200, tol=1e-6)
    state, hist = cavi.fit(seed, xs, xc, cfg)
    khat = prune.effective_k(state, n_min=1.0)
    mu, cov, w = to_render_params(state, cfg, data_params=data_params, n_min=1.0)
    rendered = render_img(np.asarray(mu), np.asarray(cov), np.asarray(w), img.shape[:2])
    psnr = float(calc_psnr(np.asarray(img, np.float32), rendered.clip(0, 1.0)))
    return psnr, khat, len(hist), rendered


def fit_dp(img, T, alpha, seed):
    return fit_ours(img, T, alpha, seed, weight_prior="dp")


def main():
    n_images = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    T = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    alpha = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0

    stream = ds.load_dataset("Maysee/tiny-imagenet", split="valid", streaming=True)
    key = jr.PRNGKey(SEED)
    rows, panels = [], []
    for i, rec in enumerate(islice(stream, n_images)):
        img = jnp.array(rec["image"]) / 255.0
        if len(img.shape) < 3:
            img = img.reshape((*img.shape, 1)).repeat(3, axis=-1)

        t0 = time.perf_counter()
        psnr_dp, khat, iters, render_dp = fit_dp(img, T, alpha, seed=SEED + i)
        t_dp = time.perf_counter() - t0

        key, k1, k2 = jr.split(key, 3)
        psnr_vbgs, n_used, render_vbgs = fit_vbgs(k1, img, 2000)
        psnr_matched, n_used_matched, render_matched = fit_vbgs(k2, img, khat)
        # deconfound: CONVERGED fixed-K Dirichlet CAVI at K = khat —
        # separates "converged vs single-pass" from "DP prior vs fixed K"
        psnr_dirk, khat_dirk, _, _ = fit_ours(img, khat, alpha, seed=SEED + i,
                                              weight_prior="dir", e0=1.0 / max(khat, 1))

        row = dict(image=i, T=T, alpha=alpha, khat=khat, dp_iters=iters, dp_seconds=t_dp,
                   psnr_dp=psnr_dp, psnr_vbgs_2000=psnr_vbgs, n_used_vbgs_2000=n_used,
                   psnr_vbgs_matched=psnr_matched, n_used_vbgs_matched=n_used_matched,
                   psnr_dir_converged_at_khat=psnr_dirk, khat_dir_converged=khat_dirk,
                   delta_at_matched=psnr_dp - psnr_matched,
                   delta_vs_converged_dir=psnr_dp - psnr_dirk,
                   accept=bool(psnr_dp - psnr_matched >= -0.5))
        rows.append(row)
        print(json.dumps(row), flush=True)
        panels.append((np.asarray(img), render_vbgs, render_matched, render_dp,
                       n_used, n_used_matched, khat, psnr_vbgs, psnr_matched, psnr_dp))

    (REPO / "experiments" / "out").mkdir(exist_ok=True)
    (REPO / "experiments" / "out" / f"phase1_image_vs_vbgs_a{alpha:g}.json").write_text(
        json.dumps({"config": dict(T=T, alpha=alpha, seed=SEED), "rows": rows}, indent=2))

    fig, axes = plt.subplots(len(panels), 4, figsize=(11, 2.9 * len(panels)))
    axes = np.atleast_2d(axes)
    for r, (orig, rv, rm, rd, nu, num, kh, pv, pm, pd) in enumerate(panels):
        for c, (im, title) in enumerate([
            (orig, "original"),
            (rv, f"VBGS K=2000 (used {nu})\n{pv:.2f} dB"),
            (rm, f"VBGS K={kh} (matched)\n{pm:.2f} dB"),
            (rd, f"DP-Splat T=2000, K̂={kh}\n{pd:.2f} dB"),
        ]):
            axes[r, c].imshow(np.clip(im, 0, 1))
            axes[r, c].set_title(title, fontsize=8)
            axes[r, c].axis("off")
    fig.suptitle(f"Phase 1: DP-Splat vs VBGS at matched effective budget (alpha={alpha:g})")
    fig.tight_layout()
    (REPO / "figures").mkdir(exist_ok=True)
    fig.savefig(REPO / "figures" / f"phase1_image_panel_a{alpha:g}.png", dpi=160)

    ok = all(r["accept"] for r in rows)
    deltas = [r["delta_at_matched"] for r in rows]
    print(f"\nACCEPTANCE (>= -0.5 dB at matched budget): {'PASS' if ok else 'FAIL'} "
          f"deltas={['%.2f' % d for d in deltas]}")


if __name__ == "__main__":
    main()
