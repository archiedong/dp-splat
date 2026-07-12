"""Phase 0 baseline reproduction: VBGS image demo on a Tiny-ImageNet subset.

Mirrors third_party/vbgs/scripts/train_image.py exactly (same functions, same
imagenet.yaml hyperparameters); only the number of evaluated validation images
is reduced for local CPU runs. Full 10k-image benchmark is deferred to the GPU
box (REPRO.md). VBGS code is used unmodified via sys.path.

Usage:
    python experiments/phase0_baseline_image.py [n_images] [n_components]
"""

import json
import sys
import time
from itertools import islice
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "third_party" / "vbgs"))
sys.path.insert(0, str(REPO / "third_party" / "vbgs" / "scripts"))

import copy

import datasets as ds
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from vbgs.data.image import image_to_data
from vbgs.data.utils import normalize_data
from vbgs.metrics import calc_mse, calc_psnr
from vbgs.model.train import fit_gmm
from vbgs.model.utils import random_mean_init
from vbgs.render.image import render_img

from model_image import get_image_model

# imagenet.yaml defaults (third_party/vbgs/scripts/configs/imagenet.yaml)
SEED = 0
N_COMPONENTS = 2000
INIT_RANDOM = True
DOF = 1.0
SCALE = None
LEARNING_RATE = 1.0
BETA = 0.0
N_ITERS = 1


def fit_one(key, img, n_components):
    data = image_to_data(img)
    x, data_params = normalize_data(data)

    key, subkey = jr.split(key)
    mean_init = random_mean_init(
        subkey,
        x,
        component_shape=(n_components,),
        event_shape=(5, 1),
        init_random=INIT_RANDOM,
        add_noise=False,
    )

    model = get_image_model(
        key,
        n_components=n_components,
        mean_init=mean_init,
        beta=BETA,
        learning_rate=LEARNING_RATE,
        dof_offset=DOF,
        position_scale=SCALE,
    )

    initial_model = copy.deepcopy(model)
    for _ in range(N_ITERS):
        model = fit_gmm(initial_model, model, x)

    mu, si = model.denormalize(data_params)
    rendered = render_img(mu, si, model.prior.alpha, img.shape[:2])

    mse = calc_mse(np.asarray(img, dtype=np.float32), rendered.clip(0, 1.0))
    psnr = calc_psnr(np.asarray(img, dtype=np.float32), rendered.clip(0, 1.0))
    n_used = int((model.prior.alpha > model.prior.prior_alpha.min()).sum())
    return {"mse": float(mse), "psnr": float(psnr), "n_used": n_used}


def main():
    n_images = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    n_components = int(sys.argv[2]) if len(sys.argv) > 2 else N_COMPONENTS

    stream = ds.load_dataset("Maysee/tiny-imagenet", split="valid", streaming=True)
    key = jr.PRNGKey(SEED)

    metrics = []
    for i, row in enumerate(islice(stream, n_images)):
        img = jnp.array(row["image"]) / 255.0
        if len(img.shape) < 3:
            img = img.reshape((*img.shape, 1)).repeat(3, axis=-1)

        key, subkey = jr.split(key)
        t0 = time.perf_counter()
        m = fit_one(subkey, img, n_components)
        m["seconds"] = time.perf_counter() - t0
        metrics.append(m)
        print(f"[{i+1}/{n_images}] psnr={m['psnr']:.2f} dB  mse={m['mse']:.5f} "
              f"n_used={m['n_used']}  {m['seconds']:.1f}s")

    summary = {
        "n_images": n_images,
        "n_components": n_components,
        "psnr_mean": float(np.mean([m["psnr"] for m in metrics])),
        "psnr_std": float(np.std([m["psnr"] for m in metrics])),
        "mse_mean": float(np.mean([m["mse"] for m in metrics])),
        "n_used_mean": float(np.mean([m["n_used"] for m in metrics])),
        "seconds_per_image_mean": float(np.mean([m["seconds"] for m in metrics])),
        "per_image": metrics,
    }
    out = REPO / "experiments" / "phase0_baseline_image_results.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_image"}, indent=2))


if __name__ == "__main__":
    main()
