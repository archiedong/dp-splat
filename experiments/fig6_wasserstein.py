"""F6 — Wasserstein-2 distance between fitted and true mixing measures vs N (brief §7 / T4).

Mixing measure = weighted atoms on component parameters; per Nguyen (2013) we use atom
locations in joint (spatial mean, color mean) space, ground cost = squared Euclidean, and
exact discrete OT via POT. Fitted atoms: posterior means with weights E[pi_k], pruned at
n_min=1 and renormalized. True: the generator's (means_s, means_c, w).

Run: ~/.venvs/dp-splat/bin/python experiments/fig6_wasserstein.py
"""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments"))

import jax

jax.config.update("jax_enable_x64", True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import ot

from dp_splat import cavi, prune
from synthetic3d import colored_gmm_3d

K_TRUE = 10
T = 30
NS = [1_000, 10_000, 100_000]
SEEDS = range(3)
VARIANTS = {
    "dp(a=1)": dict(weight_prior="dp", alpha=1.0),
    "sparse_dir(e0=0.01)": dict(weight_prior="sparse_dir", e0=0.01),
}


def w2_mixing(state, cfg, truth, n_min=1.0):
    means_s, means_c, w_true = truth
    atoms_true = np.concatenate([means_s, means_c], axis=1)
    keep = np.asarray(prune.soft_counts(state)) > n_min
    w_fit = np.asarray(prune.expected_pi(state, cfg)) * keep
    w_fit = w_fit / w_fit.sum()
    atoms_fit = np.concatenate(
        [np.asarray(state.spatial.m), np.asarray(state.color.m)], axis=1)
    M = ot.dist(atoms_fit, atoms_true, metric="sqeuclidean")
    return float(np.sqrt(ot.emd2(w_fit, np.asarray(w_true), M)))


def main():
    rows = []
    for N in NS:
        for variant, kw in VARIANTS.items():
            for seed in SEEDS:
                xs, xc, _, truth = colored_gmm_3d(300 + seed, K_TRUE, N)
                cfg = cavi.Config(T=T, max_iters=150, tol=1e-6, **kw)
                state, _ = cavi.fit(seed, xs, xc, cfg)
                w2 = w2_mixing(state, cfg, truth)
                rows.append(dict(N=N, variant=variant, seed=seed, w2=w2,
                                 khat=prune.effective_k(state, 1.0)))
                print(rows[-1], flush=True)

    (REPO / "experiments" / "out").mkdir(exist_ok=True)
    (REPO / "experiments" / "out" / "f6_wasserstein.json").write_text(
        json.dumps({"config": dict(K_true=K_TRUE, T=T), "rows": rows}))

    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    for vi, variant in enumerate(VARIANTS):
        m = [np.mean([r["w2"] for r in rows if r["N"] == N and r["variant"] == variant])
             for N in NS]
        s = [np.std([r["w2"] for r in rows if r["N"] == N and r["variant"] == variant])
             for N in NS]
        ax.errorbar(NS, m, yerr=s, marker="os"[vi], label=variant)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("N"); ax.set_ylabel(r"$W_2$(fitted, true mixing measure)")
    ax.set_title("Mixing-measure contraction")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(REPO / "figures" / "f6_wasserstein.png", dpi=160)
    print("saved figures/f6_wasserstein.png")


if __name__ == "__main__":
    main()
