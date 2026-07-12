"""F5 — calibration of predictive color variance (brief §7; Q2 predictive, provisional).

Bin held-out points by predicted tr(Cov[c|s*]) (10 quantile bins); within each bin compare
mean predicted variance to mean realized squared error ||c* - E[c|s*]||^2; report a
regression-style ECE = sum_b frac_b |mse_b - var_b| and the calibration plot.

Data: 3D synthetic with overlapping clusters (sep=3) and component-ambiguity-driven variance so
uncertainty genuinely varies across space.

Run: ~/.venvs/dp-splat/bin/python experiments/fig5_calibration.py
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

from dp_splat import cavi
from dp_splat.predictive import conditional_color_moments
from synthetic3d import colored_gmm_3d

K_TRUE = 10
N_TRAIN, N_TEST = 20_000, 10_000
SEEDS = range(3)
N_BINS = 10


def main():
    rows = []
    curves = []
    for seed in SEEDS:
        xs, xc, _, _ = colored_gmm_3d(200 + seed, K_TRUE, N_TRAIN + N_TEST, sep=3.0,
                                      color_noise=0.08)
        xs_tr, xc_tr = xs[:N_TRAIN], xc[:N_TRAIN]
        xs_te, xc_te = xs[N_TRAIN:], xc[N_TRAIN:]
        cfg = cavi.Config(weight_prior="dp", T=30, alpha=1.0, max_iters=150, tol=1e-6)
        state, _ = cavi.fit(seed, xs_tr, xc_tr, cfg)

        mean, cov = conditional_color_moments(state, cfg, jax.numpy.asarray(xs_te))
        pred_var = np.asarray(jax.numpy.trace(cov, axis1=-2, axis2=-1))
        sq_err = ((np.asarray(mean) - xc_te) ** 2).sum(1)

        qs = np.quantile(pred_var, np.linspace(0, 1, N_BINS + 1))
        qs[-1] += 1e-9
        ece, bins = 0.0, []
        for b in range(N_BINS):
            m = (pred_var >= qs[b]) & (pred_var < qs[b + 1])
            if m.sum() == 0:
                continue
            bv, be = float(pred_var[m].mean()), float(sq_err[m].mean())
            bins.append(dict(bin=b, n=int(m.sum()), pred_var=bv, mse=be))
            ece += (m.mean()) * abs(be - bv)
        rows.append(dict(seed=seed, ece=float(ece),
                         corr=float(np.corrcoef(pred_var, sq_err)[0, 1]),
                         mean_pred_var=float(pred_var.mean()),
                         mean_sq_err=float(sq_err.mean())))
        curves.append(bins)
        print(rows[-1], flush=True)

    (REPO / "experiments" / "out").mkdir(exist_ok=True)
    (REPO / "experiments" / "out" / "f5_calibration.json").write_text(
        json.dumps({"config": dict(K_true=K_TRUE, N_train=N_TRAIN, N_test=N_TEST,
                                   n_bins=N_BINS), "rows": rows, "curves": curves}))

    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    for seed, bins in zip(SEEDS, curves):
        ax.plot([b["pred_var"] for b in bins], [b["mse"] for b in bins], marker="o",
                alpha=0.8, label=f"seed {seed} (ECE {rows[seed]['ece']:.4f})")
    lim = max(max(b["pred_var"] for b in curves[0]), max(b["mse"] for b in curves[0]))
    ax.plot([0, lim], [0, lim], "k:", lw=1, label="perfect calibration")
    ax.set_xlabel("predicted tr Cov[c|s]"); ax.set_ylabel("realized ||c - E[c|s]||²")
    ax.set_title("F5: predictive color-variance calibration")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(REPO / "figures" / "f5_calibration.png", dpi=160)
    print("saved figures/f5_calibration.png")


if __name__ == "__main__":
    main()
