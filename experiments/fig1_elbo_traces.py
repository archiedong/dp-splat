"""F1 — ELBO traces, 20 seeds (brief §7, theory artifact for T1).

Config below; outputs experiments/out/f1_traces.json and figures/f1_elbo_traces.png.
Run: ~/.venvs/dp-splat/bin/python experiments/fig1_elbo_traces.py
"""

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments"))

import jax

jax.config.update("jax_enable_x64", True)  # float64: exact monotonicity is the point of F1

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dp_splat import cavi
from synthetic import colored_gmm_2d

# --- config (log all choices) ---
K_TRUE = 10
N = 10_000
T = 30  # 3 * K_true, brief Phase 1 acceptance setting
N_SEEDS = 20
MAX_ITERS = 200
VARIANTS = {
    "dp": dict(weight_prior="dp", alpha=1.0),
    "dp+learn_alpha": dict(weight_prior="dp", alpha=1.0, learn_alpha=True),
    "sparse_dir": dict(weight_prior="sparse_dir", e0=0.01),
    "dir": dict(weight_prior="dir", e0=1.0),
}


def main():
    out = {"config": dict(K_true=K_TRUE, N=N, T=T, n_seeds=N_SEEDS, max_iters=MAX_ITERS),
           "traces": {}}
    t_start = time.perf_counter()
    for name, kw in VARIANTS.items():
        traces = []
        worst = 0.0
        for seed in range(N_SEEDS):
            xs, xc, _ = colored_gmm_2d(seed, K_TRUE, N)
            cfg = cavi.Config(T=T, max_iters=MAX_ITERS, tol=1e-6, **kw)
            _, hist = cavi.fit(seed, xs, xc, cfg)
            h = np.asarray(hist)
            d = np.diff(h)
            if d.size:
                worst = min(worst, float((d / np.abs(h[:-1])).min()))
            traces.append([float(v) for v in h])
        out["traces"][name] = traces
        out.setdefault("worst_relative_decrease", {})[name] = worst
        print(f"{name:15s} worst relative ELBO decrease over {N_SEEDS} seeds: {worst:.3e}")

    out["wall_clock_s"] = time.perf_counter() - t_start
    (REPO / "experiments" / "out").mkdir(exist_ok=True)
    (REPO / "experiments" / "out" / "f1_traces.json").write_text(json.dumps(out))

    fig, axes = plt.subplots(1, len(VARIANTS), figsize=(4 * len(VARIANTS), 3.2), sharey=True)
    for ax, (name, traces) in zip(axes, out["traces"].items()):
        for tr in traces:
            ax.plot(np.arange(1, len(tr) + 1), tr, lw=0.7, alpha=0.6)
        ax.set_title(name)
        ax.set_xlabel("CAVI iteration")
    axes[0].set_ylabel("ELBO")
    fig.suptitle(f"F1: ELBO traces, {N_SEEDS} seeds (K_true={K_TRUE}, N={N}, T={T})")
    fig.tight_layout()
    (REPO / "figures").mkdir(exist_ok=True)
    fig.savefig(REPO / "figures" / "f1_elbo_traces.png", dpi=160)
    print("saved figures/f1_elbo_traces.png")


if __name__ == "__main__":
    main()
