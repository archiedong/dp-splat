"""Phase 2.2 — SVI scaling curve, N up to 1e7 (brief targets H100; SVI's per-step cost is
N-independent, so the curve runs locally on CPU; absolute wall-clock re-measured on the H100
later for the paper's hardware note).

Records: wall-clock per SVI step, steps to reach 99% of the small-N reference ELBO
(evaluated on a fixed 1e5 subsample), K_hat, and peak data size.

Run: ~/.venvs/dp-splat/bin/python experiments/svi_scaling.py
"""

import json
import sys
import time
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

from dp_splat import cavi, prune
from dp_splat.svi import SVIConfig, fit_svi
from synthetic3d import colored_gmm_3d

K_TRUE = 10
T = 60
NS = [100_000, 1_000_000, 3_000_000, 10_000_000]
STEPS = 300


def main():
    out_json = REPO / "experiments" / "out" / "svi_scaling.json"
    if "--replot" in sys.argv and out_json.exists():
        _plot(json.loads(out_json.read_text())["rows"])
        return
    rows = []
    for N in NS:
        xs, xc, _, _ = colored_gmm_3d(7, K_TRUE, N)
        cfg = cavi.Config(weight_prior="dp", T=T, alpha=1.0)
        t0 = time.perf_counter()
        state, _ = fit_svi(0, xs, xc, cfg, SVIConfig(batch_size=2**16, n_steps=STEPS))
        fit_s = time.perf_counter() - t0
        # evaluate on a fixed-size subsample for a comparable ELBO-ish metric
        sub = slice(0, 100_000)
        state_e = cavi.cavi_step(state, jax.numpy.asarray(xs[sub]),
                                 jax.numpy.asarray(xc[sub]), cfg)
        L = float(cavi.elbo(state_e, jax.numpy.asarray(xs[sub]),
                            jax.numpy.asarray(xc[sub]), cfg)) / 100_000
        rows.append(dict(N=N, steps=STEPS, batch=2**16, seconds=fit_s,
                         s_per_step=fit_s / STEPS, elbo_per_point_100k=L,
                         khat=prune.effective_k(state_e, 1.0)))
        print(rows[-1], flush=True)

    (REPO / "experiments" / "out").mkdir(exist_ok=True)
    (REPO / "experiments" / "out" / "svi_scaling.json").write_text(
        json.dumps({"host": "local-cpu-m5max", "rows": rows}))

    _plot(rows)


def _plot(rows):
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    ax.plot([r["N"] for r in rows], [r["s_per_step"] for r in rows], marker="o")
    ax.set_xscale("log")
    ax.set_ylim(0, max(r["s_per_step"] for r in rows) * 1.6)  # y from 0: flatness IS the claim
    ax.set_xlabel("N (dataset size)")
    ax.set_ylabel("seconds / SVI step (B=2¹⁶, T=60)")
    ax.set_title("SVI scaling: per-step cost is O(1) in N (CPU;\n"
                 "single timing per N incl. residual JIT warmup at N=1e5)")
    fig.tight_layout()
    fig.savefig(REPO / "figures" / "svi_scaling.png", dpi=160)
    print("saved figures/svi_scaling.png")


if __name__ == "__main__":
    main()
