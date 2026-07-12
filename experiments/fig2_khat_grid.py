"""F2 — K_hat vs K_true grid on 2D synthetics (brief §7; Phase 1 acceptance).

Grid: K_true in {3, 10, 30} x N in {1e3, 1e4, 1e5}, T = 3*K_true, variants dp (at several
fixed alphas, after diagnostics showed strong alpha sensitivity) + dp+learn_alpha + sparse_dir + dir, n_min sensitivity {0.5, 1, 2, 5} (brief §3.7).
Outputs experiments/out/f2_khat.json and figures/f2_khat_grid.png.
Run: ~/.venvs/dp-splat/bin/python experiments/fig2_khat_grid.py [n_seeds] [dp_alphas]
     e.g. ... fig2_khat_grid.py 3 0.1,1,5
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
from synthetic import colored_gmm_2d

K_TRUES = [3, 10, 30]
NS = [1_000, 10_000, 100_000]
N_MINS = [0.5, 1.0, 2.0, 5.0]
MAX_ITERS = 150


def make_variants(dp_alphas):
    v = {f"dp(a={a:g})": dict(weight_prior="dp", alpha=float(a)) for a in dp_alphas}
    v["dp(learn)"] = dict(weight_prior="dp", alpha=1.0, learn_alpha=True)
    v["sparse_dir"] = dict(weight_prior="sparse_dir", e0=0.01)
    v["dir"] = dict(weight_prior="dir", e0=1.0)
    return v


def plot(rows, variants, out_png):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=False)
    marker_cycle = ["o", "v", "D", "P", "s", "^", "X"]
    ls_cycle = ["-", "--", "-.", ":", "-", "--", "-."]
    for ax, K_true in zip(axes, K_TRUES):
        for vi, variant in enumerate(variants):
            xs_, ys_, lo_, hi_ = [], [], [], []
            for N in NS:
                vals = [r["khat"]["1.0"] for r in rows
                        if r["K_true"] == K_true and r["N"] == N and r["variant"] == variant]
                if not vals:
                    continue
                xs_.append(N * (1.0 + 0.06 * (vi - (len(variants) - 1) / 2)))  # de-occlusion
                ys_.append(np.mean(vals)); lo_.append(np.min(vals)); hi_.append(np.max(vals))
            ax.errorbar(xs_, ys_,
                        yerr=[np.array(ys_) - lo_, np.array(hi_) - ys_],
                        marker=marker_cycle[vi % 7], ls=ls_cycle[vi % 7], ms=4,
                        capsize=2, label=variant)
        ax.axhline(K_true, color="k", ls=":", lw=1)
        ax.set_xscale("log")
        ax.set_title(f"K_true = {K_true} (T = {3*K_true})")
        ax.set_xlabel("N")
    axes[0].set_ylabel(r"$\hat{K}$ ($n_{\min}=1$)")
    axes[0].legend(fontsize=7)
    fig.suptitle("Effective components vs K_true (2D synthetics)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    print(f"saved {out_png}")


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    dp_alphas = [float(a) for a in (sys.argv[2] if len(sys.argv) > 2 else "1").split(",")]
    VARIANTS = make_variants(dp_alphas)
    out_json = REPO / "experiments" / "out" / "f2_khat.json"
    if "--replot" in sys.argv and out_json.exists():
        rows = json.loads(out_json.read_text())["rows"]
        variants = list(dict.fromkeys(r["variant"] for r in rows))
        plot(rows, variants, REPO / "figures" / "f2_khat_grid.png")
        return
    rows = []
    for K_true in K_TRUES:
        T = 3 * K_true
        for N in NS:
            for variant, kw in VARIANTS.items():
                for seed in range(n_seeds):
                    xs, xc, _ = colored_gmm_2d(1000 * K_true + seed, K_true, N)
                    cfg = cavi.Config(T=T, max_iters=MAX_ITERS, tol=1e-6, **kw)
                    t0 = time.perf_counter()
                    state, hist = cavi.fit(seed, xs, xc, cfg)
                    row = dict(
                        K_true=K_true, N=N, T=T, variant=variant, seed=seed,
                        iters=len(hist), elbo=float(hist[-1]),
                        seconds=time.perf_counter() - t0,
                        khat={str(nm): prune.effective_k(state, nm) for nm in N_MINS},
                        entropy_k=prune.entropy_effective_k(state, cfg),
                    )
                    rows.append(row)
                    print(f"K_true={K_true} N={N} {variant} seed={seed}: "
                          f"khat(n_min=1)={row['khat']['1.0']} iters={row['iters']} "
                          f"{row['seconds']:.1f}s", flush=True)

    (REPO / "experiments" / "out").mkdir(exist_ok=True)
    (REPO / "experiments" / "out" / "f2_khat.json").write_text(
        json.dumps({"n_seeds": n_seeds, "rows": rows})
    )

    (REPO / "figures").mkdir(exist_ok=True)
    plot(rows, list(VARIANTS), REPO / "figures" / "f2_khat_grid.png")


if __name__ == "__main__":
    main()
