"""F4 — K_hat vs N growth curves, DP vs sparse-finite (brief §7 / T3).

Expected narrative (Miller–Harrison vs Rousseau–Mengersen) tested empirically under CAVI.
CAVI for N <= 1e5; SVI (brief §3.6 defaults, 400 steps + one full-batch E-step) above.
MFM-MCMC comparison is a flag-gated stretch goal in the brief — skipped, logged in REPRO.

Run: ~/.venvs/dp-splat/bin/python experiments/fig4_khat_vs_n.py
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
NS = [1_000, 10_000, 100_000, 1_000_000]
SEEDS = range(3)
VARIANTS = {
    "dp(a=0.1)": dict(weight_prior="dp", alpha=0.1),
    "dp(a=1)": dict(weight_prior="dp", alpha=1.0),
    "dp(a=5)": dict(weight_prior="dp", alpha=5.0),
    "dp(learn)": dict(weight_prior="dp", alpha=1.0, learn_alpha=True),
    "sparse_dir(e0=0.1)": dict(weight_prior="sparse_dir", e0=0.1),
    "sparse_dir(e0=0.01)": dict(weight_prior="sparse_dir", e0=0.01),
    "sparse_dir(e0=0.001)": dict(weight_prior="sparse_dir", e0=0.001),
}
CAVI_MAX_N = 100_000


def main():
    out_json = REPO / "experiments" / "out" / "f4_khat_vs_n.json"
    if "--replot" in sys.argv and out_json.exists():
        import json as _json
        _plot(_json.loads(out_json.read_text())["rows"])
        return
    rows = []
    for N in NS:
        for variant, kw in VARIANTS.items():
            for seed in SEEDS:
                xs, xc, _, _ = colored_gmm_3d(100 + seed, K_TRUE, N)
                cfg = cavi.Config(T=T, max_iters=150, tol=1e-6, **kw)
                t0 = time.perf_counter()
                if N <= CAVI_MAX_N:
                    state, _ = cavi.fit(seed, xs, xc, cfg)
                    method = "cavi"
                else:
                    state, _ = fit_svi(seed, xs, xc, cfg,
                                       SVIConfig(batch_size=2**16, n_steps=400))
                    state = cavi.cavi_step(state, jax.numpy.asarray(xs),
                                           jax.numpy.asarray(xc), cfg)
                    method = "svi"
                row = dict(N=N, variant=variant, seed=seed, method=method,
                           khat={str(nm): prune.effective_k(state, nm)
                                 for nm in (0.5, 1.0, 2.0, 5.0)},
                           entropy_k=prune.entropy_effective_k(state, cfg),
                           seconds=time.perf_counter() - t0)
                rows.append(row)
                print(N, variant, seed, method, row["khat"]["1.0"],
                      f"{row['seconds']:.0f}s", flush=True)

    (REPO / "experiments" / "out").mkdir(exist_ok=True)
    (REPO / "experiments" / "out" / "f4_khat_vs_n.json").write_text(
        json.dumps({"config": dict(K_true=K_TRUE, T=T), "rows": rows}))

    _plot(rows)


def _plot(rows):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    markers, lss = "ovDPsX^", ["-", "--", "-.", ":", "-", "--", "-."]
    nv = len(VARIANTS)
    for vi, variant in enumerate(VARIANTS):
        xs_, ys_, lo_, hi_ = [], [], [], []
        for N in NS:
            vals = [r["khat"]["1.0"] for r in rows
                    if r["N"] == N and r["variant"] == variant]
            if not vals:
                continue
            xs_.append(N * (1.0 + 0.07 * (vi - (nv - 1) / 2)))  # de-occlusion offset
            ys_.append(np.mean(vals)); lo_.append(np.min(vals)); hi_.append(np.max(vals))
        ax.errorbar(xs_, ys_, yerr=[np.array(ys_) - lo_, np.array(hi_) - ys_],
                    marker=markers[vi % 7], ls=lss[vi % 7], ms=4, capsize=2, label=variant)
    ax.axhline(K_TRUE, color="k", ls=":", lw=1)
    ax.axhline(T, color="gray", ls="--", lw=1)
    ax.text(NS[0], T - 2.5, "T (truncation)", fontsize=7, color="gray")
    ax.axvline(CAVI_MAX_N * 3, color="gray", lw=0.8, alpha=0.6)
    ax.text(CAVI_MAX_N * 3.3, K_TRUE + 1, "CAVI | SVI", fontsize=7, color="gray",
            rotation=90)
    ax.set_xscale("log"); ax.set_xlabel("N"); ax.set_ylabel(r"$\hat{K}$ ($n_{\min}=1$)")
    ax.set_title(f"K-hat growth with N (K_true={K_TRUE}, T={T})", fontsize=10)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(REPO / "figures" / "f4_khat_vs_n.png", dpi=160)
    print("saved figures/f4_khat_vs_n.png")


if __name__ == "__main__":
    main()
