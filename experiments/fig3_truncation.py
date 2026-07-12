"""F3 — held-out log-likelihood and K_hat vs truncation T, with the Ishwaran–James bound
(brief §7 / T2).

Bound (verified against Ishwaran & James 2001, JASA — see REPRO.md): the L1/TV distance
between the marginal of N observations under truncation T vs the full DP satisfies
    ||m_T - m_inf||_1 <= 2 [1 - E{(sum_{k<T} pi_k)^N}] ~= 4 N exp(-(T-1)/alpha),
expectation under pi ~ stick-breaking(alpha). We overlay both the exact-form MC evaluation
and the exponential approximation.

Run: ~/.venvs/dp-splat/bin/python experiments/fig3_truncation.py
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
from dp_splat.predictive import heldout_loglik
from synthetic3d import colored_gmm_3d

K_TRUE = 10
N_TRAIN = 20_000
N_TEST = 5_000
ALPHA = 1.0
TS = [10, 25, 50, 100, 200]
SEEDS = range(3)


def ij_bound_exact(T, alpha, N, n_mc=200_000, seed=0):
    """MC evaluation of 2[1 - E{(sum_{k<T} pi_k)^N}]; sum_{k<T} pi_k = 1 - prod(1 - v_k).
    Returns (value, at_floor): at_floor=True when the MC estimate underflows to 0 (the true
    value is below ~1/n_mc resolution) — plotted as censored, not as data."""
    rng = np.random.default_rng(seed)
    v = rng.beta(1.0, alpha, size=(n_mc, T - 1))
    log_tail = np.log1p(-v).sum(axis=1)  # log prod (1-v_k)
    x = -np.expm1(log_tail)  # sum_{k<T} pi_k
    val = 2.0 * (1.0 - np.mean(np.exp(N * np.log(np.clip(x, 1e-300, 1.0)))))
    return val, bool(val <= 0.0)


def ij_bound_rigorous(T, alpha, N):
    """Rigorous closed form: 1 - E[X^N] <= N E[1-X] = N E[prod(1-v_k)] = N (a/(1+a))^{T-1},
    so TV <= 2N (alpha/(1+alpha))^{T-1}. (The brief's 4N e^{-(T-1)/alpha} is Ishwaran-James'
    LARGE-alpha approximation of this and is anti-conservative at alpha ~ 1;
    see the paper's truncation proposition.)"""
    return 2.0 * N * (alpha / (1.0 + alpha)) ** (T - 1)


def compute(out_path):
    xs, xc, _, _ = colored_gmm_3d(0, K_TRUE, N_TRAIN + N_TEST)
    xs_tr, xc_tr = xs[:N_TRAIN], xc[:N_TRAIN]
    xs_te, xc_te = xs[N_TRAIN:], xc[N_TRAIN:]

    rows = []
    for T in TS:
        for seed in SEEDS:
            cfg = cavi.Config(weight_prior="dp", T=T, alpha=ALPHA, max_iters=150, tol=1e-6)
            t0 = time.perf_counter()
            state, hist = cavi.fit(seed, xs_tr, xc_tr, cfg)
            row = dict(
                T=T, seed=seed, iters=len(hist), elbo=float(hist[-1]),
                heldout_ll=float(heldout_loglik(state, cfg, xs_te, xc_te)),
                khat=prune.effective_k(state, 1.0),
                seconds=time.perf_counter() - t0,
            )
            rows.append(row)
            print(row, flush=True)

    bounds = {}
    for T in TS:
        exact, floored = ij_bound_exact(T, ALPHA, N_TRAIN)
        bounds[T] = dict(exact=exact, exact_at_mc_floor=floored,
                         rigorous=ij_bound_rigorous(T, ALPHA, N_TRAIN),
                         ij_large_alpha_approx=float(4 * N_TRAIN * np.exp(-(T - 1) / ALPHA)))
    out_path.write_text(
        json.dumps({"config": dict(K_true=K_TRUE, N_train=N_TRAIN, N_test=N_TEST,
                                   alpha=ALPHA), "rows": rows, "bounds": bounds}))
    return rows, bounds


def plot(rows, bounds):
    TS_ = sorted(set(r["T"] for r in rows))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, key, label in ((axes[0], "heldout_ll", "held-out log-lik / point (nats)"),
                           (axes[1], "khat", r"$\hat{K}$ ($n_{\min}=1$)")):
        for r in rows:  # individual seeds (bimodality at T=K_true visible)
            ax.plot(r["T"], r[key], "o", ms=3, color="C0", alpha=0.45)
        m = [np.mean([r[key] for r in rows if r["T"] == T]) for T in TS_]
        ax.plot(TS_, m, "-", color="C0")
        ax.set_xscale("log"); ax.set_xlabel("truncation T"); ax.set_ylabel(label)
    axes[1].axhline(K_TRUE, color="k", ls=":", lw=1)

    b = {int(k): v for k, v in bounds.items()} if isinstance(next(iter(bounds)), str) else bounds
    Ts_ok = [T for T in TS_ if not b[T]["exact_at_mc_floor"]]
    Ts_floor = [T for T in TS_ if b[T]["exact_at_mc_floor"]]
    axes[2].plot(Ts_ok, [b[T]["exact"] for T in Ts_ok], marker="o",
                 label=r"intermediate bound $2[1-E\{(\sum_{k<T}\pi_k)^N\}]$ (MC, exact)")
    if Ts_floor:
        axes[2].plot(Ts_floor, [1e-13] * len(Ts_floor), "v", color="C0", mfc="none",
                     label="MC floor (true value below resolution)")
    axes[2].plot(TS_, [min(b[T]["rigorous"], 2.0) for T in TS_], marker="s", ls="-",
                 label=r"rigorous $2N(\alpha/(1+\alpha))^{T-1}$")
    axes[2].plot(TS_, [min(b[T]["ij_large_alpha_approx"], 2.0) for T in TS_], marker="x",
                 ls="--", label=r"IJ $4Ne^{-(T-1)/\alpha}$ (large-$\alpha$ approx.)")
    axes[2].set_yscale("log"); axes[2].set_xscale("log")
    axes[2].set_ylim(1e-14, 4)
    axes[2].set_xlabel("truncation T"); axes[2].set_ylabel("TV bound")
    axes[2].legend(fontsize=6.5)
    fig.suptitle(f"Truncation sweep (K_true={K_TRUE}, N={N_TRAIN}, alpha={ALPHA})")
    fig.tight_layout()
    fig.savefig(REPO / "figures" / "f3_truncation.png", dpi=160)
    print("saved figures/f3_truncation.png")


def main():
    out_path = REPO / "experiments" / "out" / "f3_truncation.json"
    out_path.parent.mkdir(exist_ok=True)
    if "--replot" in sys.argv and out_path.exists():
        d = json.loads(out_path.read_text())
        rows, bounds = d["rows"], {int(k): v for k, v in d["bounds"].items()}
        # recompute bound variants if the JSON predates the Q9 fix
        if "rigorous" not in next(iter(bounds.values())):
            for T in bounds:
                exact, floored = ij_bound_exact(T, ALPHA, N_TRAIN)
                bounds[T] = dict(exact=exact, exact_at_mc_floor=floored,
                                 rigorous=ij_bound_rigorous(T, ALPHA, N_TRAIN),
                                 ij_large_alpha_approx=float(
                                     4 * N_TRAIN * np.exp(-(T - 1) / ALPHA)))
            d["bounds"] = bounds
            out_path.write_text(json.dumps(d))
    else:
        rows, bounds = compute(out_path)
    plot(rows, bounds)


if __name__ == "__main__":
    main()
