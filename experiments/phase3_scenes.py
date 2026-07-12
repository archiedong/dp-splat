"""Phase 3 (local scope): Blender scenes — K_hat trajectory, complexity ordering,
point-level held-out color prediction, uncertainty visualization.

Local hardware cannot run the INRIA CUDA rasterizer (LIMITATIONS.md E3), so rasterized
PSNR/SSIM (F7) and rendered panels (F8) are H100-deferred. This script produces the
rasterizer-free Phase 3 artifacts:
  - K_hat trajectory during continual fitting (DP-Splat SVI over frames) and VBGS n_used
    trajectory (their fit_gmm_step, exact train_objects.py protocol) at matched budget T=K.
  - Held-out-frame point-level color prediction: E[c|s] vs ground truth for both models
    (same conditional-mixture formula; for VBGS computed from their denormalized
    (mu, si, alpha) exactly as their 2D renderer does, extended to 3D points).
  - Per-point predictive color variance (DP-Splat only — VBGS has no color variance, its
    color precision is fixed) scatter-projected onto the held-out view.

Run: ~/.venvs/dp-splat/bin/python experiments/phase3_scenes.py [scene ...]
Data: ~/dp-splat-data/blender/{scene}/ (test split + depth; REPRO.md).
"""

import copy
import json
import sys
import time
from itertools import islice
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA_ROOT = Path.home() / "dp-splat-data" / "blender"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "third_party" / "vbgs"))
sys.path.insert(0, str(REPO / "third_party" / "vbgs" / "scripts"))

import vbgs  # noqa: F401  (jax x64)
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vbgs.data.blender import BlenderDataIterator
from vbgs.data.utils import normalize_data
from vbgs.model.train import fit_gmm_step
from vbgs.model.utils import random_mean_init
from model_volume import get_volume_delta_mixture

from dp_splat import cavi, prune
from dp_splat.predictive import conditional_color_moments, heldout_loglik
from dp_splat.svi import SVIConfig, svi_step

T = 2000  # matched budget (component cap for both models)
N_FRAMES = 40  # training frames (test split has 200; local scope)
SUBSAMPLE = 20_000  # points per frame
ALPHA = 1.0
PASSES = 10  # SVI epochs over the frame stream (single-pass severely underfits under the
# Hoffman step-size schedule; VBGS's continual scheme is an exact one-shot conjugate update
# by construction, so multi-epoch SVI is the fair converged counterpart — REPRO.md note)
SEED = 0


def load_frames(scene):
    it = BlenderDataIterator(DATA_ROOT / scene, subsample=SUBSAMPLE)
    frames = list(islice(it, N_FRAMES + 1))  # +1 held-out
    agg = np.concatenate(frames[:N_FRAMES])
    _, data_params = normalize_data(jnp.asarray(agg))
    norm = lambda d: np.asarray(normalize_data(jnp.asarray(d), data_params)[0])
    return [norm(f) for f in frames[:N_FRAMES]], norm(frames[N_FRAMES]), data_params


def fit_dp(frames_n, heldout_n, color_std, alpha=ALPHA):
    xs_all = jnp.asarray(np.concatenate(frames_n)[:, :3])
    xc_all = jnp.asarray(np.concatenate(frames_n)[:, 3:])
    N = xs_all.shape[0]
    cfg = cavi.Config(weight_prior="dp", T=T, alpha=alpha)
    state = cavi.init_state(SEED, xs_all, xc_all, cfg)
    svi = SVIConfig(batch_size=SUBSAMPLE, tau0=64.0, kappa_sched=0.7)
    traj, t0 = [], time.perf_counter()
    step = 0
    rng = np.random.default_rng(SEED)
    offsets = np.cumsum([0] + [f.shape[0] for f in frames_n])
    for ep in range(PASSES):  # multi-epoch SVI over the frame stream
        order = rng.permutation(N_FRAMES) if ep > 0 else np.arange(N_FRAMES)
        for fi in order:
            idx = jnp.arange(offsets[fi], offsets[fi + 1])
            rho = (step + 1 + svi.tau0) ** (-svi.kappa_sched)
            state = svi_step(state, xs_all, xc_all, cfg, idx, N / idx.shape[0], rho)
            step += 1
            traj.append(dict(step=step, epoch=ep, khat=prune.effective_k(state, 1.0)))
    secs = time.perf_counter() - t0

    xs_h = jnp.asarray(heldout_n[:, :3])
    xc_h = heldout_n[:, 3:]
    mean, cov = conditional_color_moments(state, cfg, xs_h)
    # metrics in ORIGINAL color units ([0,1] rgb): x_orig = x_norm * std
    err_unit = (np.asarray(mean) - xc_h) * color_std[None, :]
    mse = float((err_unit**2).mean())
    pvar = np.asarray(jnp.diagonal(cov, axis1=-2, axis2=-1))  # (N, Dc) per-dim variances
    hll = float(heldout_loglik(state, cfg, xs_h, jnp.asarray(xc_h)))
    return dict(traj=traj, seconds=secs, khat=prune.effective_k(state, 1.0),
                heldout_color_mse=mse, heldout_point_psnr=float(-10 * np.log10(mse)),
                heldout_ll=hll, alpha=alpha, passes=PASSES), pvar, np.asarray(mean)


def fit_vbgs(frames_n, heldout_n, color_std):
    key = jr.PRNGKey(SEED)
    agg = np.concatenate(frames_n)
    rng = np.random.default_rng(0)
    x_init = agg[rng.permutation(agg.shape[0])[:T]]
    key, sk = jr.split(key)
    mean_init = random_mean_init(key=sk, x=jnp.asarray(x_init), component_shape=(T,),
                                 event_shape=(6, 1), init_random=False, add_noise=False)
    key, sk = jr.split(key)
    prior_model = get_volume_delta_mixture(
        key=sk, n_components=T, mean_init=mean_init, beta=0, learning_rate=1,
        dof_offset=1, position_scale=T, position_event_shape=(3, 1))
    model = copy.deepcopy(prior_model)
    traj, t0 = [], time.perf_counter()
    ps = ss = cs = None
    for fi, x in enumerate(frames_n):
        model, ps, ss, cs = fit_gmm_step(prior_model, model, data=x,
                                         batch_size=SUBSAMPLE, prior_stats=ps,
                                         space_stats=ss, color_stats=cs)
        traj.append(dict(frame=fi, n_used=int(
            (model.prior.alpha > model.prior.prior_alpha.min()).sum())))
    secs = time.perf_counter() - t0

    # E[c|s] on held-out points from (mu, si, alpha), the 2D renderer's math on 3D points
    mu, si = model.denormalize({"stdevs": jnp.ones(6), "offset": jnp.zeros(6)})  # stay normalized
    mu, si, al = np.asarray(mu), np.asarray(si), np.asarray(model.prior.alpha)
    xs_h, xc_h = heldout_n[:, :3], heldout_n[:, 3:]
    L = np.linalg.cholesky(si[:, :3, :3] + 1e-9 * np.eye(3))
    logdet = 2 * np.log(np.diagonal(L, axis1=1, axis2=2)).sum(1)
    dif = xs_h[:, None, :] - mu[None, :, :3]
    # per component: quad_nk = ||L_k^{-1}(x_n - mu_k)||^2
    quad = np.empty((xs_h.shape[0], T))
    for k in range(T):
        y = np.linalg.solve(L[k], dif[:, k, :].T)
        quad[:, k] = (y**2).sum(0)
    logw = np.log(al + 1e-300)[None, :] - 0.5 * quad - 0.5 * logdet[None, :]
    logw -= logw.max(1, keepdims=True)
    w = np.exp(logw)
    w /= w.sum(1, keepdims=True)
    pred = w @ mu[:, 3:]
    err_unit = (pred - xc_h) * color_std[None, :]
    mse = float((err_unit**2).mean())
    return dict(traj=traj, seconds=secs,
                n_used=traj[-1]["n_used"], heldout_color_mse=mse,
                heldout_point_psnr=float(-10 * np.log10(mse)))


def main():
    scenes = sys.argv[1:] or ["lego", "chair"]
    results = {}
    for scene in scenes:
        print(f"=== {scene} ===", flush=True)
        frames_n, heldout_n, data_params = load_frames(scene)
        color_std = np.asarray(data_params["stdevs"]).reshape(-1)[3:]
        dp_res, _, _ = fit_dp(frames_n, heldout_n, color_std)
        print("dp:", {k: v for k, v in dp_res.items() if k != "traj"}, flush=True)
        dp100_res, pvar, dp_pred = fit_dp(frames_n, heldout_n, color_std, alpha=100.0)
        print("dp(a=100):", {k: v for k, v in dp100_res.items() if k != "traj"}, flush=True)
        vb_res = fit_vbgs(frames_n, heldout_n, color_std)
        print("vbgs:", {k: v for k, v in vb_res.items() if k != "traj"}, flush=True)
        results[scene] = dict(dp=dp_res, dp_a100=dp100_res, vbgs=vb_res)

        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
        axes[0].plot([t["step"] for t in dp_res["traj"]],
                     [t["khat"] for t in dp_res["traj"]], label="DP-Splat K̂ (α=1)")
        axes[0].plot([t["step"] for t in dp100_res["traj"]],
                     [t["khat"] for t in dp100_res["traj"]], label="DP-Splat K̂ (α=100)")
        axes[0].plot([t["frame"] + 1 for t in vb_res["traj"]],
                     [t["n_used"] for t in vb_res["traj"]], label="VBGS n_used (per frame)")
        axes[0].set_xlabel("SVI step / frame"); axes[0].set_ylabel("effective components")
        axes[0].set_title(f"{scene}: effective components (T={T})", fontsize=10); axes[0].legend(fontsize=8)
        std_c = np.asarray(data_params["stdevs"]).reshape(-1)[3:]
        off_c = np.asarray(data_params["offset"]).reshape(-1)[3:]
        pvar_tr = pvar.sum(1)  # trace for display
        pvar_disp = np.clip(pvar_tr, None, np.quantile(pvar_tr, 0.99))  # tail-clipped
        sc = axes[1].scatter(heldout_n[:, 0], heldout_n[:, 1], c=pvar_disp, s=1,
                             cmap="inferno")
        axes[1].set_title("predictive color variance", fontsize=10)
        plt.colorbar(sc, ax=axes[1])
        pred_rgb = np.clip(dp_pred * std_c[None, :] + off_c[None, :], 0, 1)
        axes[2].scatter(heldout_n[:, 0], heldout_n[:, 1], c=pred_rgb, s=1)
        axes[2].set_title("DP-Splat($\\alpha$=100) E[c|s]", fontsize=10)
        for ax in axes[1:]:
            ax.set_aspect("equal")
        fig.tight_layout()
        fig.savefig(REPO / "figures" / f"phase3_{scene}.png", dpi=160)

    (REPO / "experiments" / "out").mkdir(exist_ok=True)
    (REPO / "experiments" / "out" / "phase3_scenes.json").write_text(json.dumps(
        {"config": dict(T=T, n_frames=N_FRAMES, subsample=SUBSAMPLE, alpha=ALPHA),
         "results": results}))
    print(json.dumps({s: dict(dp_khat=r["dp"]["khat"],
                              dp_point_psnr=r["dp"]["heldout_point_psnr"],
                              vbgs_n_used=r["vbgs"]["n_used"],
                              vbgs_point_psnr=r["vbgs"]["heldout_point_psnr"])
                      for s, r in results.items()}, indent=2))


if __name__ == "__main__":
    main()
