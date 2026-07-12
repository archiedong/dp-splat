"""Calibration of Var[c|s] on REAL (misspecified) scene data.

Same 10-bin protocol as F5, applied to the held-out frame of each Blender scene under the
Phase 3 DP-Splat fit (alpha=100). Reported honestly whichever way it comes out.

Run: ~/.venvs/dp-splat/bin/python experiments/phase3_scene_calibration.py [scene ...]
"""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "third_party" / "vbgs"))
sys.path.insert(0, str(REPO / "experiments"))

import vbgs  # noqa: F401
import jax.numpy as jnp
import numpy as np

from dp_splat import cavi
from dp_splat.predictive import conditional_color_moments
from phase3_scenes import ALPHA, N_FRAMES, PASSES, SEED, SUBSAMPLE, T, fit_dp, load_frames

N_BINS = 10


def main():
    scenes = sys.argv[1:] or ["lego", "chair"]
    out = {}
    for scene in scenes:
        frames_n, heldout_n, data_params = load_frames(scene)
        color_std = np.asarray(data_params["stdevs"]).reshape(-1)[3:]
        # refit at alpha=100 (the headline scene setting)
        _, pvar_diag, mean_norm = fit_dp(frames_n, heldout_n, color_std, alpha=100.0)
        # exact per-dim conversion to original color units: var_d scales by std_d^2
        xc_h = heldout_n[:, 3:]
        err2 = (((mean_norm - xc_h) * color_std[None, :]) ** 2).sum(1)
        pred_var = (pvar_diag * (color_std**2)[None, :]).sum(1)
        qs = np.quantile(pred_var, np.linspace(0, 1, N_BINS + 1)); qs[-1] += 1e-12
        bins, ece = [], 0.0
        for b in range(N_BINS):
            m = (pred_var >= qs[b]) & (pred_var < qs[b + 1])
            if m.sum() == 0:
                continue
            bins.append(dict(bin=b, n=int(m.sum()), pred_var=float(pred_var[m].mean()),
                             mse=float(err2[m].mean())))
            ece += m.mean() * abs(err2[m].mean() - pred_var[m].mean())
        out[scene] = dict(ece=float(ece), mean_pred_var=float(pred_var.mean()),
                          mean_sq_err=float(err2.mean()),
                          corr=float(np.corrcoef(pred_var, err2)[0, 1]), bins=bins)
        print(scene, {k: round(v, 4) for k, v in out[scene].items() if k != "bins"},
              flush=True)
    (REPO / "experiments" / "out" / "phase3_scene_calibration.json").write_text(
        json.dumps({"config": dict(T=T, alpha=100.0, n_frames=N_FRAMES,
                                   subsample=SUBSAMPLE, passes=PASSES, seed=SEED,
                                   n_bins=N_BINS), "scenes": out}))


if __name__ == "__main__":
    main()
