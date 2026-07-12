# LIMITATIONS.md — model-vs-deployment gaps and numerical stabilizations

Per the paper's scope statement and numerical policy: every gap between the point-data mixture model and the deployed
rendering pipeline noticed during implementation, and every numerical stabilization, is logged here.

## Model vs deployment (scope-statement material)

- L1 (2026-07-06, from the paper's model statement): the likelihood is a mixture density on colored points; the
  splatting rasterizer (occlusion, alpha compositing) is a deployment map outside the likelihood.
  All theory in the paper is for the point mixture.

## Numerical stabilizations

- N1 (2026-07-06, `src/dp_splat/niw.py:_chol`): scale-aware ridge `eps(dtype) · tr(Ψ)/D · I`
  added to Ψ_k inside Cholesky factorizations only — never written back into stored parameters
  (numerical policy: ridge only inside factorizations). Invisible at working precision (verified: oracle tests agree to
  rtol 1e-9 in float64); guards near-singular Ψ_k when N_k ≈ 0.
- N2 (2026-07-06, `src/dp_splat/niw.py:soft_stats`): x̄_k division guarded by max(N_k, 1e-32).
  Exact, not approximate: every downstream use of x̄_k is multiplied by N_k (paper update equation), so
  the guarded value never contributes when N_k = 0 (documented in the docstring).
- N3 (2026-07-06): tests run in float64 (`tests/conftest.py`) per the numerical policy (Beta/digamma
  path); experiment scripts choose dtype explicitly.
- N4 (2026-07-06, from the final verification pass): `soft_stats` computes the scatter S via the
  one-pass uncentered identity Σ r xxᵀ − N_k x̄x̄ᵀ (exactly equivalent to the centered two-pass
  sum; verified to 5e-13 vs oracle). Caveat: cancellation-prone in float32 when the data mean
  is large vs the spread (e.g. raw pixel coordinates) — normalize/center data before fitting
  (as VBGS does), or switch to the centered two-pass form if float32 residuals ever matter.
- N5 (2026-07-06, from the final verification pass): the `_chol` ridge (N1) affects not only solves
  but also the log-determinants entering eq. (4a) and the Wishart normalizers — a slight
  stretch of the "ridge only inside linear solves" policy. At eps-scale it is invisible
  (all oracle/MC tests pass at tight tolerance); recorded for transparency.
- N6 (2026-07-06, from the final verification pass): `svi.niw_from_natural` recovers Ψ = n4 − κ·m·mᵀ — a
  rank-one subtraction that is catastrophically cancellative in float32 for uncentered data
  (stress test: 100% relative error at κ=5000, means ~100). All tests/experiments run
  float64 and fit normalized data, so results are unaffected; float32/GPU deployments must
  center data or keep the NIW blend in float64. Second float64-critical path beyond the
  Beta/digamma note.
- N7 (2026-07-06, verification-pass fix): SVI states now carry N/|B|-scaled responsibilities so that
  soft counts / K̂ / render gating are on the full-data scale (the paper's Sec. 4 semantics).
  Scene trajectories produced before this fix under-reported K̂ and were re-run.

## Environment / hardware

- E1 (2026-07-06, amended): local dev machine is Apple-Silicon macOS (arm64) —
  JAX CPU only. All unit tests, tiny-problem oracles, AND the N=10⁷ SVI scaling run locally
  (svi_scaling.json: 12 s at N=1e7); H100 (CUDA) is required only for Phase 3 rasterization
  (F7/F8), full-scale scene fits, and the paper's absolute wall-clock numbers.
- E2 (2026-07-06): hydra-core 1.3.4 crashes on Python 3.14 (argparse LazyCompletionHelp →
  "badly formed help string"), so VBGS's hydra entry points don't run on 3.14; our repro scripts
  bypass hydra with the YAML defaults hardcoded. GPU box pinned to Python 3.12 (environment.yml).
- E3 (2026-07-06): VBGS 3D rendering imports the INRIA diff-gaussian-rasterization (CUDA+torch)
  at module import time (vbgs/render/volume.py) — 3D renders impossible locally, and the render
  env is separate from the jax env even on the GPU box (dependency conflict per VBGS README).
- E4 (2026-07-06): Apple-GPU (jax-metal 0.1.1, M5 Max) evaluated and rejected: the Metal plugin
  is incompatible with current JAX (crashes at jax.random.PRNGKey), is officially experimental,
  and lacks float64 (needed for the Beta/digamma ELBO path). Local compute = CPU JAX
  (multi-threaded, ~0.2 s per 64×64 image at K=2000); scale runs = H100/CUDA.
