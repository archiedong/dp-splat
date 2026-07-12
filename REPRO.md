# REPRO.md — exact commands to reproduce every result

Policy: every figure/number in the paper gets its command logged here.

## Environment (Phase 0, 2026-07-06)

Local dev box: macOS arm64 (Apple Silicon), Python 3.14.6, JAX 0.10.2 (CPU). VBGS pinned at
`2ae3f4be` (cloned 2026-07-06, `third_party/vbgs`, treated as read-only).

```bash
python3.14 -m venv ~/.venvs/dp-splat        # venv kept OUTSIDE iCloud (sync churn)
~/.venvs/dp-splat/bin/pip install jax datasets equinox hydra-core jaxtyping matplotlib \
    multimethod opencv-python pillow rich tqdm scipy pot pytest
# vbgs is NOT pip-installed; scripts put third_party/vbgs (+ its scripts/) on sys.path.
```

Known env facts (also in LIMITATIONS.md / environment.yml):
- hydra-core 1.3.4 is broken on Python 3.14 (argparse `LazyCompletionHelp` crash) → VBGS's own
  hydra entry points (`scripts/train_image.py` etc.) cannot run on 3.14. Our repro scripts
  hardcode the YAML defaults and bypass hydra; the H100 box should use Python 3.12.
- 3D volume rendering needs the INRIA rasterizer (CUDA + torch, separate env) — not runnable
  locally; 3D demo reproduction deferred to the GPU box.

## Phase 0 — VBGS baseline reproduction

### 2D image benchmark (Tiny-ImageNet valid, 10k images, 64×64, K=2000)

Exact pipeline and hyperparameters of `scripts/train_image.py` + `configs/imagenet.yaml`
(n_components=2000, init_random=True, dof=1.0, scale=null, lr=1.0, beta=0.0, n_iters=1, seed=0):

```bash
cd dp-splat
~/.venvs/dp-splat/bin/python experiments/phase0_baseline_image.py 10000 2000
# → experiments/phase0_baseline_image_results.json
```

Smoke run (3 images): per-image PSNR 21.3 / 33.5 / 24.3 dB, n_used ≈ 1860/2000, 0.2 s/image
after JIT warmup (first-image compile 4.7 s), CPU.

Full 10k run (2026-07-06, local CPU, Apple M5 Max, ~26 min wall-clock):
**PSNR 22.33 ± 3.07 dB**, mean n_used 1857/2000 components, 0.157 s/image.
Full per-image records: `experiments/phase0_baseline_image_results.json`.

Local GPU note: jax-metal 0.1.1 evaluated 2026-07-06 and rejected (crashes on current JAX,
no float64) — LIMITATIONS.md E4. CPU JAX is the local compute path.

Reference for tolerance: the VBGS paper (arXiv 2410.03592) reports image-benchmark PSNR only as
a curve over component counts (Fig. 2a) — no exact table. Acceptance = our K=2000 mean PSNR sits
on their curve (visual check against their Fig. 2a); the precise numeric repro targets are the paper's 3D tables
(Blender/Habitat), deferred to the GPU box.

## Phase 1 — DP-Splat core (2026-07-06)

### Test suite (47 tests)

```bash
cd dp-splat
~/.venvs/dp-splat/bin/python -m pytest tests/ -q
```

- `tests/oracle_numpy.py`: brute-force NumPy oracle, written directly from the model
  equations without reference to `src/` — update-equation ground truth.
- `tests/test_elbo_mc.py`: every ELBO term vs Monte Carlo from q (scipy sampling;
  1e5 samples scalar terms, 2e4 matrix terms, 5·SE tolerance).
- `tests/test_monotonicity.py`: ELBO monotone, 20 seeds × 200 iters × 4 variants (float64).
- Independent line-by-line check of the implementation against the model equations:
  no math errors; notes recorded in LIMITATIONS.md N4/N5.

### F1 — ELBO traces (20 seeds, 4 variants; K_true=10, N=10⁴, T=30)

```bash
~/.venvs/dp-splat/bin/python experiments/fig1_elbo_traces.py
# → experiments/out/f1_traces.json, figures/f1_elbo_traces.png
```

### F2 — K̂ vs K_true grid (K_true ∈ {3,10,30} × N ∈ {1e3,1e4,1e5}, T = 3·K_true, 3 seeds)

```bash
~/.venvs/dp-splat/bin/python experiments/fig2_khat_grid.py 3 0.1,1,5
# → experiments/out/f2_khat.json, figures/f2_khat_grid.png
# (second arg = dp alpha sweep; added after the single-alpha run exposed strong
#  alpha sensitivity)
```

Results (2026-07-06): F1 — ELBO monotone to float64 precision, worst relative decrease
−9.7e-16 across 20 seeds × 4 variants; DP variants converge in ~5–16 iterations vs ~30–73
(median ≈38) for Dirichlet variants. F2 (α-sweep, 162 fits, 4.4 min compute; earlier single-α pass and
diagnostics logged alongside): K̂ recovery is exact with regime-appropriate α (α=0.1 ↔
K_true=3 incl. N=10⁵; α=1 ↔ K_true=10; α=5 ↔ K_true=30 at N=10⁴); no single α covers all
regimes; learn_alpha tracks α=1 cell-for-cell (mode-following, E[α]→1.3–1.7); DP inflates slower
with N than sparse_dir/dir at every matched cell. Artifacts:
`experiments/out/f2_khat.json`, `figures/f2_khat_grid.png`.

### Phase 1 acceptance — natural-image PSNR vs VBGS at matched effective budget

```bash
for a in 1 100; do   # final 12-image run with the converged-dir@Khat deconfound arm
  ~/.venvs/dp-splat/bin/python experiments/phase1_image_vs_vbgs.py 12 2000 $a
done
# scene-side additional arms:
#   experiments/phase3_scene_calibration.py lego chair   -> misspecified calibration
#   phase3_single_pass.json produced via PASSES=1 driver (see paper Table 2)
# → experiments/out/phase1_image_vs_vbgs_a{1,10,100,1000}.json
# → figures/phase1_image_panel_a{...}.png
```

Results (2026-07-06): **acceptance PASS in all 12 cells** — DP-Splat exceeds matched-K VBGS by
+2.6..+5.7 dB (never below −0.5 dB). Second finding: K̂ saturates at ~30 components in α
(α=1 → K̂≈9–14 ≈ prior E[K]; α=1000 → K̂≈28–33 despite prior E[K]≈1600) → CAVI component-death
dynamics dominate the prior above α≈1; DP renders sit 3–7 dB below VBGS's full 2000-budget
run.  DP fit wall-clock: 13–52 s/image
(64×64, T=2000, float64, CPU).

## Phase 2 — statistical experiment suite (2026-07-06)

```bash
P=~/.venvs/dp-splat/bin/python
$P experiments/fig3_truncation.py     # F3: T-sweep + Ishwaran–James bound
$P experiments/fig4_khat_vs_n.py      # F4: K̂ growth curves (T3), CAVI<=1e5, SVI above
$P experiments/fig5_calibration.py    # F5: predictive-variance calibration
$P experiments/fig6_wasserstein.py    # F6: mixing-measure W2 (T4), needs `pip install pot`
$P experiments/svi_scaling.py         # SVI per-step cost vs N (up to 1e7, local CPU)
```

Results (2026-07-06, 3 seeds each unless noted):
- **F3**: held-out LL and K̂ flat for T ∈ {25,50,100,200} at K_true=10, N=2e4, α=1
  (K̂ = 10.3 ± 0.5); T=10 (=K_true) degrades via local optima (2/3 seeds — no slack for
  reordering). IJ bound overlay: exact MC form 2[1−E{(Σ_{k<T}π_k)^N}] + 4Ne^{−(T−1)/α} approx;
  bound form verified against Ishwaran & James 2001 (JASA); the
  bound is astronomically small for T ≥ 50 at α=1 — truncation is a non-issue at scene budgets).
- **F5**: predictive color variance is well calibrated on 3D synthetics with overlapping
  clusters: ECE ∈ {0.0013, 0.0030, 0.0030}, global mean predicted variance vs realized MSE
  within ~2% (e.g. 0.0706 vs 0.0691). Uses the mixture-of-Students predictive (paper Eq. 7).
- **F6**: W₂(fitted, true mixing measure) contracts: dp(α=1) 1.56→0.77→0.46 for
  N=1e3→1e4→1e5 (sparse_dir similar at small N, worse at N=1e5: 0.52 vs 0.46 mean — the
  over-split components carry weight off the true atoms).
- **F4 + SVI scaling**: complete — full tables in RESULTS.md (headline: under CAVI the
  asymptotic ordering inverts — DP log-like inflation, sparse-Dir truncation saturation at
  N=1e6 for both e₀=0.01 and 0.001; SVI per-step cost O(1) in N, N=1e7 in 12 s local CPU).
  F4 rerun 2026-07-06 with the full sweep arms (α ∈ {0.1,1,5}+learn, e₀ ∈ {0.1,0.01,0.001})
  after review. e₀-insensitivity verified genuine (posteriors differ by exactly Δe₀;
  ELBOs differ; K̂ identical).

## Phase 3 (local scope, 2026-07-06)

Data: nerf-synthetic test splits (with depth PNGs) from HF mirror
`pablovela5620/nerf-synthetic-mirror` → `~/dp-splat-data/blender/{lego,chair}/`
(kept outside iCloud; ~0.5 GB).

```bash
~/.venvs/dp-splat/bin/python experiments/phase3_scenes.py lego chair
```

Rasterized PSNR/SSIM (F7) and render panels (F8) require the INRIA CUDA rasterizer —
**H100-deferred** (LIMITATIONS.md E3). Local artifacts: K̂/n_used trajectories during
continual fitting, held-out point-level color prediction (both models, same conditional
formula), predictive-variance visualization on held-out views.

### 3D demos (Blender objects / Habitat rooms)

Deferred to H100 box: needs (a) CUDA jax, (b) separate torch+INRIA-rasterizer env for rendering,
(c) datasets: Blender NeRF-synthetic **with per-frame depth PNGs**; Habitat via Dust3r
preprocessing. Commands will be logged here when run.
