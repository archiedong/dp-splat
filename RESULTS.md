# RESULTS.md — result tables

All commands + configs in REPRO.md; raw records in `experiments/out/*.json`.
Dates are run dates; 3 seeds unless noted. "PASS/FAIL" refers to the pre-specified validation criteria (paper appendix).

## Phase 1 (2026-07-06)

### VBGS baseline reproduction (2D, Tiny-ImageNet valid 10k, K=2000)
| metric | value |
|---|---|
| PSNR | 22.33 ± 3.07 dB |
| components used | 1857 ± n/a of 2000 |
| runtime | 0.157 s/image (CPU, float64) |

### F1 — ELBO monotonicity (20 seeds × 4 variants) — **PASS**
Worst relative ELBO decrease: −9.7e-16 (one float64 ULP, dir variant; the other three are
exactly 0). DP variants converge in ~5–16 iterations; Dirichlet variants ~30–73 (median ≈38).

### F2 — K̂ recovery grid (2D synthetics; T = 3·K_true; α-sweep)
Headline: recovery within ±1 with regime-appropriate
α — α=0.1 ↔ K_true=3 (exact 3.0 at ALL N incl. 1e5); α=1 ↔ K_true=10 (exact at N≤1e4, 11.0
at 1e5); α=5 ↔ K_true=30 (exact at N=1e4; 28.3 at 1e3, 32.3 at 1e5). No single α covers all
regimes; dp(learn) tracks dp(α=1) cell-for-cell; **DP inflates slower with N than
sparse-Dirichlet at every matched cell** (e.g. 11 vs 21 at K_true=10, N=1e5).

### Natural-image acceptance + deconfound (12 images × α ∈ {1,100}, T=2000) — **PASS (24/24)**
| α | K̂ | vs VBGS@K̂ (single-pass) | vs CONVERGED dir@K̂ | vs VBGS@2000 |
|---|---|---|---|---|
| 1 | 9–17 | +2.80 [+0.64,+4.77] | −0.01 [−0.77,+0.77] | −6.47 mean |
| 100 | 29–53 | +2.70 [+0.51,+5.46] | +0.17 [+0.02,+0.70] | −4.79 mean |

**Deconfound:** against an equally CONVERGED fixed-K
Dirichlet at the same K̂, the DP advantage vanishes — the +2.7 dB was convergence, not the
prior. The DP prior's contribution = complexity selection + uncertainty machinery. Earlier
3-image α∈{10,1000} sweep (superseded): same VBGS@K̂ advantage, K̂ 22–33.
K̂ stays two orders below T=2000 at all α: CAVI component death dominates the prior.

## Phase 2 (2026-07-06)

### F3 — truncation sweep (K_true=10, N=2e4, α=1)
| T | held-out LL/pt (nats) | K̂ | rigorous bound 2N(α/(1+α))^{T−1} | IJ 4Ne^{−(T−1)/α} (large-α approx) |
|---|---|---|---|---|
| 10 | −0.74 ± 0.56 (bimodal: 1 good / 2 stuck seeds) | 8.7 | 8e1 (vacuous) | ≈10 (vacuous) |
| 25 | 0.051 ± 0.003 | 10.3 | 2.4e−3 | 3e−6 |
| 50 | 0.051 ± 0.003 | 10.3 | 7e−11 | 4e−17 |
| 100 | 0.051 ± 0.003 | 10.3 | 6e−26 | 8e−39 |
| 200 | 0.051 ± 0.003 | 10.3 | 5e−56 | 3e−82 |

Truncation is a non-issue for T ≳ 2.5·K̂; the T=K_true edge shows CAVI local optima, not
truncation error per se. NOTE: the widely quoted 4Ne^{−(T−1)/α} form is Ishwaran–James'
large-α approximation and is **anti-conservative at α=1** (it undershoots the exact bound);
the rigorous small-α form is 2N(α/(1+α))^{T−1} — both plotted in F3.

### F5 — calibration of predictive color variance 
ECE ∈ {0.0013, 0.0030, 0.0030}; binned curve sits on the diagonal across the full variance
range (figures/f5_calibration.png); global mean predicted variance vs realized MSE within ~2%.

### F6 — mixing-measure W₂ contraction
| N | dp(α=1) | sparse_dir(e₀=0.01) |
|---|---|---|
| 1e3 | 1.56 ± 0.15 | 1.54 ± 0.09 |
| 1e4 | 0.77 ± 0.03 | 0.76 ± 0.04 |
| 1e5 | 0.46 ± 0.09 | 0.52 ± 0.08 |

Contraction ≈ power law; dp better at large N (its extra components carry less weight).

### F4 — K̂ growth with N (3D, K_true=10, T=60; full sweep)
| N | dp(α=0.1) | dp(α=1) | dp(α=5) | dp(learn) | sp e₀=0.1 | sp e₀=0.01 | sp e₀=0.001 |
|---|---|---|---|---|---|---|---|
| 1e3 | 4.0 | 10.0 | 10.0 | 10.0 | 10.0 | 10.0 | 10.0 |
| 1e4 | 3.7 | 9.7 | 10.0 | 9.7 | 10.0 | 10.0 | 10.0 |
| 1e5 | 4.0 | 11.7 | 16.3 | 11.7 | 22.0 | 22.0 | 22.0 |
| 1e6 | 4.7 | 13.3 | 32.0 | 13.3 | **60.0 (=T)** | **60.0** | **60.0** |

**Under CAVI the textbook asymptotic ordering inverts**: DP shows the slow (log-like)
Miller–Harrison inflation; the sparse overfitted Dirichlet — which Rousseau–Mengersen theory
says empties — saturates the full truncation by N=1e6, **identically across two decades of
e₀ (0.1/0.01/0.001)** ⇒ the over-splitting is likelihood/CAVI-dynamics-driven, not
prior-driven (verified: posteriors differ by exactly Δe₀; K̂ doesn't). dp(α=0.1) over-shrinks
(K̂≈4 ≪ 10) — the regime pattern from F2 replicated in 3D. dp(learn) tracks dp(α=1)
cell-for-cell. 

### SVI scaling (B=2¹⁶, T=60, 300 steps, local CPU)
| N | s/step | total | K̂ |
|---|---|---|---|
| 1e5 | 0.05 | 14 s | 13 |
| 1e6 | 0.04 | 12 s | 13 |
| 3e6 | 0.04 | 12 s | 13 |
| 1e7 | 0.04 | 12 s | 14 |

Per-step cost is N-independent (as designed); **N=10⁷ is trivially feasible locally with
SVI** — the H100 is needed for Phase 3 rasterization, not for the scaling experiment.

## Phase 3 — local scope (2026-07-06)

Protocol: T=K=2000 budget, 40 training frames × 20k subsampled points, DP-Splat = 10-epoch
SVI over the frame stream, VBGS = their exact continual protocol (one-shot conjugate
accumulation). Metric = held-out-frame point-level color prediction (E[c|s] vs truth, original
[0,1] units) — a rasterizer-free proxy, NOT comparable to rasterized PSNR.

### Final (final protocol; T=2000, 40 frames × 20k pts, 10-epoch SVI)
| scene | model | effective K | point-PSNR (dB) | held-out LL/pt | fit s |
|---|---|---|---|---|---|
| lego | DP-Splat α=1 | 20 | 14.56 | −1.65 | 223 |
| lego | **DP-Splat α=100** | **234** | **17.42** | **+1.03** | 216 |
| lego | VBGS (K=2000) | 1374 | 17.13 | n/a | 36 |
| chair | DP-Splat α=1 | 19 | 14.55 | −4.68 | 143 |
| chair | DP-Splat α=100 | 181 | 16.03 | −1.95 | 144 |
| chair | VBGS (K=2000) | 1373 | 16.17 | n/a | 35 |

**Headlines:** (1) on lego, DP-Splat(α=100) **beats** VBGS's held-out color prediction
(+0.29 dB) with **5.9× fewer** components; on chair it matches within 0.14 dB with 7.6×
fewer. (2) **Single-pass arm** (`phase3_single_pass.json`): SINGLE-pass DP-Splat (23–25 s
≈ VBGS's 35 s) already reaches 17.25/16.03 dB — extra epochs mainly prune K̂ (276→234),
not improve prediction. (3) **Complexity ordering correct**: lego K̂=234 > chair K̂=181 at
identical settings (n=2, reported as such). (4) K̂ on scenes rises far past the ~30 image
saturation (data-complexity-limited, not a ceiling). (5) **Misspecified calibration** (`phase3_scene_calibration.json`):
ECE 0.011 (lego, +18% over-estimate) / 0.016 (chair, −21% under-estimate) vs 0.003 on
model-matched synthetics — degraded but informative; reported in the paper.

### F7 cite-only reference row (rasterized, full-res, NOT comparable to the local proxy)
3DGS (Kerbl et al. 2023, "Ours-30K", 200–500k Gaussians): chair 35.83 dB, lego 35.78 dB,
NeRF-synthetic average 33.32 dB. To be used in the H100 F7 frontier figure; verified 2026-07-06.
