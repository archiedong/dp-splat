# DP-Splat

Bayesian nonparametric complexity control for Gaussian splatting: a (truncated) stick-breaking
Dirichlet-Process prior — plus a sparse overfitted-Dirichlet variant — over mixture weights in a
VBGS-style conjugate variational Gaussian-splatting model. The number of occupied components adapts
to scene complexity; all updates remain closed-form CAVI; ELBO convergence, a truncation-error
bound, and an honest treatment of what the posterior on K estimates.

The model and update equations implemented here are exactly those stated in the paper
(Sections 3--4); every numerical stabilization or deviation is documented in `LIMITATIONS.md`.

## Project documents

| File | Purpose |
|---|---|
| `REPRO.md` | Exact commands to reproduce every figure / result |
| `CODEMAP.md` | Map of the VBGS codebase: which functions we touch |
| `LIMITATIONS.md` | Every model-vs-deployment gap and numerical stabilization |
| `RESULTS.md` | Result tables (Phase 3) |

## Layout

```
src/dp_splat/
  priors.py        # stick-breaking, sparse-Dirichlet, Dirichlet (weight_prior switch)
  niw.py           # NIW updates + expected log-density identities
  cavi.py          # full-batch loop, ELBO terms (one function per term)
  svi.py           # natural-gradient stochastic variant (natural-space blend)
  predictive.py    # mixture-of-Students predictive: held-out LL, Var[c|s]
  prune.py         # effective-K, thresholding
  render.py        # §3.7 parameter map onto the VBGS renderer
tests/             # NumPy oracles, MC-vs-analytic ELBO, monotonicity, SVI-equivalence,
                   # VBGS cross-code regression (45 tests)
experiments/       # config-driven scripts: figures F1–F6 (F7/F8 H100-deferred), phase0/1/3
                   # comparisons, SVI scaling, scooping check, synthetic generators
third_party/vbgs/  # VBGS baseline, pinned @ 2ae3f4be — VERSES Academic Research License,
                   # unmodified, import-only external baseline
```

## Status

All experiments reported in the paper (F1–F6, image and scene comparisons, SVI scaling)
are complete and reproducible on a laptop CPU. See `REPRO.md` for commands and `RESULTS.md`
for tables. Rasterized novel-view benchmarks are scoped to follow-up work (GPU rasterizer).

## License / acknowledgment

`src/`, `tests/`, `experiments/` are MIT-licensed (see `LICENSE`).
`third_party/vbgs` is © 2024 VERSES AI, Inc., under the VERSES Academic / Nonprofit Research
License (see `third_party/vbgs/LICENSE.txt`); it is used here for academic research only and
is excluded from release artifacts. Any publication from this project includes: "The Software
used in this research was created by VERSES, Inc. © 2024 VERSES AI, Inc."
