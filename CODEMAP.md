# CODEMAP.md — map of the VBGS codebase (pinned @ 2ae3f4be, 2024-11-20)

Produced 2026-07-06; every function/line claim below was independently verified against
the source. Paths relative to `third_party/vbgs/`.

## The one-sentence architecture

VBGS is conjugate exponential-family VI: a generic `Conjugate` class holds natural parameters and
implements the universal CAVI update `η = η₀ + lr·Σₙ rₙₖ T(xₙ)`; the Dirichlet weight update and
the NIW component update are both instances of it, and the weight prior is consumed
**polymorphically** through exactly three methods — which is where DP-Splat plugs in.

## Where the math lives

| CAVI quantity (paper Sec. 4) | Function | Location |
|---|---|---|
| Generic natural-param update (Dirichlet **and** NIW, eqs. 1–2) | `Conjugate.update_from_statistics` | `vbgs/vi/conjugate/base.py:370` |
| Weight-prior class (q(π), Dirichlet) | `Multinomial` | `vbgs/vi/conjugate/multinomial.py` |
| E[log π_k] (eq. 3, Dirichlet form) | `Multinomial.log_mean` | `vbgs/vi/conjugate/multinomial.py:282` |
| same quantity, duplicated for the generic ELL machinery | `Multinomial.expected_posterior_statistics` | `vbgs/vi/conjugate/multinomial.py:188` |
| Weight-prior ELBO term KL(q(π)‖p(π)) | `Multinomial.kl_divergence` | `vbgs/vi/conjugate/multinomial.py:235` |
| E[π_k] (rendering opacity/weights) | `Multinomial.mean` | `vbgs/vi/conjugate/multinomial.py:272` |
| NIW class (q(μ,Σ) per component) | `MultivariateNormal` | `vbgs/vi/conjugate/mvn.py` (natural-param map :286, E-stats :317, E[Σ] :445, KL :498) |
| Expected Gaussian log-density (eq. 4) | `Conjugate.expected_log_likelihood` | `vbgs/vi/conjugate/base.py:199` |
| Responsibilities (eq. 5) + softmax/logsumexp | `fit_gmm` / `compute_elbo_delta` | `vbgs/model/train.py:60,94` (softmax :129, logsumexp :119); stable ops in `vbgs/vi/utils/math.py:123,132` |
| Weight-update call sites | `model.mixture.prior.update_from_statistics(...)` | `vbgs/model/train.py:83` (full-batch), `:209` (continual/batched) |
| E[log π] call sites | `model.mixture.prior.log_mean()` | `vbgs/model/train.py:68,113` |
| N_k statistics assembly | `Mixture._to_stats` | `vbgs/vi/models/mixture.py:131` |

**No full ELBO exists in VBGS** — only per-part KLs and a `logsumexp(logprob)` "elbo delta".
Our `src/dp_splat/cavi.py` implements the complete §3.5 ELBO regardless.

Natural-param convention gotcha: `Multinomial.to_natural_params` stores `eta_1 = α` (the
concentration itself, **not** α−1); `N_k` is recoverable as `alpha − prior_alpha`.

## Model assembly & training

- `DeltaMixture` (`vbgs/model/model.py:29`): **two conditionally independent modalities given one
  shared z** — spatial MVN with NIW prior + color MVN with `fixed_precision=True` (a "delta"
  likelihood; color covariance is **never learned**; rendering uses identity for the color block).
  log ρ_nk = spatial ELL + color ELL + E[log π_k] — matches paper Sec. 3's factorization, but see
  the paper's discussion (paper Sec. 3 wants NIW on color too).
- Factories: `get_image_model` (`scripts/model_image.py:27`, prior built at :104) and
  `get_volume_delta_mixture` (`scripts/model_volume.py:27`, prior at :101). Both hard-code
  `Multinomial(event_shape=(K,), initial_count=1/K)` — i.e. **VBGS's shipped default is already a
  sparse symmetric Dirichlet, α₀ = 1/K** (see the paper's discussion).
- Training is pure conjugate CAVI, no gradients. Full-batch: `fit_gmm` (`train.py:60`), E-step →
  exact Dirichlet+NIW updates; `n_iters` in configs (default **1 pass**; >1 = iterated CAVI).
  Continual: `fit_gmm_step` (`train.py:133`) — responsibilities from a **frozen** copy of the
  initialized model, sufficient statistics accumulated across batches and frames, zero-padded last
  batch for JIT.
- `update_from_statistics` supports `lr`/`beta` (damped blend with old posterior — the SVI-style
  knob; reduces to exact CAVI at lr=1, β=0).
- Init: `random_mean_init` (`vbgs/model/utils.py:37`), uniform in [−1.7, 1.7], colors zeroed.
- `reassign.py`: heuristic recycling of dead components (α ≤ prior_alpha.min()) — teleports their
  means onto high-loss points via `eqx.tree_at`; hardcoded 3D slicing (volume-only). **This is the
  ADC-style heuristic that a DP/sparse prior should subsume** — Phase 2 ablation material.

## Rendering & data

- **2D**: pure JAX, no external deps. `model.denormalize` (`model.py:50`) → `render_img` /
  `render_patch` (`vbgs/render/image.py:64/:44`): per-pixel responsibilities over the spatial
  2D marginal × expected color = posterior-predictive E[c|uv]. Weights passed as raw `prior.alpha`
  (ratios only). **Ideal for Phase 1.**
- **3D**: `vbgs/render/volume.py` imports the INRIA `diff-gaussian-rasterization` (CUDA + torch,
  separate env per README/`install_deps.sh`) at module import — unusable locally; H100 box only.
  `vbgs_model_to_splat` (:108): xyz = μ[:3], color = RGB2SH(μ[3:]) (SH deg 0), scale/rotation via
  Cholesky of E[Σ] (:74), **opacity = hard binary (α_k > 1e-6)** — mixture weight only gates
  existence; paper Sec. 4 (rendering map) says map exactly as VBGS does, so this is the mapping we inherit.
- Data: images = HF `Maysee/tiny-imagenet` valid split (10k × 64×64, small). Blender objects
  require per-frame **depth** PNGs; Habitat rooms via Dust3r preprocessing (heavy). Configs in
  `scripts/configs/{imagenet,blender,habitat}.yaml` (README's `scripts/config/` is a typo).
- Metrics: `vbgs/metrics.py` — `calc_mse`, `calc_psnr` (no SSIM in repo).

## The ≤3 touch points for `weight_prior ∈ {dp, sparse_dir, dir}`

1. **`vbgs/vi/conjugate/multinomial.py` — subclass `Multinomial` (e.g. `StickBreakingMultinomial`)
   in our own `src/dp_splat/priors.py`**, overriding `log_mean` (stick-breaking E[log π_k] =
   E[log v_k] + Σ_{j<k} E[log(1−v_j)], paper update equation (3), `expected_posterior_statistics` (must stay
   consistent with `log_mean`), and `kl_divergence` (Σ of T−1 Beta KLs vs Beta(1, α₀)).
   The stored natural params (α = α₀ + N_k) already contain everything needed: γ_{k,1} = 1 + N_k,
   γ_{k,2} = E[α] + Σ_{j>k} N_j are reverse-cumulative sums of N_k = α − prior_alpha — so the
   generic `update_from_statistics` needs **no change** (paper update equation (2) falls out of it).
2. **Factory construction sites** `scripts/model_image.py:104` / `scripts/model_volume.py:101`
   (or our own factory in `src/dp_splat/`): instantiate the chosen prior class; `sparse_dir` is
   literally `initial_count=e₀`; `dir` = VBGS baseline (note: VBGS ships with e₀ = 1/K, not 1).
3. **`vbgs/model/train.py` — likely zero changes** (all weight-prior access is polymorphic through
   the three methods above); only needed if we add α-learning (paper update equation (6), which introduces a
   Gamma(w₁,w₂) global that no VBGS structure holds.

`src/dp_splat/` (subclass/wrap, import vbgs unmodified), not to edit `third_party/vbgs`.
