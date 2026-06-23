# Mathematical reference — Sparse manifold-guided flow

Self-contained derivation backing the stages. Notation: feature space `X = ∏_j X_j ⊆ [0,1]^d`
(MinMax-scaled), labels `Y ∈ {0,1}`, data law `P`, target class `t`, current class `c=1−t`,
factual `x0` with `ŷ(x0)=c`. Actionable set `A`, immutable `I`. Feasible slice
`F(x0) = {x : x_I = (x0)_I}`. Discriminator `h_t(x) = P̂(Y=t | x)`.

---

## §1. The counterfactual posterior

Unify validity + plausibility + sparsity into one measure on `F(x0)`:

```
π_β(x) ∝ p(x | Y=t)^α · exp( β·[h_t(x) − τ] ) · exp( −λ·‖x − x0‖_0 ),   x ∈ F(x0)
         └ plausibility ┘   └ validity tilt ┘      └ sparsity prior ┘
```

- `argmax_x π_β` = the single best counterfactual (what greedy approximates).
- `x ~ π_β` = a diverse, plausible recourse (Stage 5).
- `β → ∞` = hard flip constraint `h_t ≥ τ`; `α` tunes manifold adherence; `λ` tunes sparsity.

This strictly generalizes the predecessor target (P): `min ‖x−x0‖_0 s.t. h_t≥τ, x∈M_ρ`
is the `β→∞`, `argmax` reading.

---

## §2. TabPFN is a joint-score oracle (the enabling lemma)

For any joint density, `log p(x) = log p(x_j | x_{−j}) + log p(x_{−j})`, and the second
term is independent of `x_j`. Therefore

```
[ ∇_x log p(x | t) ]_j  =  ∂/∂x_j log p( x_j | x_{−j}, Y=t )
```

— the j-th joint-score component equals the derivative of the **conditional** log-density,
which is exactly what `sampler.predictive_distribution(x, target_col=j, fixed_target=t)`
returns (a `FullSupportBarDistribution`). Stacking over `j` yields the full class-conditional
score field `s_t(x) = ∇_x log p(x|t)` at no extra model.

**The piecewise-constant obstacle.** The bar density is constant within each bucket ⟹ the
naive `∂/∂x_j log p` is 0 inside a bucket and undefined at borders. Usable estimators
(Stage 1 selects by cosine-accuracy vs a numerical KDE-score on MOONS):

- **mean-shift (primary):** `s_j ≈ (μ_j − x_j)`, `μ_j = E[x_j | x_{−j}, t]` (bar mean via
  `mean_of_prediction`). For near-Gaussian conditionals `score = (μ−x)/σ²`, so this is the
  score up to a positive scale — and it is robust to the bucketization.
- **finite-difference:** `s_j ≈ [log p(x_j+ε) − log p(x_j−ε)] / (2ε)` from the bar log-prob
  at shifted query values (clamp to `[0,1]`).
- **smoothed-derivative:** differentiate a smooth interpolant of the bucket density (fallback
  if neither above clears the cosine gate).

**Discrete columns.** Classifier-routed (low-cardinality integer) columns have no continuous
gradient; they evolve by **Gibbs** resampling `x_j ~ p(x_j | x_{−j}, t)`. The class contrast
for selecting which discrete col to jump reuses `class_conditional_shift` (TV distance).

---

## §3. The flow update (proximal / IHT Langevin)

Work in `δ = x_A − (x0)_A` (immutables vanish ⟹ actionability exact-by-construction). Smooth
part of `−log π_β`:

```
J(δ) = α·log p(x0⊕δ | t) + β·σ( κ·(h_t − τ) )
```

Update (continuous actionable coords):

```
δ_{k+1} = H_B( δ_k + η_k·[ α·s_t(x_k)_A + β·σ'(κ(h_t−τ))·∇h_t(x_k)_A ] + sqrt(2η_k/β)·ξ_k )
```

- `H_B` = hard-threshold keeping the **B largest-|·|** coords (the L0 prox); B annealed `|A|→B_min`.
- `s_t` = generative drift (§2). `∇h_t` = discriminative drift; for `LogisticRegression` the
  logit gradient is the constant `coef_[0]` (cheap, exact).
- `ξ_k ~ N(0,I)`; `noise=0` ⟹ deterministic flow (Stages 2–4). `β_k↑` anneals (Stage 5).
- Discrete coords: propose `gibbs_proposal` for the max-divergence discrete col; accept if it
  doesn't lower `h_t` (deterministic) / Metropolis under the chain (annealed).

**The three predecessor seams as corners of this one update:**

| Predecessor mechanism | Flow special case |
|---|---|
| `prob_ascent` selector | discriminative drift only (`α=0, β>0`) |
| `class_divergence` selector | generative drift only (`α>0, β=0`) |
| near-MAP commit (`t≈1e-9`) | `β→∞`, `noise→0` |
| static kNN context | `context_refit=0` |
| forward selection (one feat/step) | `H_B` with B incremented by 1, no joint gradient |

---

## §4. kNN context as moving-bandwidth local score

Restricting context to the k nearest neighbours of the query makes `ŝ_t` a *local* estimator;
bias ~ `O(h²·κ(M))`, variance ~ `O(1/k)`, `κ` = local manifold curvature. HELOC is high-curvature
(bias dominates ⟹ small-k kNN wins, `iterative-greedy-cf` Decision #14); MOONS is low-curvature
(variance dominates ⟹ larger random context fine). **Path-adaptive** refit (Stage 4) re-anchors
the kNN around the moving iterate `x_k`, keeping bias low along the whole path, not just at `x0`.

---

## §5. Annealing + recourse distribution (Stage 5)

Finite-β Langevin with `β_k↑` samples `π_β` rather than seeking its mode, so repeated chains
(distinct seeds) yield diverse valid recourses — turning L0-minimizer non-uniqueness into a
measured **diversity** (mean pairwise L2) + **coverage** (fraction with ≥1 valid CF). Under a
log-Sobolev inequality for `π_β` on `F(x0)`, annealed Langevin mixes in `Õ(1/ε)` — giving a
coverage statement instead of a bare validity rate.

---

## §6. Why this should lift the MOONS plateau (the gate)

Greedy's myopia is governed by the **submodularity ratio** `γ` of the validity-coverage
function `F(S)=h_t(x0⊕g_S)`: greedy cover obeys `|S_greedy| ≤ (|S*|/γ)(1+ln(·))`. MOONS'
two-moons flip is an interaction-dominated (XOR-like) move ⟹ single-coordinate marginal gains
vanish while a joint move flips ⟹ `γ→0` ⟹ greedy provably plateaus (≈0.70–0.82). The flow's
drift `∇J` **couples all actionable coords simultaneously**, so a B=2 step expresses the 2-D
interaction greedy cannot — the mechanism reason the flow is expected to beat the plateau.
RSC of `J` on `≤2B`-sparse displacements (not `γ`) controls the flow, and does **not** vanish
on interaction flips. (If the flow still does not beat 0.82, index Decision #9: report it with
failure_rate + Stage-1 score accuracy to attribute cause — myopia vs score bias vs landscape.)
