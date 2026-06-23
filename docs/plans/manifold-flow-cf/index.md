# Plan: Sparse Manifold-Guided Flow Counterfactuals with TabPFN (Unified Flow)

**Date**: 2026-06-23
**Branch**: `manifold-flow-cf` (create from `iterative-greedy-cf`)
**Predecessors**:
- [`iterative-greedy-cf`](../iterative-greedy-cf/index.md) (Stages 1–4 DONE) — provides `greedy.py`, the `ConditionalDensitySampler` with `predictive_distribution()` + kNN context, the discriminator oracle, the metrics harness, the Exp4/5/6 runners, and the headline result this plan extends: greedy + `prob_ascent` + `knn_both@256` salvages HELOC (validity 0.90, frac_oob 0.00, LOF 1.98) but **MOONS validity plateaus at ≈0.70–0.82** with ≈30% failure rate.
- [`zeroshot-tabpfn-cf`](../zeroshot-tabpfn-cf/index.md) (Stages 1–9 DONE) — original offline v2 checkpoints, sampler, discriminator, datasets, metrics.

**Goal**: Replace discrete one-feature-at-a-time greedy forward selection with a **single continuous mechanism** — a sparse, manifold-guided gradient flow — that (1) uses TabPFN's per-column conditional density as an *exact joint-score oracle*, (2) couples all actionable coordinates through one gradient so it sees feature interactions (lifting the MOONS plateau), (3) unifies the two rival greedy selectors (`prob_ascent`, `class_divergence`) as the two drift terms of one flow, and (4) replaces near-MAP coordinate commits with annealed Langevin sampling that yields a *distribution* of counterfactuals. **Fully offline, local TabPFN v2 checkpoints only, no API, no retraining, no `src/tabpfn/**` changes.**

---

## Context

### Where the predecessor landed

The `iterative-greedy-cf` plan established a strong, well-instrumented baseline (see `experiments/zeroshot_cf/results/REPORT.md` §7c, `exp6_summary.md`):

| Dataset | Best config | Validity | L0 (feat changed) | frac_oob | LOF | Note |
|---------|-------------|----------|-------------------|----------|-----|------|
| HELOC | `prob_ascent`, `knn_both`, ctx 256 | **0.90** (n=50) | 1.67 | 0.04 → **0.00** (knn) | 9.5M → **1.98** (knn) | salvaged vs one-pass 0.538 |
| MOONS | `prob_ascent`, `random_both`, ctx 512 | **0.82** | 1.33 | 0.00 | 1.03 | **plateaus** |

But it carries five ad-hoc seams, each a known theoretical weakness:

1. **Two rival selectors** picked by ablation — `prob_ascent` (steepest ascent on `disc.predict_proba`) vs `class_divergence` (TabPFN-intrinsic class-conditional shift) — with no principle unifying them.
2. **Discrete forward selection is myopic.** Greedy adds one feature at a time by *marginal* gain; it cannot see interactions. The MOONS two-moons flip is intrinsically a 2-D move, so single-coordinate gains can vanish while a joint move flips. This is the submodularity-ratio `γ→0` regime — the MOONS plateau is a **theorem**, not a tuning failure (see Decisions §recorded analysis; `iterative-greedy-cf` Decision #9).
3. **Coordinate-wise MAP commits seek conditional modes** (`temperature ≈ 1e-9`). The assembled point is a *product of conditional modes*, which need not lie in the *joint* high-density set — the ICM (iterated conditional modes) trap. Per-step LOF only sometimes catches it; joint plausibility is never measured.
4. **Context selection** (kNN/random × size) is a third grid-searched axis, selected once around the static factual `x_0` — stale once the iterate moves.
5. **Near-MAP determinism** yields exactly one arbitrary L0-minimizer; recourse non-uniqueness (many equally-sparse CFs) is collapsed to a single tie-break with no diversity or epistemic honesty.

### The unifying idea

Define the counterfactual **posterior** on the feasible slice `F(x0) = {x : x_I = (x0)_I}` (immutables `I` fixed):

```
π_β(x) ∝ p(x | Y=t)^α · exp(β·[h_t(x) − τ]) · exp(−λ·‖x − x0‖_0)
         └ plausibility ┘  └ validity tilt ┘  └ sparsity prior ┘
```

Interrogating `π_β` recovers everything: `argmax` is the best CF (what greedy approximates); a *sample* is a diverse recourse; `β→∞` is the hard flip constraint. The enabling fact is that **TabPFN already gives the joint score**: since `log p(x) = log p(x_j | x_{−j}) + log p(x_{−j})`, the j-th score component is

```
[∇_x log p(x|t)]_j = ∂/∂x_j log p(x_j | x_{−j}, Y=t)
```

— exactly the per-column conditional density TabPFN returns via its `FullSupportBarDistribution` head. Stacking over `j` turns the existing sampler into a manifold score field at no extra model. The flow is then a proximal/IHT Langevin step:

```
δ_{k+1} = H_B( δ_k + η[ α·s_t(x_k)_A + β·σ'·∇h_t(x_k)_A ] + sqrt(2η/β)·ξ_k )
          └ hard-threshold to B coords ┘ └ generative drift ┘ └ discriminative drift ┘ └ noise ┘
```

where `δ = x_A − (x0)_A` (immutables drop out by construction), `H_B` keeps the B largest-|·| coordinates (the L0 prox), and `β_k↑` anneals. This single update:

- **sees interactions** — `∇J` couples all actionable coordinates ⟹ lifts the MOONS `γ→0` plateau (the headline gate);
- **unifies both selectors** — the discriminative drift `∇h_t` *is* `prob_ascent`; the generative contrast `s_t − s_c` *is* `class_divergence`; the mix `α:β` subsumes both;
- **removes the ICM trap** — follows the *joint* score, never commits conditional modes;
- **makes context a moving-bandwidth estimator** — re-select kNN context around `x_k`;
- **yields a recourse distribution** — annealed Langevin samples `π_β`, turning non-uniqueness into diversity.

### The hard technical risk (drives Stage 1)

The bar distribution is **piecewise-constant in density per bucket**, so a naive `∂/∂x_j log p` is zero inside a bucket and undefined at borders. Stage 1 must therefore build a *usable* score estimator and validate it before any flow is run. Three candidates, ranked by expected robustness:

- **(c) mean-shift drift** — drive coordinate j toward `E[x_j | x_{−j}, Y=t]` (the bar-distribution mean, already available via `mean_of_prediction`). For a near-Gaussian conditional, `score ∝ (μ − x)/σ²`, so this is the score up to scale and sidesteps the piecewise problem entirely. **Primary candidate.**
- **(b) finite-difference on bar log-prob** — `[log p(x_j+ε) − log p(x_j−ε)]/2ε` using the bar distribution's log-prob at shifted query values (the bar head evaluates log-prob at arbitrary values cleanly). Truest to the score; needs ε tuning.
- **(a) smoothed-density derivative** — fit a smooth interpolant to the bucket density and differentiate. Most faithful, most fiddly.

Classifier-routed columns (HELOC low-cardinality integers, auto-routed to TabPFN's classifier head — see `iterative-greedy-cf` Fixed Issue #1) have **no continuous gradient**; they evolve by discrete **Gibbs resampling** from `p(x_j | x_{−j}, t)`. The flow is therefore a hybrid PDMP: Langevin on continuous columns, Gibbs jumps on discrete columns — a clean, principled replacement for the int-cast/TV kludge that sank `class_divergence` on HELOC.

### Inherited infrastructure (do not rebuild)

- `experiments/zeroshot_cf/checkpoints.py` — `get_models()` loads **local v2 `.ckpt` files**; the only model entry point. Never `tabpfn_client`.
- `experiments/zeroshot_cf/sampler.py` — `ConditionalDensitySampler`: `set_context(X_context, y_context, target_class, max_context, selection={random,knn}, query)`; `predictive_distribution(X_query, target_col, fixed_target)` → regressor `{"logits": Tensor[m,num_bars], "criterion": FullSupportBarDistribution}` or classifier `{"proba", "classes"}`; `sample_feature(...)`; module helpers `mean_of_prediction(logits, criterion)`, `class_conditional_shift(dist_tgt, dist_cur)`.
- `src/tabpfn/architectures/base/bar_distribution.py` — `FullSupportBarDistribution(borders)`: `.mean(logits)`, `.cdf(logits, y)`, `.compute_scaled_log_probs(logits)`, `.icdf(logits, p)`, attrs `borders`, `bucket_widths`, `num_bars`. **Read-only — do not modify.**
- `experiments/zeroshot_cf/greedy.py` — `greedy_counterfactual(sampler, disc, x, y_target, actionable_idx, selector, *, tau, budget, temperature) → (x_cf, changed, info)`. The new flow mirrors this signature.
- `experiments/zeroshot_cf/discriminator.py` — `DiscriminatorModel.predict/predict_proba`; underlying sklearn `LogisticRegression` exposes `coef_` (shape `(1,d)` binary) ⟹ **exact, cheap, constant `∇h_t`** (logit gradient = `coef_`); MLP path uses `decision_function`.
- `experiments/zeroshot_cf/exp4_greedy_cf.py` — `generate_counterfactuals(dataset_name, selector, tau, budget, temperature, n_permutations, max_context, max_test) → (X_test, y_test, X_cf, info)`; CLI pattern to mirror for the Exp7 driver.
- `experiments/zeroshot_cf/metrics_harness.py` — `compute_metrics(disc_model, X_cf, X_test, X_train, y_test, y_target, immutable_idx, ...)` → `{validity, lof_scores_cf, sparsity, actionability, true_actionability, proximity_l2_jaccard}`. **`frac_oob` is computed inline** in the exp runners, not here — recompute the same way.
- `experiments/zeroshot_cf/data.py` — `load_dataset`, `get_actionable_immutable` (HELOC 6 immutable / 17 actionable; MOONS both actionable).

### Risks

- **Score-estimator bias** is the new bottleneck (Stage 1). Mitigation: validate against a numerical KDE-score ground truth on MOONS (2-D, tractable) before trusting the flow; pick the estimator with lowest cosine error.
- **Flow cost** — each step needs `|A|` `predictive_distribution` calls (one per actionable column for the score) + cheap classifier evals. Comparable to `prob_ascent`'s `O(|A|²)`; bound via `--max-test` held identical across compared cells; heavy runs on the **remote DGX `gx10-bdc5`** (`iterative-greedy-cf` Decision #12).
- **Path-adaptive kNN context** re-fits per step around the moving iterate ⟹ strictly more `set_context` calls than the static path; bound with a re-fit cadence (every k steps) and log it.
- **Non-convexity** — IHT global guarantees need restricted strong concavity; far from the manifold the flow may stall. Annealing (Stage 5) mitigates; certificates remain local. Report failure_rate honestly.
- **Determinism** — the flow with noise off and a fixed schedule must be reproducible (seeded); assert this in tests.

---

## Strategy

Five stages in three phases, all offline against local v2 checkpoints.

- **Phase A — Score oracle (Stage 1):** build & validate the class-conditional score estimator from the bar distribution (mean-shift primary, finite-difference fallback), plus the discrete-column Gibbs proposal. Gate: cosine-accuracy vs numerical ground truth on MOONS. Nothing downstream is trustworthy until this passes.
- **Phase B — Continuous flow core (Stages 2–3):**
  - **Stage 2 — Flow generator** (`flow.py`): the sparse manifold-guided proximal/IHT update with dual drift (generative score + discriminative classifier gradient), hard-thresholding to budget B, hybrid continuous-Langevin / discrete-Gibbs dynamics. Mirrors `greedy_counterfactual`'s interface. Unit tests only.
  - **Stage 3 — Exp7 headline test**: run the flow vs the greedy baseline on MOONS (the **plateau gate**, B=2) and HELOC (the **hold gate**). Joint-NLL plausibility monitor added. This is the decisive experiment.
- **Phase C — Enhancements + synthesis (Stages 4–5):**
  - **Stage 4 — Path-adaptive kNN context**: re-select context around the moving iterate; ablate static vs adaptive on HELOC.
  - **Stage 5 — Annealed Langevin + mix ablation + REPORT**: anneal β, sample N diverse recourses per factual, ablate the generative:discriminative mix `α:β`, report coverage/diversity/joint-NLL, write the consolidated REPORT section + recommended config.

Stage 1 gates all others. Stage 2 depends on Stage 1; Stage 3 on Stage 2. Stage 4 and Stage 5 both depend on Stage 3 and are mutually independent (Stage 4 = context lever, Stage 5 = sampling/synthesis), though Stage 5's REPORT should fold in Stage 4's result if available.

---

## Success Criteria

Headline gate = **break the MOONS plateau** while holding HELOC. Targets stay honest — this remains out-of-the-box exploration; a negative result (flow does not beat greedy) is a legitimate, publishable finding, reported with failure_rate.

| Metric | Baseline (greedy) | Target | Rationale |
|--------|-------------------|--------|-----------|
| Pipeline runs fully offline (v2 ckpts, no API) | yes | yes | `get_models()` only; no `tabpfn_client`; no network; no `src/tabpfn/**` edits. |
| **Score-estimator accuracy (MOONS)** | n/a | mean cosine sim ≥ **0.9** vs numerical KDE-score on a held grid | Stage-1 gate — the flow is meaningless if the score is wrong. |
| **MOONS validity (headline)** | 0.82 (`prob_ascent`, ctx 512) | **> 0.82** at budget B=2, deterministic flow | The plateau gate: joint-gradient coupling sees the 2-D interaction greedy cannot. |
| MOONS failure_rate | ≈0.30 | **markedly lower** | Fewer points plateau when the move is gradient-coupled, not myopic. |
| **HELOC validity (hold)** | 0.90 (n=50) / 0.67 (n=15 grid) | ≥ **0.85** | Must not regress HELOC while fixing MOONS. |
| HELOC frac_oob (hold) | 0.00 (`knn_both@256`) | ≤ **0.05** | Dense conditioning + manifold drift must keep CFs in-distribution. |
| L0 count (both) | MOONS 1.33 / HELOC 1.67 | ≤ baseline + 1 | Sparsity prior (H_B) must keep CFs sparse; small budget growth acceptable if it buys the flip. |
| true_actionability (immutables unchanged) | 1.0 | 1.0 | Immutables never in `δ` — by construction. |
| **Joint plausibility (new)** | not measured | report joint-NLL of CFs vs train distribution; flow ≤ greedy | Instruments the ICM trap (limitation #3): joint-NLL, not just per-step LOF. |
| Selector unification | 2 separate selectors | one flow; mix `α:β` ablation shows dual drift ≥ either alone | Proves `prob_ascent` & `class_divergence` are two terms of one mechanism. |
| Path-adaptive context | static (selected once at x0) | report whether per-step kNN lowers HELOC frac_oob along the path | Stage-4 lever. |
| Recourse distribution | single CF | annealed Langevin yields N diverse valid CFs/factual; report diversity + coverage | Stage-5 epistemic upgrade. |

---

## Files That May Be Changed

### New experiment code (under `experiments/zeroshot_cf/`)
- `score.py` — class-conditional score oracle: `conditional_score(...)` (continuous, from bar distribution) + discrete-column Gibbs proposal + estimator selection (Stage 1).
- `flow.py` — `flow_counterfactual(...)`: sparse manifold-guided proximal/IHT Langevin generator, dual drift, hybrid continuous/discrete dynamics (Stage 2).
- `exp7_flow_cf.py` — flow CF runner + greedy-baseline comparison + joint-NLL monitor; `--mode {flow,greedy}`, `--budget`, `--alpha`, `--beta`, `--steps`, `--noise`, `--context-refit` (Stages 3–5).
- `tests/test_score.py` — score-oracle accuracy + Gibbs-proposal unit tests (Stage 1).
- `tests/test_flow.py` — flow loop unit tests: immutability, budget/H_B, termination, determinism at noise=0 (Stage 2).

### Modified
- `sampler.py` — only if a thin read-only accessor is needed to expose the bar `borders`/`logits` for the score (default behaviour preserved; no signature breaks). Prefer using the existing `predictive_distribution()` return as-is.
- `results/REPORT.md` (new §for this plan), `results/exp7_*` CSV/MD artefacts.

> **Artefact path convention**: every `results/...` path resolves to `experiments/zeroshot_cf/results/` (the `RESULTS_DIR = Path(__file__).parent / "results"` convention), not a repo-root `results/`.

> The core `src/tabpfn/**` package is **not** modified (zero architecture changes). No `tabpfn_client` / cloud API anywhere.

---

## Progress Tracker

| # | Stage | Status | Notes | Commit |
|---|-------|--------|-------|--------|
| 1 | [Class-conditional score oracle](stages/01-score-oracle.md) | PENDING | | |
| 2 | [Sparse manifold-guided flow core](stages/02-flow-core.md) | PENDING | | |
| 3 | [Exp7 headline test (MOONS plateau gate)](stages/03-headline-test.md) | PENDING | | |
| 4 | [Path-adaptive kNN context](stages/04-path-adaptive-context.md) | PENDING | | |
| 5 | [Annealed Langevin + mix ablation + REPORT](stages/05-annealing-report.md) | PENDING | | |

Statuses: `PENDING` -> `IN_PROGRESS` -> `DONE` | `BLOCKED` | `SKIPPED`

Phases: **A = Stage 1 (score oracle, gates all)**, **B = Stages 2–3 (flow core + headline test)**, **C = Stages 4–5 (enhancements + synthesis)**. Stage 1→2→3 is a strict chain. Stages 4 and 5 both depend on 3 and are mutually independent. See `resources/math.md` for the full derivation, `resources/grids.md` for ablation grids, `resources/commands.md` for run commands.

---

## Execution Protocol

This plan is built for **autonomous, unattended execution**. The guiding principle is
**keep making progress**: resolve problems in place when you can, defer them when you
can't, and never halt the whole plan over a single fixable or deferrable issue.

For each stage:

1. **Read the progress tracker** above and pick the stage to work on. If a stage is
   **IN_PROGRESS**, a previous run was interrupted mid-stage — resume and finish that one
   (re-read its steps, inspect the working tree to see what's already done) before
   starting anything new. Otherwise, take the first **PENDING** stage.
2. **Read the stage file** -- follow the link in the tracker to the stage's .md file.
3. **Read resources** -- if the stage references shared resources, find them in `resources/`.
4. **Resolve ambiguity yourself** -- there is no user to ask during an autonomous run.
   Pick the most reasonable interpretation that fits the codebase and existing
   conventions, record it under **Decisions**, and proceed. Only defer to the Backlog
   if the ambiguity genuinely blocks any sensible implementation.
5. **Implement** -- execute the steps described in the stage.
6. **Validate** -- run the verification checks and the test suite. **If anything fails,
   do not stop — triage it via the self-healing loop below.**
7. **Update this index** -- mark the stage DONE in the progress tracker, add brief notes
   about what was done and any deviations. Log every problem you hit in **Fixed Issues**
   (if resolved) or **Backlog** (if deferred). Never silently drop a problem.
8. **Commit** -- create an atomic commit with the message specified in the stage.
   Include all changed files (code, config, docs, and this plan's index.md).

Repeat until every stage is DONE or terminally deferred. After the last stage, **sweep
the Backlog**: attempt any items that are now resolvable, and leave the rest for a
follow-up run.

### Self-healing loop (handling problems)

When a step fails — failing test, build/lint/type error, a bug in the new code, an
unexpected runtime error:

1. **Triage** the problem as *light* or *heavy*.
   - **Light** -- self-contained and fixable in a focused effort: a failing unit test,
     a lint/type error, a missing import, a small logic bug in code you just wrote.
   - **Heavy** -- needs an architectural decision, spans many files, depends on an
     external blocker, contradicts the plan's assumptions, or has already survived a
     fix attempt.
2. **Light → delegate the fix to a subagent.** Spawn a focused subagent (Agent/Task
   tool) with: the failing command and its full output, the relevant file paths, the
   stage goal, and a crisp deliverable (e.g. "make `<test>` pass without weakening
   assertions"). Delegating keeps the main execution context clean. Re-run verification
   when it returns. Cap at **2 attempts per issue** — if still failing, treat it as heavy.
3. **Heavy → defer to the Backlog.** Add a self-contained entry (see the Backlog table).
   Do **not** keep grinding and do **not** halt the plan.
4. **Decide the stage's disposition:**
   - If the stage's core goal is met without the deferred item → mark **DONE**, note the
     backlog reference, and continue.
   - If the deferred item is essential to this stage → mark **BLOCKED**, note the backlog
     reference, and continue to the next *independent* stage. Only stop the run when every
     remaining stage depends on blocked work.
5. **Record** the outcome: resolved problems → **Fixed Issues**; deferred problems → **Backlog**.

### Guardrails

- Keep every commit in a working, buildable state.
- **Never weaken, skip, or delete a test to make it pass.** If a test is genuinely wrong,
  fix it correctly and note it in Fixed Issues.
- Never use `git commit --no-verify`.
- Don't expand a stage's scope to chase a heavy problem — that's what the Backlog is for.
- **Offline guarantee**: every run must work with no network. Models load **only** via
  `from checkpoints import get_models`; never import `tabpfn_client` or call the cloud API.
- **Do not modify `src/tabpfn/**`** — the whole premise is zero architecture changes. The
  bar distribution is read-only.
- **A negative headline result is valid.** If the flow does not beat greedy on MOONS,
  report it honestly with failure_rate and joint-NLL; do not tune until a number appears.

---

## Fixed Issues

Problems encountered during execution and resolved (in place or via a fix subagent).
Leave empty until execution surfaces something.

| # | Stage | Symptom | Root Cause | Resolution | Fixed By |
|---|-------|---------|-----------|------------|----------|
| | | | | | |

---

## Backlog (Deferred Issues)

Problems deferred for later — too heavy to fix inline without derailing the plan.
Each entry must be **self-contained enough for a future run to pick it up cold**:
state the symptom, where it came from, and a concrete lead for resolving it.

| # | Title | Origin Stage | Severity | Why Deferred | Suggested Next Step | Status |
|---|-------|--------------|----------|--------------|---------------------|--------|
| | | | | | | |

Statuses: `OPEN` -> `IN_PROGRESS` -> `RESOLVED`. When an item is resolved, flip its
status and summarize the fix in **Fixed Issues**. Heavy items may warrant their own
follow-up plan — link it here.

---

## Decisions

Decisions made during planning (2026-06-23):

1. **Branch from `iterative-greedy-cf`, not `main`.** This plan reuses `greedy.py`, the kNN-context sampler, the discriminator, and the Exp4/5/6 runners that live on `iterative-greedy-cf` (not yet merged to main). Rebase/merge onto main is a separate concern.
2. **The counterfactual posterior `π_β` is the single governing object** (see Context). All four predecessor mechanisms (two selectors, kNN context, near-MAP commit) are corners of it: discriminative drift = `prob_ascent`, generative drift = `class_divergence`, moving-bandwidth context = per-step kNN, `β→∞`+noise→0 = near-MAP.
3. **Score estimator ranked mean-shift (primary) → finite-difference → smoothed-derivative.** The bar density is piecewise-constant, so the naive analytic derivative is unusable; mean-shift (`(μ−x)` toward the conditional mean) is the robust default and equals the score up to scale for near-Gaussian conditionals. Stage 1 validates the choice against a numerical KDE-score ground truth on MOONS and records the winner.
4. **Hybrid dynamics for mixed data.** Continuous columns evolve by Langevin (score drift); classifier-routed (low-cardinality integer) columns evolve by discrete Gibbs resampling from `p(x_j|x_{−j},t)`. This replaces the int-cast/TV kludge (`iterative-greedy-cf` Fixed Issue #1) with a principled jump process.
5. **Sparsity via hard-thresholding `H_B` to a budget B**, with B annealed `|A|→B_min` (an L1→L0 homotopy). Greedy is the special case B incremented by 1 with no joint gradient. Default `B=2` for the MOONS plateau gate (the minimal block that can express the 2-D interaction).
6. **Discriminative drift uses the exact LR gradient.** The discriminator is `LogisticRegression`; `∇h_t` in logit space is the constant `coef_` vector — cheap and exact, no autograd through the classifier needed. MLP path falls back to `decision_function` finite difference.
7. **Validity stop oracle stays `disc.predict` flip** (same as greedy, `iterative-greedy-cf` Decision #1) so the metric is directly comparable. The validity *tilt* in `π_β` (smooth `σ(β(h_t−τ))`) drives the flow; the *stop* is the hard flip.
8. **Heavy runs on the remote DGX `gx10-bdc5`** (`iterative-greedy-cf` Decision #12); local CPU is for tests + smoke only. HELOC test counts may be bounded (`--max-test`) and logged, as in `iterative-greedy-cf` Decision #13 — when bounded, frac_oob/LOF/joint-NLL trends are the robust signal over noisy validity.
9. **A negative headline (flow ≤ greedy on MOONS) is a recorded finding, not a bug** (mirrors `iterative-greedy-cf` Decision #9). Report it with failure_rate, joint-NLL, and the score-estimator accuracy so the cause (myopia vs score bias vs landscape) is attributable.

Decisions made during autonomous execution should be appended below.
