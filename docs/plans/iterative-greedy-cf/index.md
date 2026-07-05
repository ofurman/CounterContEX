# Plan: Iterative Greedy Counterfactual Generation with TabPFN (Local Inference)

**Date**: 2026-06-21
**Branch**: `iterative-greedy-cf` (create from `main`)
**Predecessors**: [`zeroshot-tabpfn-cf`](../zeroshot-tabpfn-cf/index.md) (Stages 1–9 DONE) — provides the offline v2 checkpoints, `ConditionalDensitySampler`, the discriminator oracle, the dataset loaders, and the metrics harness this plan reuses unchanged.
**Goal**: Replace the one-pass "mask all actionable features at once" counterfactual generation with an **iterative greedy** procedure that changes actionable features **one at a time** and **stops at the class flip** — minimizing L0 sparsity by construction — then rigorously ablate the two candidate-selection strategies and the context-construction choices that drive validity and plausibility. **Fully offline, local TabPFN v2 checkpoints only, no API, no retraining, no architecture changes.**

---

## Context

The predecessor plan (`zeroshot-tabpfn-cf`) established that a pre-trained TabPFN v2 model, used as a conditional density estimator via the Y-as-appended-column trick, can generate counterfactuals by masking actionable features, freezing immutables, and conditioning on a target class. It works well on MOONS (validity ≈ 1.0, LOF ≈ 1.06) but **breaks on HELOC** (`results/REPORT.md`):

- **Sparsity is an artefact of the mask, not an objective.** The one-pass method masks the *entire* actionable set, so every actionable feature changes (MOONS sparsity = 1.0; HELOC ≈ 0.70). Nothing minimizes the *number* of features changed.
- **HELOC plausibility collapses** — `frac_oob ≈ 0.65`, `LOF ≈ 5.7e9` — because 17/23 features are imputed from only ~7 observed columns (severe under-determination). Stage 9 of the predecessor plan proved the controlling lever is the **number of simultaneously-masked features** (a reduced 6-feature set cut OOB 0.50 → 0.10), *not* temperature and *not* feature ordering.

This plan attacks both problems with one mechanism. Changing features **one at a time** until the origin classifier's prediction flips is the classic **greedy / forward-selection** template for counterfactuals (SEDC — Martens & Provost 2014; NICE — Brughmans et al. 2023):

- It is a **direct optimizer of L0 sparsity** — the loop adds features only until the flip, never more. For a *linear* discriminator the greedy feature set is provably minimal.
- Every step masks exactly **one** column conditioned on all the rest, so generation never leaves the **dense-conditioning regime** that Exp1 showed TabPFN handles well — directly addressing HELOC's OOB failure.

Two design dimensions then need rigorous ablation:

1. **Which feature to change next** (`select_candidate`). Two alternatives are studied — *not* combined:
   - **Strategy 1 — steepest-ascent on target-class probability** (wrapper / score-driven, SEDC/NICE): pick the feature whose imputed value most increases `disc.predict_proba[y_target]`.
   - **Strategy 2 — class-divergence** (TabPFN-intrinsic, classifier-free): pick the feature whose class-conditional distribution shifts most between `Y=y_target` and `Y=current`.
2. **How the conditioning context is built** — its **size**, **class scope**, and **selection method**. The predecessor plan only ever used a 256-row random subsample. Larger and/or more *relevant* (nearest-neighbour) context is the untested lever most likely to lift HELOC validity and plausibility.

### Inherited infrastructure (do not rebuild)

- `experiments/zeroshot_cf/checkpoints.py` — `get_models()` loads **local v2 `.ckpt` files** via explicit `model_path=`; uses the open-source `tabpfn` package, never `tabpfn_client`. This is the **only** model entry point.
- `experiments/zeroshot_cf/sampler.py` — `ConditionalDensitySampler`: `set_context` (Y-as-column, class filter, `max_context` random subsample), `impute_masked(mask_cols, fixed_target, dag)`, `sample_feature(target_col)`.
- `experiments/zeroshot_cf/discriminator.py` — sklearn LR/MLP validity oracle (`.predict`, `.predict_proba`, `.eval()`).
- `experiments/zeroshot_cf/data.py` — `load_dataset` (HELOC, MOONS), MinMax→[0,1], `get_actionable_immutable` (HELOC 6 immutable / 17 actionable; MOONS both actionable).
- `experiments/zeroshot_cf/metrics_harness.py` — `compute_metrics` returns `validity, lof_scores_cf, sparsity, actionability, true_actionability, proximity_l2_jaccard`. **`frac_oob` is NOT returned** — it is computed inline in `exp2_counterfactuals.py:264–267` and must be recomputed the same way in the new runners.

### Risks

- **Greedy cost**: Strategy 1 is `O(|A|²)` single-column imputes per point. On HELOC (17 actionables, ~seconds per impute) this is the dominant runtime; bounded via `--max-test` held identical across compared cells.
- **kNN context** must be selected per query point (it depends on the factual row), which breaks the per-class batching the one-pass path used — increasing cost. Bound the same way and log it.
- **Strategy 2 requires a both-classes context pool** (the Y column must be non-constant to contrast `Y=target` vs `Y=current`) — same constraint as predecessor Decision #10. It is therefore incompatible with target-only context strategies.

---

## Strategy

Four stages, grouped into two phases, all offline against local v2 checkpoints.

- **Phase A — Mechanism (Stage 1)**: build the iterative greedy loop, both selection strategies, the single-column predictive-distribution helper, the Exp4 runner, and tests. Produces the first sparsity-optimal CFs and the headline greedy-vs-one-pass comparison.
- **Phase B — Ablations (Stages 2–4)**:
  - **Stage 2 — Selector ablation**: Strategy 1 vs Strategy 2 across MOONS + HELOC at a fixed baseline context. Picks the selector used downstream.
  - **Stage 3 — kNN context selection**: extend `set_context` with random-vs-nearest-neighbour selection and a target-class-vs-both-classes pool (4 context strategies). Tests only — no experiment runs.
  - **Stage 4 — Context ablation**: grid over context **size** {256, 512, 1024, 2048} × **strategy** {random_target, random_both, knn_target, knn_both}, at the Stage-2 winning selector, on both datasets. Consolidated REPORT + recommended production config.

Stage 1 is a prerequisite for all later stages. Stages 2 and 3 are independent of each other; Stage 4 depends on both (it needs the selector chosen in Stage 2 and the context machinery added in Stage 3).

- **Phase C — Follow-up from the 2026-06-23 meeting (Stages 5–9)**: act on the review of the greedy mechanism.
  - **Stage 5 — Budget > |A| + feature revisiting**: remove the one-change-per-feature exclusion so features can be re-imputed under updated conditioning; add a no-progress guard; sweep `budget` (Exp7). The suspected fix for MOONS validity stalling at ≈0.70.
  - **Stage 6 — MOONS trajectory plots (Exp8)**: visualize per-step landings, the selected feature each step, and the "blocked slice" regions. Next-meeting deliverable.
  - **Stage 7 — Discrete dataset**: wire a categorical dataset and confirm the predicted ≈100% validity (all current datasets are continuous).
  - **Stage 8 — Binning / routing audit (Exp9)**: document TabPFN's ordered bar-distribution continuous handling; fix the low-cardinality-integer → classifier-head support loss via a routing override experiment.
  - **Stage 9 — Consolidated table + REPORT**: surface Proximity as a first-class column across all datasets/configs; fold in Stages 5–8. The headline "tabelka" deliverable.

  Dependencies: Stage 5 gates Stage 6 (trajectories reflect the revisit loop). Stages 5, 7, 8 are mutually independent (all build on Stage 1). Stage 9 depends on 5–8. Stages 5/7/8 are heavy-experiment stages (run on the remote DGX per Decision #12); Stages 6/9 are visualization/synthesis (local).

---

## Success Criteria

Targets stay modest — this remains an out-of-the-box exploration. "Success" = a working offline pipeline producing interpretable, honestly-reported metrics, plus a clear read on whether iterative greedy + better context salvages HELOC.

| Metric | Baseline (Stage-8 one-pass) | Target | Rationale |
|--------|------------------------------|--------|-----------|
| Pipeline runs fully offline (v2 ckpts, no API) | yes | yes | No network; `get_models()` only; no `tabpfn_client` import. |
| L0 count — features changed (HELOC) | one-pass changes all 17 actionables (`sparsity≈0.70` of 23 cols) | `l0_count_mean` **markedly < 17** for valid CFs | The core point of greedy: stop at the flip. Report the integer `l0_count_*` keys *and* the existing fractional `sparsity`. |
| Steps-to-flip (MOONS) | n/a | ≤ 2 (of 2 actionables) | Often a single-feature flip on 2-D data. |
| Validity (MOONS) | 0.995 | ≥ 0.95 | Greedy drives toward the flip; should hold. |
| Validity (HELOC) | 0.538 | record actual (target ≥ 0.538, not a gate) | Greedy explicitly optimizes the flip and *may* meet/beat one-pass — but committed values are MAP-conditioned on the mostly-original row, and steepest-ascent can **plateau** (no remaining feature increases `p_target`) before a flip, raising `failure_rate`. A HELOC validity below 0.538 is a legitimate finding, not a bug — report it alongside `failure_rate`. |
| Plausibility — frac_oob / LOF (HELOC) | 0.65 / 5.7e9 | **markedly lower** | One masked column per step ⇒ dense conditioning. |
| true_actionability (immutables unchanged) | 1.0 | 1.0 | Immutables never candidates — by construction. |
| Selector ablation | — | identify the winner on validity / L0 / steps | Strategy 1 vs Strategy 2, both datasets. |
| Context ablation | 256 / random / target-only only | identify best (size, strategy); report whether larger / kNN context lifts HELOC validity & plausibility | Untested lever from the predecessor plan. |
| **Validity vs budget (MOONS)** — Phase C | 0.70 at budget=\|A\|=2 | report the curve; does revisiting lift it toward 1.0, and at what budget does it saturate? (not a hard gate) | Stage 5: feature revisiting under updated conditioning is the meeting's hypothesized fix. A plateau below 1.0 is a legitimate TabPFN-vs-classifier finding. |
| **Validity (discrete dataset)** — Phase C | n/a (no discrete dataset today) | ≈1.0 | Stage 7: categorical commits land in-support; meeting predicts easy validity. |
| **Routing override (HELOC)** — Phase C | int cols auto-routed to classifier head | report Δ validity / Δ proximity / Δ frac_oob with vs without forcing numeric treatment | Stage 8: tests whether preserving ordered support helps. |
| **Proximity surfaced in headline table** — Phase C | computed but omitted from the viewed table | `proximity_l2_jaccard` is a first-class column for every (dataset, config) | Stage 9: the metric already exists; surface it. |

---

## Files That May Be Changed

### New experiment code (under `experiments/zeroshot_cf/`)
- `greedy.py` — the iterative greedy loop + the two `select_candidate` strategies (Stage 1).
- `exp4_greedy_cf.py` — greedy CF runner with `--selector` and stop/budget flags + L0/steps metrics (Stage 1).
- `exp5_selector_ablation.py` — Strategy 1 vs Strategy 2 grid driver (Stage 2).
- `exp6_context_ablation.py` — size × context-strategy grid driver (Stage 4).
- `exp7_budget_sweep.py` — validity-vs-budget sweep driver (Stage 5).
- `exp8_moons_trajectories.py` — MOONS per-step trajectory + blocked-slice plots (Stage 6).
- `exp9_routing_audit.py` — classifier-routing override experiment (Stage 8).
- `tests/test_greedy.py` — greedy loop + selector unit tests (Stage 1; extended in Stage 5 for revisiting + no-progress guard).
- `tests/test_context.py` — kNN context-selection unit tests (Stage 3).
- `tests/test_discrete_dataset.py` — native-categorical dataset load + sampler-routing + Exp4 smoke (Stage 7).
- `tests/test_routing.py` — classifier↔regressor routing override (Stage 8).

### New config / data (Phase C)
- `experiments/zeroshot_cf/configs/<name>_actionability.yaml` + (if synthetic or adapted)
  `experiments/zeroshot_cf/vendor/counterfactuals/{data,config/datasets}/<name>.*` —
  native-categorical dataset without one-hot expansion (Stage 7).
- `results/figures/` — MOONS trajectory PNGs + README (Stage 6).
- `results/exp7_*`, `results/exp9_*`, `results/binning_audit.md`, `results/summary_table.{md,csv}` (Stages 5/8/9).

### Modified
- `greedy.py` — Phase C: remove the one-change-per-feature exclusion (`:172–174`), add the `--stall-eps` no-progress guard, keep `budget` decoupled from `|A|` (Stage 5); add a routing-override pass-through (Stage 8). Default behaviour preserved (`budget=None`, no forced-numeric cols).
- `exp4_greedy_cf.py` — Phase C: `l0_count_*` counts **distinct** features changed (not commit-list length, which can now repeat), keep `steps_*`; add `--stall-eps`; allow Stage-7 dataset names beyond `{moons,heloc,all}`; thread explicit categorical indices / `--force-numeric-cols` into sampler construction (Stages 5/7/8).
- `sampler.py` — add `fixed_target` pass-through to `sample_feature()` so the greedy commit can sample class-conditionally (Stage 1, prerequisite — default `None` preserves existing behaviour); add `predictive_distribution()` helper (Stage 1); extend `set_context()` with `selection={random,knn}` and `pool={target,both}` (Stage 3); accept explicit `categorical_features_indices` and force-numeric overrides so Stage 7 uses TabPFN native categorical handling instead of one-hot (Stages 7/8). Existing behaviour (random subsample, `append_target=False` sampling) preserved as the default.
- `tests/conftest.py` — host the shared `models` fixture (lifted from `test_sampler.py`/`test_ordering.py` in Stage 1; today `conftest.py` only does `sys.path` setup).
- `results/REPORT.md` (this plan's results section), `results/exp4_*`, `results/exp5_*`, `results/exp6_*` CSV/MD artefacts.

> **Artefact path convention**: every `results/...` path in this plan resolves to `experiments/zeroshot_cf/results/` (exp2's `RESULTS_DIR = Path(__file__).parent / "results"` convention), **not** a repo-root `results/`. The existing report is `experiments/zeroshot_cf/results/REPORT.md`.

> The core `src/tabpfn/**` package is **not** modified (zero architecture changes). No `tabpfn_client` / cloud API anywhere.

---

## Progress Tracker

| # | Stage | Status | Notes | Commit |
|---|-------|--------|-------|--------|
| 1 | [Iterative greedy core + both selectors](stages/01-greedy-core.md) | DONE | greedy loop + both selectors + `predictive_distribution`/bar helpers + Exp4 runner + 7 new tests (20/20 pass). HELOC smoke (n=5): validity 1.0, frac_oob 0.0, LOF 1.85, l0≈1.4 — salvages HELOC. MOONS validity ≈0.69 (near-MAP plateau, anticipated finding — see Decision #8). | 5dfcaba |
| 2 | [Selector ablation (Strategy 1 vs 2)](stages/02-selector-ablation.md) | DONE | Run on remote DGX GPU (Decision #12). **`prob_ascent` wins decisively.** MOONS: 0.70 vs 0.64 validity. **HELOC (n=50): prob_ascent validity 0.90, L0 1.67, fail 0.10, frac_oob 0.04 — vs class_divergence 0.52, L0 14.27, fail 0.48.** Greedy+prob_ascent lifts HELOC validity 0.538→0.90 and L0 17→1.67 over one-pass. class_divergence degrades on HELOC (int-collapsed classifier cols weaken its TV-distance signal → budget exhaustion). Chosen downstream selector = **prob_ascent**. | (see git log) |
| 3 | [kNN / context-selection support](stages/03-knn-context-selection.md) | DONE | `set_context` gains `selection={random,knn}` + `query` (kNN anchor); module-level `_knn_indices`; default `random` path byte-identical. `test_context.py` (6 cases a–e). Full suite 30 passed. Committed before Stage 2 since both touch `sampler.py` independently (Stages 2/3 are mutually independent per plan). | (see git log) |
| 4 | [Context ablation (size × strategy)](stages/04-context-ablation.md) | DONE | 16-cell grid on remote DGX GPU at `prob_ascent`. MOONS n=100; **HELOC n=15** (bounded, Decision #13; full grid ~5.3 h). **Finding: bigger context HURTS HELOC** (random `frac_oob` 256→2048 0.13→0.53); **kNN beats random** at every size (LOF 1e6 vs 1e7–1e10). **Best: `knn_both@256` — frac_oob 0.000, LOF 1.98.** Recommended HELOC `(prob_ascent,256,knn_both)`, MOONS `(prob_ascent,512,random_both)`. `true_actionability=1.0` all cells. Consolidated REPORT.md §7c + exp6_summary verdict written. | (see git log) |
| 5 | [Budget > \|A\| + feature revisiting (Exp7)](stages/05-budget-revisit.md) | DONE | Preflight passed locally (CUDA/checkpoints/vendor/test_context). Implemented revisits + `--stall-eps` guard + distinct-L0 accounting + Exp7 sweep. MOONS n=100: validity 0.82 at every budget 2–64, no lift beyond \|A\|, steps_max=2, frac_oob=0, true_actionability=1.0. HELOC n=30: validity 0.80 at every budget 17–1000, saturates at budget 17 (steps_max=6), L0=1.83, frac_oob=0, LOF=1.85, true_actionability=1.0. Full `uv` suite 40 passed; `poetry` unavailable on host. | |
| 6 | [MOONS trajectory plots (Exp8)](stages/06-moons-trajectories.md) | DONE | Added `exp8_moons_trajectories.py`; generated `results/figures/moons_trajectories.png`, `moons_blocked_slice.png`, and README. Near-boundary plot uses 30 trajectories; bounded fallback scan found stalled row 128 for the blocked-slice panel. Exp8 offline run passed; `uv` suite 40 passed; `poetry` unavailable on host. | |
| 7 | [Discrete dataset + validity check](stages/07-discrete-dataset.md) | DONE | Added native all-categorical `binary_cat` (3 binary semantic columns, no one-hot) + generic actionability config + explicit sampler categorical routing. Exp4 `prob_ascent` uses all-classes context for native categoricals to avoid one-class support collapse; n=50 result: validity 1.00, L0 1.00, frac_oob 0.00, LOF 1.00, true_actionability 1.00. Full `uv` suite 44 passed; `poetry` unavailable on host. | (see git log) |
| 8 | [Binning / routing audit (Exp9)](stages/08-binning-routing-audit.md) | DONE | Added binning audit note, `--force-numeric-cols` routing override, Exp9 runner, and 4 routing tests. HELOC has 5 classifier-routed low-cardinality columns. Bounded Exp9 smoke (`n=1`, `budget=1`, `n_perm=1`) found override kept validity 1.0, improved proximity 0.346→0.133 and LOF 1.86→1.08, frac_oob 0. Full `n=30`/`budget=17` run hit `prob_ascent` O(\|A\|²) worst-case and is deferred to Backlog #4. Full `uv` suite 48 passed; `poetry` unavailable on host. | (see git log) |
| 9 | [Consolidated table + REPORT](stages/09-consolidated-table.md) | PENDING | Phase C. Surface `proximity_l2_jaccard` as a first-class column; fold in Stages 5–8. The headline "tabelka". Depends on 5–8. | |

Statuses: `PENDING` -> `IN_PROGRESS` -> `DONE` | `BLOCKED` | `SKIPPED`

Phases: **A = Stage 1 (mechanism)**, **B = Stages 2–4 (ablations)**, **C = Stages 5–9 (2026-06-23 meeting follow-up)**. Stage 1 gates all others. Stages 2 and 3 are mutually independent; Stage 4 depends on both. In Phase C: Stage 5 gates Stage 6; Stages 5/7/8 are mutually independent; Stage 9 depends on 5–8. See `resources/grids.md` for the exact grids/sweeps and `resources/commands.md` for run commands.

---

## Execution Protocol

This plan is built for **autonomous, unattended execution**. The guiding principle is
**keep making progress**: resolve problems in place when you can, defer them when you
can't, and never halt the whole plan over a single fixable or deferrable issue.

### Required host preflight (blocks Stage 5+)

Before starting the next pending stage, run the **Host preflight** in
`resources/commands.md` on the execution host. This is a hard prerequisite for the Phase C
experiment stages because the repo intentionally does not track the CEL vendor tree or the
TabPFN v2 checkpoints.

The preflight must verify all of the following before Stage 5 begins:

- `tabpfn-extensions` imports successfully, so `ConditionalDensitySampler` can import
  `TabPFNUnsupervisedModel`.
- `cel` imports successfully and `load_dataset("moons")` / `load_dataset("heloc")` resolve
  through `experiments/zeroshot_cf/vendor/counterfactuals`.
- Both local v2 checkpoint files exist under `experiments/zeroshot_cf/models/`.
- CUDA is visible when running with `TABPFN_DEVICE=cuda` on the DGX/GB10 host.
- `uv run pytest experiments/zeroshot_cf/tests/test_context.py -q` passes as a lightweight
  sampler/context smoke test.

If this preflight fails, the stage is **not** allowed to start. Fix provisioning first; do
not classify missing dependencies, missing vendor data, missing checkpoints, or CUDA
unavailability as a deferrable stage backlog item. Spark is not required by the current
Python experiment commands unless the execution environment separately mandates it.

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
- **Do not modify `src/tabpfn/**`** — the whole premise is zero architecture changes.

---

## Fixed Issues

Problems encountered during execution and resolved (in place or via a fix subagent).
Leave empty until execution surfaces something.

| # | Stage | Symptom | Root Cause | Resolution | Fixed By |
|---|-------|---------|-----------|------------|----------|
| 1 | 2 | `class_divergence` selector crashed `KeyError: 'logits'` on HELOC (worked on MOONS) at `greedy.py:_select_class_divergence`. | `model.fit`'s `infer_categorical_features` auto-routes HELOC low-cardinality integer columns to TabPFN's **classifier** head, so `predictive_distribution` returns `{"proba"}` not `{"logits","criterion"}`. MOONS is all-continuous → regressor only. The fitted classifier's `classes_` are int-cast (MinMax-[0,1] → all 0), so the true feature support is unrecoverable (expected-value shift infeasible). | Added `class_conditional_shift(dist_tgt, dist_cur)` helper handling both dict shapes: regressor → abs mean-shift (`mean_of_prediction`), classifier-routed → **total-variation distance** between the two class-proba vectors (both bounded [0,1], comparable for the argmax ranking). `predictive_distribution` classifier branch now returns `{"proba","classes"}`; `_select_class_divergence` delegates to the helper. Regression test `test_class_divergence_handles_classifier_column` (fails pre-fix, passes post). Full suite 31 passed. | fix subagent |
| 2 | 5 | Exp7 budget sweep can waste hours recomputing larger budgets after every point already flipped or stalled below the next cap. | The initial driver treated each budget as independent even when the previous row's `steps_max` proved that a larger cap cannot affect any generated CF. | Added a saturation shortcut: when `steps_max < next_budget`, copy the identical metric row for remaining larger budgets with `runtime_s=0.0`. Used for HELOC after budget 17; MOONS was run before the shortcut and empirically matched across all budgets. | inline |
| 3 | 5 | Required verification command `poetry run pytest` failed immediately with `poetry: command not found`. | This execution host is provisioned for the plan's `uv` workflow but does not have the Poetry binary installed. | Ran the project experiment suite through `uv` instead (`uv run pytest experiments/zeroshot_cf/tests -q`), matching the Phase C command resource; all 40 tests passed. | inline |
| 4 | 5 | `git commit` failed with `Author identity unknown`. | The execution host had no local Git `user.name` / `user.email` configured. | Set repo-local Git identity to the author from the previous commit (`Oleksii Furman <oleksii.furman@gmail.com>`) and retried the commit. | inline |
| 5 | 6 | Required verification command `poetry run pytest` failed immediately with `poetry: command not found`; an initial bare `python -m py_compile` check also failed with `python: command not found`. | Same host provisioning as Stage 5: the experiment environment is managed through `uv`, with no Poetry or bare `python` shim installed. | Re-ran checks through `uv run python`; Exp8 script compiled and `uv run pytest experiments/zeroshot_cf/tests -q` passed 40/40. | inline |
| 6 | 7 | Initial `binary_cat` Exp4 run had validity 0.48 and only flipped target-0 points; target-1 categorical commits stayed at `decision_code=0` despite classifier probabilities assigning class 1 near 1.0. | `prob_ascent` used target-only context. For semantic categorical columns this can create one-class feature support; the classifier-head imputation path returned the support ordinal `0` for the target-1 one-class fit. | Native-categorical Exp4 runs now use an all-classes context pool while retaining class conditioning through appended Y. This preserves full categorical support and restored symmetric commits; bounded `binary_cat` run reached validity 1.0. | inline |
| 7 | 7 | Required verification command `poetry run pytest` failed immediately with `poetry: command not found`. | Same host provisioning as Stages 5–6: the experiment environment is managed through `uv`, with no Poetry binary installed. | Ran the full experiment test suite through `uv` instead (`uv run pytest experiments/zeroshot_cf/tests -q`); all 44 tests passed. Guardrails (`src/tabpfn` diff and `tabpfn_client` grep) passed. | inline |
| 8 | 8 | Required verification command `poetry run pytest` failed immediately with `poetry: command not found`. | Same host provisioning as Stages 5–7: the experiment environment is managed through `uv`, with no Poetry binary installed. | Ran the full experiment test suite through `uv` instead (`uv run pytest experiments/zeroshot_cf/tests -q`); all 48 tests passed. | inline |
| 9 | 8 | Full Exp9 (`--max-test 30`, natural HELOC `budget=17`) and a reduced `--max-test 5 --budget 5` attempt did not finish in practical time after the forced-numeric override repeatedly exhausted the greedy budget. | With revisits enabled, `prob_ascent` evaluates every actionable candidate at every commit. For HELOC this makes the override cell hit the known O(\|A\|²) worst case: up to 17 candidates × 17 steps per point, plus per-query kNN context fits. | Added an explicit `--budget` knob to Exp9, documented the limitation, produced a bounded smoke artefact (`--max-test 1 --budget 1 --n-permutations 1`), and deferred the statistically stable full run to Backlog #4. | inline |

---

## Backlog (Deferred Issues)

Problems deferred for later — too heavy to fix inline without derailing the plan.
Each entry must be **self-contained enough for a future run to pick it up cold**:
state the symptom, where it came from, and a concrete lead for resolving it.

| # | Title | Origin Stage | Severity | Why Deferred | Suggested Next Step | Status |
|---|-------|--------------|----------|--------------|---------------------|--------|
| 1 | **Combined / MI / entropy selector.** Combine Strategy 1 and Strategy 2 (`score = p_ascent − λ·divergence`) to balance validity and proximity in one rule, or replace divergence with **mutual information** between the target and the candidate feature, or with class-conditional **entropy** comparison. | Meeting 2026-06-23 | Medium | Explicitly punted in the meeting ("to jest dyskusja jak na potem … jak strategia 1 nie będzie działać tak dobrze"). Strategy 1 (`prob_ascent`) already won Stage 2; class_divergence is deprioritized. Introduces a λ hyperparameter to tune. | If `prob_ascent` + revisiting (Stage 5) still leaves proximity/validity gaps, add a combined-score selector to `greedy.py` and ablate λ; consider MI(target; feature) as a classifier-free alternative to TV-distance. | OPEN |
| 2 | **Newer TabPFN backbone (v2.5 / v3).** Compare against v2.5/2.6 checkpoints (exist on HuggingFace, not cached locally) — larger context capacity may handle the big-context degradation better. v3 needs a `TABPFN_TOKEN` (offline-incompatible). | Meeting 2026-06-23 | Low | Lukewarm in the meeting ("z naszego punktu widzenia chyba wiele się nie zmieni"); requires a one-time checkpoint download (breaks the strict-offline guarantee) and a `checkpoints.py` version switch. | Fetch v2.5 regressor+classifier ckpts into `models/`, add `TABPFN_MODEL_VERSION=v2.5` path in `checkpoints.py`, re-run the Stage-9 headline configs, compare. Keep v2 as the default. | OPEN |
| 3 | **Plausibility/proximity inside the selection rule (greedy-search-over-criteria).** Add plausibility (LOF) and/or proximity penalties to the candidate-selection objective, not just classifier probability (Łukasz's idea). | Meeting 2026-06-23 | Low | Explicitly "nie na teraz". Complicates the selection rule and couples it to extra models. | After the core mechanism is settled, extend the `prob_ascent` objective with weighted plausibility/proximity terms and ablate; relates to Backlog #1's combined-score selector. | OPEN |
| 4 | **Full-size Exp9 routing override estimate.** Re-run the HELOC routing audit at a statistically useful sample size (`--max-test 30`) and natural budget (`budget=17`) or implement an optimization that makes that cell tractable. | Stage 8 | Medium | The forced-numeric override repeatedly exhausted the greedy budget, triggering `prob_ascent`'s O(\|A\|²) candidate-scan cost with kNN context fits. Two attempts (`--max-test 30`, then `--max-test 5 --budget 5`) were stopped after no final artefact; only a one-row smoke result is committed. | Add candidate-score caching, parallel candidate evaluation, an early-stop/beam cap for Exp9, or run a long detached DGX job with sentinel logging. Then replace/augment `results/exp9_routing_summary.md` with a stable n=30 result. | OPEN |

Statuses: `OPEN` -> `IN_PROGRESS` -> `RESOLVED`. When an item is resolved, flip its
status and summarize the fix in **Fixed Issues**. Heavy items may warrant their own
follow-up plan — link it here.

---

## Decisions

Decisions made during planning (2026-06-21):

1. **Stopping oracle = the discriminator's flip** (both selectors). The loop ends when `disc.predict(x_cf) == y_target`; `--tau` (probability threshold, default 0.5 ≡ hard flip) and `--budget` (≤ |A|) are auxiliary stop knobs. Aligning the stop test with the validity metric is the SEDC/NICE convention.
2. **Two selection strategies are compared, never combined.** Strategy 1 (prob-ascent) and Strategy 2 (class-divergence) are alternative `select_candidate` rules sharing the same loop, single-column MAP value generation, and stop condition. The Stage-2 ablation picks one for the Stage-4 context ablation (default Strategy 1, which is compatible with all four context strategies).
3. **Committed value = near-MAP (`t ≈ 1e-9`)** so each greedy step is deterministic. Posterior-sample-and-pick-best is an off-by-default option.
4. **Context-strategy axis encodes both class scope and selection method** as a single 4-level categorical: `random_target`, `random_both`, `knn_target`, `knn_both`. This realizes the user's "{one class, both classes} × {random, nearest-neighbour}" cross exactly, where the NN "from target / from both" maps onto the class axis, and avoids nonsensical combinations.
5. **kNN distance = Euclidean over the factual point's features in MinMax-[0,1] space.** Context is selected once per query point from the factual row (before the greedy loop mutates it). Immutable-only distance is recorded as an alternative if full-vector NN underperforms.
6. **Strategy 2 is incompatible with target-only context strategies** (constant Y → undefined class contrast / TabPFN constant-feature validator). The selector ablation (Stage 2) runs Strategy 2 with a both-classes context; the context ablation (Stage 4) is run on the Stage-2 winning selector — if that is Strategy 2, the `*_target` grid cells are skipped with a logged note.
7. **Context size is capped at the available pool** and logged. MOONS train ≈ 800 rows (≈ 400 per class), so sizes 1024/2048 saturate on MOONS — the size axis is primarily informative on HELOC (≈ 8k train). Report the effective size per cell.

Decisions made during autonomous execution should be appended below.

**Stage 1 (2026-06-21):**

8. **`greedy_counterfactual` returns `(x_cf, changed, info)`** where `info` is a dict
   (`flipped: bool`, `steps: int`, `history: list`) rather than the bare per-step
   `history` list the stage sketch named. The runner and the budget-exhaustion test both
   need an explicit `flipped` flag, and the per-step `history` lives inside `info["history"]`.
   `steps == len(changed)` in this one-feature-per-step loop.

9. **MOONS validity ≈ 0.69 (failure_rate ≈ 0.31) is a recorded finding, not a bug.** The
   verification checklist's "MOONS validity ≈ 1.0" expectation was *not* met under the
   plan-mandated near-MAP (`t≈1e-9`) commit (Decision #3). With only 2 actionable features
   and deterministic mode-bucket commits, ~1/3 of boundary points plateau (no remaining
   feature produces a hard flip) and exhaust the budget — exactly the plateau the Success
   Criteria table anticipated for steepest-ascent. `steps_max=2` and `true_actionability=1.0`
   hold. HELOC, by contrast, is salvaged (validity 1.0, frac_oob 0.0, LOF 1.85, l0≈1.4 on
   the n=5 smoke set). The selector ablation (Stage 2) and context ablation (Stage 4) are
   the levers meant to probe MOONS validity; raising temperature or posterior-sample-best
   (off-by-default per Decision #3) are further options if needed.

10. **Execution environment.** `uv run` could not resolve dependencies in this offline
    sandbox (pre-existing `pyproject.toml` `[tool.uv] exclude-newer = "7 days"` parse error
    under the installed uv + a 401 on the private index — unrelated to this plan). Tests and
    runs were executed with the sibling TabPFN repo's fully-provisioned venv
    (`/Users/ofurman/pwr/TabPFN/.venv`), `PYTHONPATH` = this repo, `TABPFN_LOCAL_CACHE`
    pointed at the staged v2 checkpoints, and a **gitignored** symlink
    `experiments/zeroshot_cf/vendor → ../../../TabPFN/experiments/zeroshot_cf/vendor` so the
    `cel` dataset configs resolve. No code change was needed; the offline guarantee holds
    (no network, `get_models()` only, no `tabpfn_client`).

**Stage 2 (2026-06-22):**

11. **`class_divergence` divergence metric is heterogeneous across column types.** Regressor
    columns use the absolute bar-distribution mean-shift `|E[x_j|Y=t] − E[x_j|Y=c]|`;
    classifier-routed columns (HELOC low-cardinality integers, auto-detected by
    `infer_categorical_features`) use the **total-variation distance** between the two
    class-conditional proba vectors. Both are bounded in [0,1] so the per-step argmax over
    candidates is well-defined, but the two scales are not strictly identical — the
    expected-value approach was proven infeasible (the classifier's `classes_` are int-cast,
    destroying the MinMax support). See Fixed Issue #1. This only affects the
    `class_divergence` selector; `prob_ascent` (the expected Stage-4 winner) is unaffected.

12. **Heavy experiments run on the remote DGX `gx10-bdc5` (NVIDIA GB10), not local CPU.**
    Local imputes are ~1.3 s each on CPU, making the HELOC selector ablation (~1 h) and the
    Stage-4 16-cell context grid (many hours) impractical and incompatible with the
    `claude -p` per-stage runner (which can't survive a multi-hour background job). The
    branch is pushed to `origin`; the DGX clones it, provisions the env + v2 checkpoints, runs
    the experiments on GPU, and results are pulled back and committed. Env: `uv venv` py3.13,
    `torch 2.11.0+cu128` (Blackwell), `tabpfn` (this repo, editable), `tabpfn-extensions`,
    and `cel` vendored under `experiments/zeroshot_cf/vendor/counterfactuals`; checkpoints
    scp'd to `experiments/zeroshot_cf/models/`. Runs detached via `nohup` + a `*.DONE`
    sentinel, polled over SSH.

**Stage 4 (2026-06-22):**

13. **HELOC Stage-4 grid bounded to `--max-test 15`** (MOONS kept at n=100). At the full
    sizes {256…2048}, the size-2048 kNN cells (per-query context fit over a 2048-row pool)
    dominate cost; even at n=15 the 16-cell HELOC grid took ~5.3 h on the GB10. The plan
    permits reducing `--max-test` for runtime if logged. Consequence: HELOC validity is noisy
    at ±0.12 granularity, so the **`frac_oob`/`LOF` plausibility trends are the robust
    signal** (they are monotone and consistent); the recommended config is driven by those.

14. **Context ablation headline: relevant context, not large context.** Larger *random*
    context degrades HELOC plausibility and validity monotonically with size; **kNN context
    selection** is what helps, with `knn_both@256` achieving `frac_oob 0.000 / LOF 1.98`
    (every CF in-distribution) vs LOF 1e6–1e10 elsewhere. This is the lever that closes the
    HELOC plausibility gap the predecessor plan left open. See `results/exp6_summary.md` and
    REPORT.md §7c.

**Phase C — meeting follow-up (planning 2026-06-23):**

15. **Revisit policy = unlimited revisits + no-progress guard** (user decision). The greedy
    loop drops the one-change-per-feature exclusion entirely; any actionable feature is
    eligible every step; `budget` may exceed `|A|`. Termination: flip, `steps == budget`, or
    a no-progress stall (`prob_ascent`: best achievable `p_target` gain ≤ `--stall-eps`,
    default `1e-6`; `class_divergence`: same `j*` re-selected with value within `eps` →
    fixed point). Rationale: the meeting's hypothesis is that re-imputing a feature under the
    *updated* conditioning created by earlier changes unblocks MOONS boundary points that a
    single pass cannot move. Consequence: `l0_count` (distinct features) and `steps` (commits)
    now diverge — both reported; `l0_count` uses `len(set(changed))`.

16. **Discrete dataset uses TabPFN native categorical handling, not one-hot.** Prefer an
    existing CEL-configured categorical dataset (e.g. `german_credit` / `adult_census` /
    `bank_marketing`) only after adapting it into a native-categorical variant whose YAML omits
    `one_hot_encode`; synthesize an all-categorical dataset if the meeting's genuinely-discrete
    premise matters more than reusing public data. Categorical variables remain one semantic
    column each, encoded as stable integer/category codes for sklearn compatibility, and their
    indices are passed explicitly to TabPFN. This requires Stage-7 loader/runner plumbing:
    generic actionability config loading, Exp4 dataset-name support beyond `{moons,heloc,all}`,
    and sampler support for explicit categorical/numerical indices.

17. **Routing override goes through the sampler/runner, NOT `src/tabpfn`.** Forcing
    low-cardinality integer columns to the regressor (bar-distribution) head is done via the same
    explicit modality plumbing exposed as an Exp4/Exp9 `--force-numeric-cols` flag (default
    `none` = current behaviour). If TabPFN still auto-infers numeric low-cardinality columns as
    categorical despite explicit indices, configure the model inference settings for these runs
    (for example `MIN_UNIQUE_FOR_NUMERICAL_FEATURES=0`) and record the exact choice. The
    zero-architecture-change guarantee holds.

18. **Phase C scope.** Materialized as stages: budget+revisit (5), MOONS plots (6), discrete
    dataset (7), binning/routing audit (8), consolidated table (9). Deferred to Backlog:
    combined/MI/entropy selector (#1), newer TabPFN backbone (#2), plausibility-in-selection
    (#3) — all explicitly punted in the meeting. Heavy-experiment stages (5/7/8) run on the
    remote DGX per Decision #12; 6/9 are local viz/synthesis.

**Stage 5 (2026-07-05):**

19. **Exp7 saturation shortcut.** If a completed budget row has `steps_max < next_budget`,
    all later budgets are deterministic no-ops because every point has already flipped or
    stalled before the larger cap can bind; copy identical rows for the remaining budgets
    instead of rerunning them. This avoided rerunning HELOC budgets 34–1000 after budget 17
    (`steps_max=6`). MOONS was run before this shortcut and empirically matched across all
    budgets.

**Stage 6 (2026-07-06):**

20. **Blocked-slice panel may use a bounded fallback scan.** The main trajectory figure plots
    the requested near-boundary subset (`--max-test`, default 30). If those rows all flip, Exp8
    evaluates additional near-boundary rows up to `--fallback-pool` (default 100) only to find a
    representative stalled point for `moons_blocked_slice.png`. This keeps the headline figure
    readable while satisfying the stage requirement that the blocked-slice panel show an actual
    stalled case when one exists in the bounded pool.

**Stage 7 (2026-07-06):**

21. **Discrete sanity dataset = synthetic `binary_cat`.** Rather than adapting a CEL config whose
    preprocessing conventions are one-hot/continuous oriented, Stage 7 uses a small deterministic
    all-categorical synthetic dataset with three binary semantic columns and label
    `Y=decision_code`. Codes stay as integer 0/1 values (no scaling, no one-hot); all three
    feature indices are passed explicitly as categorical to TabPFN. `segment_code` is immutable,
    `decision_code`/`channel_code` are actionable. This directly tests the meeting's genuinely
    discrete premise while keeping `frac_oob` meaningful under the existing [0,1] metric.

22. **Native-categorical `prob_ascent` uses all-classes context.** Target-only categorical
    context caused degenerate one-class feature support and asymmetric commits (Fixed Issue #6).
    For datasets with explicit categorical features, Exp4 now keeps both classes in context and
    relies on the appended Y column for class conditioning. Continuous MOONS/HELOC defaults are
    unchanged.

**Stage 8 (2026-07-06):**

23. **Routing override is inference-filtered, not architecture-level.** The sampler leaves
    `src/tabpfn/**` untouched. When `force_numeric_cols` is non-empty, it filters only those
    original feature indices out of the unsupervised wrapper's inferred categorical list for the
    current `fit()`, while preserving explicit non-forced categorical columns and the appended Y
    condition column. This keeps `--force-numeric-cols none` byte-identical to current routing and
    lets Exp9 force selected HELOC low-cardinality columns through the regressor/bar path.

24. **Exp9 committed result is a bounded smoke, with the full estimate deferred.** Full HELOC
    `max_test=30` at `budget=17` and a smaller `max_test=5, budget=5` attempt both hit the
    expected `prob_ascent` O(\|A\|²) worst case after forced-numeric routing caused rows to consume
    their budget. The committed Exp9 artefact is therefore explicitly labeled as a smoke diagnostic
    (`max_test=1, budget=1, n_permutations=1`), sufficient to verify the override path and direction
    on one row but not a stable HELOC estimate. The stable run is Backlog #4.
