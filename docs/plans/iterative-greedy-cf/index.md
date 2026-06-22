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

---

## Files That May Be Changed

### New experiment code (under `experiments/zeroshot_cf/`)
- `greedy.py` — the iterative greedy loop + the two `select_candidate` strategies (Stage 1).
- `exp4_greedy_cf.py` — greedy CF runner with `--selector` and stop/budget flags + L0/steps metrics (Stage 1).
- `exp5_selector_ablation.py` — Strategy 1 vs Strategy 2 grid driver (Stage 2).
- `exp6_context_ablation.py` — size × context-strategy grid driver (Stage 4).
- `tests/test_greedy.py` — greedy loop + selector unit tests (Stage 1).
- `tests/test_context.py` — kNN context-selection unit tests (Stage 3).

### Modified
- `sampler.py` — add `fixed_target` pass-through to `sample_feature()` so the greedy commit can sample class-conditionally (Stage 1, prerequisite — default `None` preserves existing behaviour); add `predictive_distribution()` helper (Stage 1); extend `set_context()` with `selection={random,knn}` and `pool={target,both}` (Stage 3). Existing behaviour (random subsample, `append_target=False` sampling) preserved as the default.
- `tests/conftest.py` — host the shared `models` fixture (lifted from `test_sampler.py`/`test_ordering.py` in Stage 1; today `conftest.py` only does `sys.path` setup).
- `results/REPORT.md` (this plan's results section), `results/exp4_*`, `results/exp5_*`, `results/exp6_*` CSV/MD artefacts.

> **Artefact path convention**: every `results/...` path in this plan resolves to `experiments/zeroshot_cf/results/` (exp2's `RESULTS_DIR = Path(__file__).parent / "results"` convention), **not** a repo-root `results/`. The existing report is `experiments/zeroshot_cf/results/REPORT.md`.

> The core `src/tabpfn/**` package is **not** modified (zero architecture changes). No `tabpfn_client` / cloud API anywhere.

---

## Progress Tracker

| # | Stage | Status | Notes | Commit |
|---|-------|--------|-------|--------|
| 1 | [Iterative greedy core + both selectors](stages/01-greedy-core.md) | DONE | greedy loop + both selectors + `predictive_distribution`/bar helpers + Exp4 runner + 7 new tests (20/20 pass). HELOC smoke (n=5): validity 1.0, frac_oob 0.0, LOF 1.85, l0≈1.4 — salvages HELOC. MOONS validity ≈0.69 (near-MAP plateau, anticipated finding — see Decision #8). | 5dfcaba |
| 2 | [Selector ablation (Strategy 1 vs 2)](stages/02-selector-ablation.md) | IN_PROGRESS | | |
| 3 | [kNN / context-selection support](stages/03-knn-context-selection.md) | DONE | `set_context` gains `selection={random,knn}` + `query` (kNN anchor); module-level `_knn_indices`; default `random` path byte-identical. `test_context.py` (6 cases a–e). Full suite 30 passed. Committed before Stage 2 since both touch `sampler.py` independently (Stages 2/3 are mutually independent per plan). | (see git log) |
| 4 | [Context ablation (size × strategy)](stages/04-context-ablation.md) | PENDING | | |

Statuses: `PENDING` -> `IN_PROGRESS` -> `DONE` | `BLOCKED` | `SKIPPED`

Phases: **A = Stage 1 (mechanism)**, **B = Stages 2–4 (ablations)**. Stage 1 gates all others. Stages 2 and 3 are mutually independent; Stage 4 depends on both. See `resources/grids.md` for the exact ablation grids and `resources/commands.md` for run commands.

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
- **Do not modify `src/tabpfn/**`** — the whole premise is zero architecture changes.

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
