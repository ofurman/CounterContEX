# Plan: Zero-Shot Autoregressive Feature Estimation with TabPFN (Local Inference)

**Date**: 2026-06-15
**Branch**: `zeroshot-tabpfn-cf` (create from `main`)
**Predecessors**: None
**Goal**: Test, fully offline, whether a pre-trained TabPFN v2 model used as a conditional density estimator can (1) reconstruct a single masked feature (sanity check) and (2) generate actionable counterfactuals by masking actionable features, freezing immutable ones, and conditioning on a target class — with **no retraining or architecture changes**.

---

## Context

We are pivoting to test TabPFN's **built-in** feature-estimation ability out-of-the-box, relying entirely on local open-source checkpoints (TabPFN v2 weights) on our own hardware — **no cloud API**. Rather than modifying the architecture or retraining, we use the pre-trained model as a conditional density estimator and generate features autoregressively.

### Why this is feasible (research findings)

The mechanism already exists in `tabpfn-extensions`. We compose three existing pieces:

1. **Local TabPFN core** (`/Users/ofurman/pwr/TabPFN`, v8.0.8). `TabPFNRegressor.predict(X, output_type="full")` returns a dict with `"criterion"` (a `FullSupportBarDistribution`) and `"logits"`. Sampling from the conditional density is `criterion.sample(logits, t=temperature)`; log-density is `criterion.forward(logits, y)` (NLL). Runs offline via `model_path=` or the `TABPFN_MODEL_CACHE_DIR` env var; uses MPS on Apple Silicon (`device="auto"`). No generative code in core — this is a downstream application.

2. **`tabpfn-extensions` unsupervised module** — the script we reuse (per the task brief). `TabPFNUnsupervisedModel(tabpfn_clf, tabpfn_reg)`; `.fit(X_context)` stores the **entire** context matrix as the conditioning set; `.impute(X_with_NaNs, t=…, n_permutations=…)` fills **only the NaN cells**, conditioning each masked cell on the observed cells **in the same row**, autoregressively over random feature orderings (or a `dag`). This is exactly masked conditional feature estimation.
   - **No native Y-conditioning.** The idiomatic workaround: append the target Y as an extra (categorical) column to the context, mark it categorical via `set_categorical_features`, fit on the augmented matrix, then at impute time **fix the Y column to the desired target class** (observed) and NaN-mask the actionable feature columns. Because impute conditions on all observed columns, this yields class-conditional feature generation `p(actionable | immutable, Y=target)`.
   - Knobs: `t` (temperature; `1.0` for generation, `1e-9` near-MAP for imputation — only affects numerical/regressor columns), `n_permutations` (Monte-Carlo over feature orderings), optional `dag` (parent-conditioning).

3. **`ofurman/counterfactuals`** (Python package **`cel`**) — datasets + evaluation harness.
   - **HELOC**: 23 continuous features, target `RiskPerformance` (`{Bad:0, Good:1}`). `data/heloc.csv`, `config/datasets/heloc.yaml`. **All 23 features are configured actionable** — there is no immutable subset, so we must define one (see Decisions).
   - **MOONS**: 2 continuous features (cols `"0"`,`"1"`), target col `"2"`, subsampled to 1000 rows. `data/moons.csv`, `config/datasets/moons.yaml`.
   - Split: 80/20 stratified, `random_state=42`. Default preprocessing: MinMax→[0,1] on continuous features, fit on train.
   - **Metrics (registry path / `evaluate_cf`)** — our subset:
     - `validity` (`cel/metrics/basic_metrics.py`): fraction where `disc_model.predict(X_cf) != y_test`. Needs a `disc_model` with `.predict()` + `.eval()`.
     - `lof_scores_cf` (`cel/metrics/plausibility.py`): `LocalOutlierFactor(n_neighbors=20, novelty=True)` fit on `X_train`, returns `(-score_samples(X_cf)).mean()` — **lower = more plausible**.
     - `sparsity` (`basic_metrics.py`): `(X_test != X_cf).mean()` — mean fraction of entries changed (lower = sparser).
     - `actionability` (`basic_metrics.py`): **caveat** — this metric is mislabeled; it computes `np.all(X_test==X_cf, axis=1).mean()` (fraction of *unchanged* CFs), NOT immutable-constraint compliance. We will additionally compute a **true actionability** check (immutable columns unchanged) ourselves.
     - `proximity_l2_jaccard` (`cel/metrics/distance.py`): for these all-continuous datasets reduces to **pure mean per-instance L2** between factual and valid CF.
   - Harness entry: `evaluate_cf(disc_model, gen_model, X_cf, …, y_target=…)` builds a `MetricsOrchestrator` and returns a dict; metrics with unmet `required_inputs` are skipped with a warning, so we can run our subset without a generative model.

### Risks (from the brief)

- **Validity**: model was not trained for CF generation; out-of-the-box validity may be low.
- **Proximity vs. plausibility**: nothing enforces minimal perturbation. Generated features are plausible under the target distribution but not guaranteed to be the *closest* CF. Refinement (context selection, temperature, ordering) is the lever if results are weak.

---

## Strategy

Two sequential experiments behind shared infrastructure, all offline.

- **Phase A — Infrastructure (Stages 1–3)**: offline checkpoints + deps; the cel dataset/discriminator/metrics harness; a reusable `ConditionalDensitySampler` wrapper around `TabPFNUnsupervisedModel` that handles context selection, the Y-as-column trick, and masked imputation.
- **Phase B — Experiments (Stages 4–5)**:
  - **Experiment 1 (Stage 4, sanity check)**: mask one feature of a factual point, reconstruct it from the target-class context, evaluate reconstruction quality. Gate: if TabPFN cannot sensibly reconstruct single features, Experiment 2 is unlikely to work.
  - **Experiment 2 (Stage 5, counterfactuals)**: freeze immutable features, mask actionable ones, fix Y=target class, impute, assemble CFs, evaluate the 5 metrics on HELOC + MOONS.
- **Phase C — Refinement & reporting (Stage 6)**: if initial validity/proximity is weak, sweep the **inference** process only (context selection strategy & size, temperature `t`, `n_permutations`, feature ordering). Produce a results report.

This is **strictly in-context, zero retraining**. All tuning is inference-side.

---

## Success Criteria

Targets are deliberately modest — this is an exploratory out-of-the-box test, and the brief flags validity as the main risk. "Success" = a working, fully-offline pipeline that produces interpretable metrics, plus an honest read on whether the approach is promising.

| Metric | Baseline | Target | Rationale |
|--------|----------|--------|-----------|
| Pipeline runs fully offline | n/a | Yes | No network calls; checkpoints loaded from local cache. |
| Exp 1 single-feature reconstruction (MOONS) | random-guess MSE | Reconstructed feature MSE clearly below the marginal-mean baseline | Sanity check that conditional density is informative. |
| Exp 1 single-feature reconstruction (HELOC) | marginal-mean MSE | Beats marginal-mean baseline on a majority of features | Confirms conditioning helps on a high-dim real dataset. |
| Exp 2 validity (MOONS) | — | ≥ 0.7 | 2-D, easy class structure; high validity expected if mechanism works. |
| Exp 2 validity (HELOC) | — | ≥ 0.5 (record actual) | Out-of-the-box on 23-D; record honestly even if lower. |
| Exp 2 LOF plausibility | cel baselines (record) | Competitive with / better than cel baselines | Density-driven generation should be plausible by construction. |
| Exp 2 actionability (true, immutable check) | — | 1.0 | We freeze immutable columns by construction → must be exactly preserved. |
| Exp 2 proximity (L2) & sparsity | cel baselines (record) | Report; not expected to win | Brief flags proximity as a known weakness; refinement may help. |

Baselines from cel for HELOC/MOONS (PPCEF, DiCE, etc.) are recorded in `resources/api-reference.md` once read off the repo, for side-by-side context.

---

## Files That May Be Changed

### New experiment code (in TabPFN repo)
- `experiments/zeroshot_cf/__init__.py` -- package marker.
- `experiments/zeroshot_cf/checkpoints.py` -- offline checkpoint staging/loading helpers.
- `experiments/zeroshot_cf/data.py` -- cel dataset loading (HELOC, MOONS), actionability spec.
- `experiments/zeroshot_cf/discriminator.py` -- train/wrap cel disc_model (LR/MLP) as validity oracle.
- `experiments/zeroshot_cf/sampler.py` -- `ConditionalDensitySampler` wrapping `TabPFNUnsupervisedModel`.
- `experiments/zeroshot_cf/metrics_harness.py` -- adapter to cel `evaluate_cf` + true-actionability metric.
- `experiments/zeroshot_cf/exp1_single_feature.py` -- Experiment 1 runner.
- `experiments/zeroshot_cf/exp2_counterfactuals.py` -- Experiment 2 runner.
- `experiments/zeroshot_cf/refine.py` -- inference sweeps (Stage 6).
- `experiments/zeroshot_cf/configs/` -- run configs (datasets, immutable splits, sweep grids).
- `experiments/zeroshot_cf/results/` -- output metrics/plots (gitignored data, committed report).
- `experiments/zeroshot_cf/README.md` -- how to run, offline setup, results summary.
- `experiments/zeroshot_cf/requirements.txt` -- pinned `tabpfn-extensions`, `cel` install refs.

### Possibly touched
- `.gitignore` -- ignore large checkpoint files / results artifacts under the experiments dir.

> The core `src/tabpfn/**` package is **not** modified (zero architecture changes).

---

## Progress Tracker

| # | Stage | Status | Notes | Commit |
|---|-------|--------|-------|--------|
| 1 | [Environment & offline checkpoint setup](stages/01-environment-offline-setup.md) | DONE | TabPFN v2 (no-license); cel vendored editable; smoke test passed offline | 2868713 |
| 2 | [Data, discriminator & metrics harness](stages/02-data-discriminator-metrics.md) | DONE | HELOC 23-feat/MOONS 2-feat loaded; 6-feature immutable split; sklearn LR oracle (MOONS 87%, HELOC 72%); direct metric computation (no orchestrator stub needed) | 57b18d0 |
| 3 | [Conditional density sampler wrapper](stages/03-conditional-density-sampler.md) | DONE | ConditionalDensitySampler: set_context, impute_masked, sample_feature; explicit v2 model_path fix in get_models(); 4 tests all PASS (MSE 0.0036 vs 0.2014 baseline) | f86e488 |
| 4 | [Experiment 1: single-feature estimation](stages/04-exp1-single-feature.md) | DONE | MOONS: WEAK (1/2 beats marginal, calib 0.70); HELOC: PASS (15/23=65% beats marginal, calib 0.60). Overall gate: PASS — proceed to Stage 5 | ae0e555 |
| 5 | [Experiment 2: counterfactual generation](stages/05-exp2-counterfactual-generation.md) | DONE | MOONS: validity=0.85 LOF=1.055 true_action=1.0; HELOC: validity=0.66 LOF=2.5B (66% OOB extrapolation) true_action=1.0. Validity targets met; HELOC plausibility poor due to sparse conditioning (17/23 features masked). Cel baselines deferred to Stage 6. | 2525e27 |
| 6 | [Refinement & results report](stages/06-refinement-and-report.md) | DONE | MOONS sweep: t=0.5+all_classes best (prox=0.629, validity=0.783). HELOC sweep: no config fixes OOB (MAP gives OOB=100%; root cause is sparse conditioning, not temperature). REPORT.md written; README updated; key learnings recorded in results/REPORT.md and Decisions section of this index. | 1c415ca |
| 7 | [Post-review correctness & reproducibility fixes](stages/07-postreview-fixes.md) | DONE | P1: validity now scores disc(X_cf)==y_target; LOF on unclipped X_cf; CLI flags added to exp2. P2: RNG seeding in sampler; immutability assert. P3: sample_temperature arg; scatter uses per-class context; api-reference TODO cleared; Stage6 memory claim resolved. 8/8 tests pass. | 319b05a |
| 8 | [Regenerate results & report](stages/08-regenerate-results.md) | DONE | All exp1/exp2/sweep CSVs regenerated; sampler RNG seeding fix (calibration 0→0.69/0.62); corrected MOONS validity 0.85→1.0, HELOC 0.66→0.52; REPORT/README/notebook rebuilt; memory updated. | 9c74cba |
| 9 | [Feature-ordering (DAG) ablation](stages/09-feature-ordering-ablation.md) | DONE | build_chain_dag + dag kwarg in sampler; exp2 --ordering/--actionable-set/--reduced-k flags; exp3_feature_ordering.py grid runner; test_ordering.py (5 new tests, 13/13 pass). MOONS: dag≈random (validity 0.93 vs 0.96). HELOC: dag/full improves validity (0.55 vs 0.40) but raises OOB; dag/reduced best cell (validity=0.50, frac_oob=0.10). Context_type=all_classes required (Decision #10). | 1c65db4 |

Statuses: `PENDING` -> `IN_PROGRESS` -> `DONE` | `BLOCKED` | `SKIPPED`

Phases: A = Stages 1–3 (infra), B = Stages 4–5 (experiments), C = Stage 6 (refinement), **D = Stages 7–8 (post-review remediation)**, **E = Stage 9 (feature-ordering ablation)**. Stage 4 is a **gate** for Stage 5: if single-feature reconstruction is no better than the marginal baseline, note it prominently before proceeding — Stage 5 still runs but expectations are adjusted. Stage 9 is an independent follow-up ablation; it depends only on Stages 1–8 being DONE.

> **Phase D context (added 2026-06-15 after `/plan-post-review`)**: a post-implementation review found 3 P1, 4 P2, 5 P3 issues. The most important: Exp2 **validity is scored against the wrong reference** (`y_test` instead of the generation target `y_target = 1 - y_pred`), so the headline Exp2 numbers must be recomputed before they can be cited. Stage 7 fixes the code; Stage 8 regenerates the artifacts. Full findings are reproduced in `stages/07-postreview-fixes.md`.

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
- **Offline guarantee**: every run must work with no network. If a step appears to need a
  download (checkpoints, datasets, deps), it belongs in Stage 1's pre-staging — defer
  rather than reach for the network mid-experiment.
- **Do not modify `src/tabpfn/**`** — the whole premise is zero architecture changes.

---

## Fixed Issues

Problems encountered during execution and resolved (in place or via a fix subagent).
Leave empty until execution surfaces something.

| # | Stage | Symptom | Root Cause | Resolution | Fixed By |
|---|-------|---------|-----------|------------|----------|
| 1 | 3 | `TabPFNLicenseError: model weights for v3` during unsupervised impute | `settings.tabpfn.model_version` defaults to V3 at import time; `TABPFN_MODEL_VERSION=v2` env var set after import has no effect on already-instantiated pydantic-settings object | Updated `get_models()` to pass explicit `model_path=<v2_ckpt_file>` so `_resolve_model_version` reads version from filename, bypassing the settings object | inline |
| 2 | 8 | Exp1 calibration collapsed to 0.00 (was 0.70) after Stage 7 RNG seeding fix | `impute_masked()` re-seeds from `self.random_state` on every call; all N posterior samples in `sample_feature(n_samples>1)` are identical (IQR=0 → calibration=0.00) | In `sample_feature`, offset `self.random_state` by sample index `i` inside the multi-sample loop so each draw uses a distinct seed | inline |

---

## Backlog (Deferred Issues)

Problems deferred for later — too heavy to fix inline without derailing the plan.
Each entry must be **self-contained enough for a future run to pick it up cold**:
state the symptom, where it came from, and a concrete lead for resolving it.

| # | Title | Origin Stage | Severity | Why Deferred | Suggested Next Step | Status |
|---|-------|--------------|----------|--------------|---------------------|--------|
| 1 | Record cel baselines (PPCEF/DiCE) for HELOC/MOONS | 5/6 | P2 (med) | cel CF-method baselines need the heavy TF/alibi dependency chain (no Py3.13 wheels) and per-method training; out of scope for the offline zero-shot test | Run cel's `run_ppcef_pipeline.py` (+ DiCE) in a Py3.10 TF-compatible env, capture validity/proximity/sparsity/LOF for both datasets, fill `resources/api-reference.md §cel baseline numbers`, add the side-by-side column to `results/REPORT.md §5`. | OPEN |

Statuses: `OPEN` -> `IN_PROGRESS` -> `RESOLVED`. When an item is resolved, flip its
status and summarize the fix in **Fixed Issues**. Heavy items may warrant their own
follow-up plan — link it here.

---

## Decisions

Decisions made with the user during planning (2026-06-15):

1. **Validity oracle = cel repo `disc_model` (LR/MLP).** Counterfactual validity is judged by the counterfactuals repo's standard discriminator, for comparability with its existing CF-method baselines (PPCEF, DiCE, …). Note: this means TabPFN-generated CFs are judged by a *different* model family than the one that generated them — record this caveat in the report.
2. **HELOC actionability = domain-based immutable subset.** Freeze realistically non-actionable / history-length features and mask the rest. Proposed immutable set (history/age fields that a person cannot directly act on): `MSinceOldestTradeOpen`, `MSinceMostRecentTradeOpen`, `AverageMInFile`, `NumTotalTrades`, `MSinceMostRecentDelq`, `MSinceMostRecentInqexcl7days`. The remaining ~17 (balances, utilization, inquiry counts, delinquency rates, install-trade %) are actionable. This split is a documented judgment call — finalize the exact list in Stage 2's config and record it. MOONS: both features actionable (no immutables).
3. **Exp 2 mechanism = reuse `TabPFNUnsupervisedModel.impute`** with the Y-as-appended-categorical-column trick (fix Y=target, NaN-mask actionable features). Directly reuses the `tabpfn-extensions` scripts as instructed.
4. **Code location = new `experiments/zeroshot_cf/` directory** inside the TabPFN repo, importing `tabpfn-extensions` and `cel` as dependencies. Keeps the work next to the local checkpoints.

Decisions made during autonomous execution should be appended below.

5. **Use TabPFN model version "v2" for all experiments.** TabPFN v3 (the default) requires a TABPFN_TOKEN license-acceptance API call before downloading; this is unavailable in an unattended/offline environment. TabPFN v2 models download directly from HuggingFace with no license gate and still expose the full conditional-density API (`output_type="full"`, `criterion.sample()`). `checkpoints.py` enforces this by setting `TABPFN_MODEL_VERSION=v2` before constructing any model.

6. **`cel` installed as editable vendor package with `--no-deps` (Python 3.13 / TensorFlow incompatibility).** The `ce-library` package depends on TensorFlow, which does not have Python 3.13 wheels. Vendored the repo at `experiments/zeroshot_cf/vendor/counterfactuals/`, installed it editable with `--no-deps`, then installed required transitive deps (cel-nflows, torchdiffeq, UMNN, omegaconf, hydra-core) one by one. The `cel/__init__.py` was patched to make CF-method imports optional (try/except) so `cel.datasets` and `cel.metrics` load without nflows/alibi chain. `cel.models` (discriminator LR/MLP) may need further dep-install if needed in Stage 2.

7. **Discriminator = sklearn `LogisticRegression` wrapped with `.eval()` no-op (Stage 2).** cel's `LogisticRegression` model requires PyTorch DataLoaders and epoch-based training, adding unnecessary complexity. sklearn LR achieves equivalent accuracy (MOONS 87%, HELOC 72%) with no boilerplate. Wrapped in `DiscriminatorModel` to satisfy the cel metrics contract.

8. **Metrics harness computes metrics directly, bypassing `MetricsOrchestrator` and `evaluate_cf` (Stage 2).** The orchestrator unconditionally calls `gen_model.eval()` — requiring a stub even if gen_model is unused. The registered `proximity_l2_jaccard` metric would compute `0 * NaN` for empty categorical features. Direct computation via sklearn/numpy is cleaner and avoids both issues.

9. **`get_models()` passes explicit `model_path` for v2 checkpoints (Stage 3 fix).** `settings.tabpfn.model_version` is a pydantic-settings object instantiated at import time with default V3. Setting `TABPFN_MODEL_VERSION=v2` env var after import has no effect. Fix: pass `model_path=cache_dir/<v2-filename>` directly to `TabPFNClassifier/Regressor` constructors so `_resolve_model_version` reads the version from the filename rather than the settings object.

10. **Exp3 (Stage 9) uses `context_type="all_classes"` for all cells, including the random baseline.** The DAG construction places Y as an explicit conditioning parent; with `target_only`, Y is constant in context (single class) → TabPFN's constant-feature validator raises `TabPFNValidationError`. Switching to `all_classes` makes Y informative and preserves the within-exp3 random/dag comparison. Stage-8 results (with `target_only`) remain the reference for the recommended production configuration; exp3 numbers are not directly comparable to Stage-8's.
