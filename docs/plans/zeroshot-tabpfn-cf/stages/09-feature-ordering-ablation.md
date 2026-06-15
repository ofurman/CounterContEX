# Stage 9: Feature-Ordering (DAG) Ablation

**Goal**: Test whether replacing `impute`'s random-permutation averaging with an explicit, fixed autoregressive **DAG** ordering — Y first, then the frozen immutables, then the actionable features last (in a deterministic chain) — changes counterfactual quality; and crucially, whether it helps when paired with a **reduced actionable set** (the factor that actually targets HELOC's sparse-conditioning failure).
**Dependencies**: Stages 1–8 DONE. Reuses `ConditionalDensitySampler`, `exp2`'s CLI/metrics, and the cel discriminator/metrics harness unchanged.

> **Why this stage exists — and an honest framing of the hypothesis.**
> The follow-up request was: *"change feature ordering — Y label should always be the first feature, followed by non-actionable known values, with actionable features (the ones to predict) at the end."*
>
> Reading the actual `impute` implementation (`.venv/.../tabpfn_extensions/unsupervised/unsupervised.py`) clarifies what this can and cannot do:
> - In the **current** path (`dag=None` → `condition_on_all_features=True`, `unsupervised.py:379`), every masked cell already conditions on **all** other columns, and observed columns (Y + immutables) are skipped in the fill loop (`:456`) — so they are *always* conditioning parents **regardless of their position**. Putting Y/immutables "first" is therefore a **no-op** in this path; only the relative order of the *masked* actionable columns has any effect, and that effect is averaged away over `n_permutations` random orderings.
> - The literal request becomes meaningful only by switching to the **DAG path** (`impute(..., dag=…)`, which auto-sets `condition_on_all_features=False`, `:701`). There, `conditional_idx = full_dag[column_idx]` (`:375`) — each actionable conditions on exactly the parents we declare. We can then impose the requested factorization:
>   `p(A₁ | Y, immutables) · p(A₂ | Y, immutables, A₁) · … · p(Aₖ | Y, immutables, A₁..Aₖ₋₁)`.
>
> **What to expect (state this in the report, do not oversell):**
> - This is a test of **deterministic structured ordering vs random-permutation averaging** — it primarily affects *coherence among generated features* and possibly sparsity, **not** the conditioning *quantity*.
> - It will **not** fix HELOC's out-of-distribution failure. HELOC breaks because 17 masked features are conditioned on only 6 immutables + Y (under-determination → 72% OOB). The DAG gives each actionable a parent set that is a **subset** of the current all-observed-plus-filled-siblings set, so DAG-alone is expected to be **neutral-to-worse** on HELOC.
> - The lever that targets HELOC is the **reduced actionable set** (fewer masked → denser conditioning). The DAG is tested *in combination* with it.

**Guardrail reminder**: do NOT modify `src/tabpfn/**`. Runs must work offline (checkpoints already staged). Be honest about deltas in the report — if DAG ordering does not beat the random-permutation baseline, say so plainly.

---

## Design: a two-factor ablation

| Factor | Levels | Notes |
|--------|--------|-------|
| **Ordering** | `random` (baseline = current Stage-5/8 behavior) · `dag` (Y → immutables → actionable chain) | The requested change. |
| **Actionable set** | `full` · `reduced` (HELOC only) | The lever that can actually move HELOC OOB. MOONS has only 2 actionable features → `reduced` ≡ `full` (no-op; skip the duplicate cell). |

Grid: MOONS = {random, dag} × {full} = 2 runs. HELOC = {random, dag} × {full, reduced} = 4 runs. Everything else (temperature, n_permutations, context strategy, max_context, n_estimators, sample size) is held at the **Stage-8 baseline defaults** so ordering is the only thing that varies within a column.

### DAG construction (augmented index space)

The sampler appends Y as the **last** column (index `Y_idx = n_original_features`); `set_categorical_features([Y_idx])` is already called before fit. Feature/immutable indices keep their original 0..d−1 values. For an ordered list of actionable indices `ordered = [a₁, …, aₖ]`:

```python
dag = {}
for i, a in enumerate(ordered):
    dag[a] = [Y_idx] + list(immutable_idx) + ordered[:i]   # parents: Y, all immutables, earlier actionables
# Y and immutables get no entry → empty parent lists (roots). They are observed (non-NaN),
# so impute() skips imputing them and only uses them as parents. _resolve_dag_order fills
# the empty deps and topologically sorts → Y/immutables first, actionables in chain order.
```

This realizes the requested ordering exactly: **Y first, immutables next (as known parents), actionable features generated last** in a fixed chain where each new actionable also sees the ones already generated.

- **Actionable chain order (primary)**: dataset feature-index order (deterministic, reproducible). MOONS: `[0, 1]`.
- **Reduced HELOC actionable set**: pick the **top-6** actionable features by `|coef|` of the trained LR discriminator (most class-relevant, reproducible from the fitted oracle). Mask only those; **all other non-immutable features are frozen at their factual values** (treated as observed, like immutables). Log the chosen 6 indices/names. (6 ≈ matches MOONS-like density: 6 masked vs 6 immutables + Y observed.)

---

## Steps

1. **Add DAG support to the sampler.**
   - File: `experiments/zeroshot_cf/sampler.py`, `impute_masked()` (≈ line 139) and `__init__`.
   - Add an optional `dag: Optional[Dict[int, List[int]]] = None` parameter to `impute_masked` (default `None` preserves exact current behavior). When non-`None`, pass it through to the underlying call: `self.model.impute(X_aug, t=self.temperature, n_permutations=self.n_permutations, dag=dag)`. Do **not** also pass `condition_on_all_features` — `impute()` derives it from `dag` (`unsupervised.py:701`).
   - The `dag` keys/values are in **augmented** index space; the caller builds it (next step). Add a one-line assertion that every dag index is `< X_aug.shape[1]` to catch off-by-one vs the appended Y column.
   - Keep the existing RNG re-seeding and the "restore non-masked columns" / "drop appended target" logic unchanged.

2. **Add a DAG builder helper.**
   - File: `experiments/zeroshot_cf/sampler.py` (module-level function) or a small `ordering.py`.
   - `def build_chain_dag(ordered_actionable, immutable_idx, y_idx) -> dict[int, list[int]]` implementing the construction above. Unit-test it (Step 6).

3. **Add ablation flags to the Exp2 runner.**
   - File: `experiments/zeroshot_cf/exp2_counterfactuals.py`.
   - Add argparse flags: `--ordering {random,dag}` (default `random`) and `--actionable-set {full,reduced}` (default `full`), plus `--reduced-k` (default `6`, HELOC only).
   - When `--ordering dag`: compute `Y_idx = n_original_features`, determine the ordered actionable list (index order), build the dag via `build_chain_dag`, and pass it into `impute_masked(..., dag=dag)`. When `random`: pass `dag=None` (unchanged path).
   - When `--actionable-set reduced` (HELOC): restrict `mask_cols` to the top-`reduced_k` actionable features by `|LR coef|` (read from the already-trained discriminator); freeze the rest at factual values. Log the selected feature names/indices. For MOONS, `reduced` is rejected/skipped with a clear log line (only 2 actionable features).
   - The immutability `assert` (Stage 7) must still pass — with `reduced`, the *frozen* non-immutable features must also remain unchanged; assert on the union of frozen columns.

4. **Add the ablation driver.**
   - New file: `experiments/zeroshot_cf/exp3_feature_ordering.py`.
   - Runs the grid for a dataset (MOONS: 2 cells; HELOC: 4 cells) by invoking the Exp2 generation+metrics path for each (ordering, actionable-set) combination at the Stage-8 baseline config.
   - Writes `results/exp3_ordering_{moons,heloc}.csv` — one row per cell with columns: `ordering, actionable_set, n_masked, validity, lof_scores_cf, sparsity, true_actionability, proximity_l2_jaccard, frac_oob, runtime_s`.
   - Writes `results/exp3_summary.md` with a per-dataset comparison table and a short honest verdict: did `dag` beat `random` at equal actionable-set size? Did `reduced` move HELOC validity/OOB? Explicitly state whether the requested Y-first ordering changed anything and why (referencing the no-op-in-impute / meaningful-in-DAG mechanism).

5. **Run the ablation (offline).**
   - `uv run python experiments/zeroshot_cf/exp3_feature_ordering.py --dataset moons`
   - `uv run python experiments/zeroshot_cf/exp3_feature_ordering.py --dataset heloc`
   - Use the same sample size as Stage 8's Exp2 for comparability. If runtime is a concern (DAG runs the inner loop per actionable in topo order), it is acceptable to reduce the HELOC sample size — but log it and keep it identical across the 4 HELOC cells so the comparison is fair.

6. **Tests.**
   - File: `experiments/zeroshot_cf/tests/test_ordering.py`.
   - Test `build_chain_dag`: (a) Y and immutables are roots / absent from keys; (b) each actionable's parents = `[y_idx] + immutable_idx + earlier actionables`; (c) the dict is acyclic and its topological order places Y+immutables before all actionables.
   - Add a sampler-level test (small synthetic context, `FAST_TEST_MODE` ok): `impute_masked(..., dag=...)` returns the same shape as the `dag=None` path, leaves non-masked columns byte-identical, and fills all masked cells (no NaN remains).
   - All prior tests (8/8) must still pass.

7. **Report & index.**
   - Add an **"Experiment 3: feature-ordering ablation"** section to `results/REPORT.md` summarizing the 2×(2/4) grid, with the honest framing from the rationale blockquote (no-op-in-impute vs meaningful-in-DAG; DAG targets coherence, reduced-set targets OOB).
   - Optionally extend `build_notebook.py` to render the exp3 CSVs (deterministic, CSV-driven — no model re-runs). If skipped for budget, `log()` it.
   - Update this plan's progress tracker (mark Stage 9 DONE with the headline deltas) and the Phases note.

---

## Verification

- [ ] `uv run python experiments/zeroshot_cf/exp2_counterfactuals.py --help` lists `--ordering`, `--actionable-set`, `--reduced-k`.
- [ ] `uv run pytest experiments/zeroshot_cf/tests -q` passes, including the new `test_ordering.py`; prior tests still pass.
- [ ] `results/exp3_ordering_{moons,heloc}.csv` exist; HELOC has 4 rows, MOONS has 2; `n_masked` is constant within each HELOC `actionable_set` level.
- [ ] `true_actionability == 1.0` for every cell (immutables **and** any frozen-reduced columns unchanged — the assert held).
- [ ] `results/exp3_summary.md` states plainly whether `dag` beat `random` at equal actionable-set size, and whether `reduced` moved HELOC validity/OOB — even if the answer is "no".
- [ ] `git diff --name-only main..HEAD -- src/tabpfn` is empty (core untouched).

---

## Expected outcomes (record actuals against these)

- **MOONS**: `dag` ≈ `random` (only 2 actionable features, little ordering freedom). Validity should stay ~1.0.
- **HELOC `dag`/`full`**: neutral-to-worse vs `random`/`full` (parent set is a subset of the current conditioning set). If it *improves* coherence/sparsity without losing validity, note it as a mild positive.
- **HELOC `reduced` (either ordering)**: the cell most likely to show denser conditioning → lower `frac_oob` and more plausible LOF; this is the real test of whether the approach can be salvaged on high-dim data. If `dag`+`reduced` is the best HELOC cell, recommend it as the configuration for any future work.

---

## Commit

`feat(zeroshot-cf): add feature-ordering DAG ablation (Exp3) + reduced actionable-set`
