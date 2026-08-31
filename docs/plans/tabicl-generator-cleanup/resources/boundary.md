# Retained, Extract-First, and Removed Boundary

This is the planning-time ownership map. Execution records any justified changes in
`decisions.md` and `journal.md`; it must not silently broaden deletion scope.

## Keep: generator and diversity

- `exp8_tabicl_cf.py`: compatibility CLI and adapter around the retained generator.
- `tabicl_sampler.py`: conditional point/quantile prediction, context selection, confidence
  augmentation, batching, and estimator reuse.
- `grouped_categorical.py`: compact mixed representation and greedy numerical/categorical search,
  revisits, global action ranking, and plausibility-refinement shortlist.
- `tabicl_joint_plausibility.py`: joint conditional scorer used for one-shot refinement.
- `mixed_distance.py`: grouped Gower distances and action-unit counts.
- `diverse_search.py`: bounded beam/archive search, grouped action signatures, quality filtering,
  and exact fixed-size DPP selection.
- `tabicl_checkpoints.py` and `tabicl_smoke_test.py`: checksum-verified staging and offline probe.

## Keep: benchmark, baselines, and operations

- `exp9_dicoflex_benchmark.py`: four-dataset benchmark and aggregate CLI.
- `exp11_nice_nun_baseline.py`: NICE nearest-unlike-neighbour baseline.
- `exp12_optimization_baselines.py`: Wachter coordinate search and Growing Spheres.
- `exp13_dice_baseline.py`: DiCE genetic adapter.
- `exp14_face_baseline.py`: density-weighted FACE-kNN.
- `data.py`, `discriminator.py`, and the DiCoFlex portion of `metrics_harness.py`.
- `configs/heloc_actionability.yaml`.
- `athena/README.md`, `exp9_dicoflex_array.sbatch`, `exp9_dicoflex_cases.tsv`, and
  `submit_exp9_dicoflex.sh`.
- Tests for data cleaning, diversity, Exp9, Exp11–14, grouped categorical search, common metrics,
  mixed distance, TabICL backend, and plausibility.

## Benchmark contract to freeze

- Datasets: HELOC, Bank Marketing, Give Me Some Credit, and Lending Club; Adult remains excluded.
- Seed-42 stratified 64/16/20 train/validation/test split; scaling fits the final training set only.
- Up to 1,000 seed-42 stratified factuals; target is `1 - classifier.predict(x)`, not ground truth.
- Logistic-regression discriminator: `C=1`, `max_iter=1000`, `random_state=42`.
- HELOC removes all-`-9` no-bureau rows before splitting and preserves six immutable fields.
- Categories change as atomic one-hot groups; immutable features equal their factual values.
- Common metrics cover coverage, classifier validity, actionability, transformed and action-unit
  sparsity, valid-only grouped/continuous proximity, LOF, and Isolation Forest.
- Exp9 primary metrics use one primary CF per factual; diversity metrics evaluate the complete set.
- Single-CF `data_plausible` refinement and multi-CF sparse diversity are separate supported modes.

## Extract before deleting legacy modules

1. Move `infer_feature_domains()` and `project_candidate_values()` from `greedy.py` into a neutral
   candidate-domain module; update Exp8, grouped search, diversity, and migrated tests.
2. Move `TAU`, `_DATASET_PARAMS`, and retained reporting behavior out of `exp4_greedy_cf.py`.
3. Move `OneHotActionGroup` into a dependency-light action-space module while allowing `data.py`
   to construct it; importing distances/search must not import CEL.
4. Move dataset names, validation/test defaults, deterministic factual selection, classifier
   protocol, and result schemas into a benchmark protocol module.
5. Move `ActionUnit`, action-unit construction, pruning, and scalar contraction into baseline-common
   modules; Exp12–14 must not import NICE or each other for utilities.
6. Split `metrics_harness.py`: retain common DiCoFlex metrics and remove Exp2/4-only reporting only
   after Exp8 uses the retained reporting path.
7. Migrate useful projection and quantile tests out of legacy `test_greedy.py`; remove its TabPFN
   sampler/class-divergence cases.

## Remove after extraction

- Runners: `exp1_single_feature.py`, `exp2_counterfactuals.py`, `exp3_feature_ordering.py`,
  `exp4_greedy_cf.py`, `exp5_selector_ablation.py`, `exp6_context_ablation.py`, and `refine.py`.
- Legacy runtime: `checkpoints.py`, `sampler.py`, `smoke_test.py`, and `greedy.py` after all retained
  helpers and tests move.
- Assembly: `build_notebook.py`, `results.ipynb`, and `configs/sweep.yaml`.
- Committed legacy outputs: `results/REPORT.md` and all tracked `results/exp1_*` through
  `results/exp6_*` files.
- Legacy tests: `test_context.py`, `test_context_ablation.py`, `test_ordering.py`, `test_sampler.py`,
  `test_selector_ablation.py`, legacy-only parts of `test_greedy.py` and
  `test_metrics_harness.py`, plus the old real-TabPFN fixture in `tests/conftest.py`.
- Root/editable TabPFN and `tabpfn-extensions` entries from the suite dependency set after retained
  imports are clean.

## Remove after the retained suite is independent

- `src/tabpfn/` (116 tracked upstream source files), root `tests/` (121 files), `examples/`
  (13 files), `changelog/`, `CHANGELOG.md`, and TabPFN-only `scripts/`.
- The ignored top-level `tabpfn/` directory, which currently contains only bytecode/cache artifacts;
  it is explicitly in scope despite the general local-state preservation rule.
- TabPFN-only GitHub workflows, issue templates, release automation, Dependabot entries, and
  upstream-specific `.gemini/config.yaml` settings.
- Completed predecessor plans `docs/plans/zeroshot-tabpfn-cf/` and
  `docs/plans/iterative-greedy-cf/` after their durable lessons are retained.

## Replace at the repository root

- Convert root `pyproject.toml` and `uv.lock` from the TabPFN distribution into a thin authoritative
  workspace/project entry for the retained suite; root `uv sync --locked` and `uv run pytest` must
  exercise it.
- Rewrite root `README.md`, `SECURITY.md`, `.pre-commit-config.yaml`, `.gitignore`, CODEOWNERS,
  contribution templates, and CI around the TabICL generator/benchmark workflows.
- Update `THIRD-PARTY-NOTICES.md` for retained dependencies and copyright. Do not remove root
  `LICENSE`; retained files carry Prior Labs copyright and licensing changes require legal review.

## Preserve after root cleanup

- Root `LICENSE` and the updated `THIRD-PARTY-NOTICES.md`.
- Pre-existing untracked `experiments/zeroshot_cf/ARCHITECTURE.md`.
- Ignored local checkpoints, raw data, vendor checkout/symlink, discriminator models, results,
  experiment caches, and logs outside the explicitly authorized top-level `tabpfn/` cache.

## Data and dependency hazards

- `data.py` imports CEL and hardcodes `vendor/counterfactuals` for four configs and CSVs. An
  installed CEL package alone is insufficient.
- The local vendor revision is `3587f943826f6b087a0d198c8c4aa4373712c7ee`; both the existing
  shallow clone script and requirements are currently unpinned.
- The raw benchmark files total roughly 20 MB and cannot be committed or redistributed without a
  license/provenance decision.
- `tabicl_sampler.py` imports TabICL private APIs and mutates fitted attributes, so exact
  `tabicl==2.1.1` is a correctness dependency.
- DiCE and `raiutils` import eagerly; include them in the benchmark environment or make the DiCE
  entry point explicitly optional without breaking normal test collection.
- `TABICL_LOCAL_CACHE` and `TABICL_DEVICE` are captured at import; checkpoint staging performs
  network access, but normal runtime and `--help` must not.
- Aggregation and baseline `--help` currently transitively load the TabICL/legacy stack through
  shared constants; Stage 4 must make these lightweight.
- CEL remains the benchmark data/preprocessing dependency; this plan pins and validates it but does
  not replace it.
- Working-tree deletion does not remove upstream objects from Git history. The plan uses ordinary
  commits on `cleanup/tabicl-suite`; the user will squash them after removal, without initializing a
  new repository or filtering history.

## Historical anchors

- `89f5416`: initial TabICL comparison.
- `cbe1cb7`: quantile candidates.
- `38ba847`: confidence conditioning.
- `6f3e07e`: revisits.
- `522721d` and `87b0ea3`: plausibility refinement and one-shot joint reranking.
- `2e45074`: Exp9 benchmark.
- `092d8d3`, `5801c65`, and `21fb556`: baseline additions.
- `41ef597`: Gower context selection.
- `e80925d`: bounded-beam/DPP diversity implementation at the planning baseline.
