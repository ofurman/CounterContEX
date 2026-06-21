# Commands

All runs are **fully offline** against the staged local TabPFN **v2** checkpoints. Models
load only via `from checkpoints import get_models`. Never import `tabpfn_client` or call the
cloud API. Run from the repo root (`/Users/ofurman/pwr/CounterContEX`).

> **Artefact path**: all `results/...` artefacts land in `experiments/zeroshot_cf/results/`
> (the runners use exp2's `RESULTS_DIR = Path(__file__).parent / "results"`), **not** a
> repo-root `results/`. The report to extend is `experiments/zeroshot_cf/results/REPORT.md`.

## Environment (offline guarantee)

```bash
export HF_HUB_OFFLINE=1
export TABPFN_MODEL_VERSION=v2
# TABPFN_LOCAL_CACHE / TABPFN_DEVICE may be set per checkpoints.py defaults.
```

## Tests

```bash
uv run pytest experiments/zeroshot_cf/tests -q                       # full suite (prior + new)
uv run pytest experiments/zeroshot_cf/tests/test_greedy.py -q        # Stage 1
uv run pytest experiments/zeroshot_cf/tests/test_context.py -q       # Stage 3
```

## Stage 1 — baseline greedy generation (Exp4)

```bash
# MOONS smoke test first (≤2 steps), then HELOC bounded.
uv run python experiments/zeroshot_cf/exp4_greedy_cf.py --dataset moons --selector prob_ascent
uv run python experiments/zeroshot_cf/exp4_greedy_cf.py --dataset heloc --selector prob_ascent --max-test 50
# class-divergence selector requires all_classes context (wired automatically):
uv run python experiments/zeroshot_cf/exp4_greedy_cf.py --dataset moons --selector class_divergence
```

## Stage 2 — selector ablation (Exp5)

```bash
uv run python experiments/zeroshot_cf/exp5_selector_ablation.py --dataset moons
uv run python experiments/zeroshot_cf/exp5_selector_ablation.py --dataset heloc --max-test 50
# Keep --max-test identical across the two selectors within a dataset.
```

## Stage 4 — context ablation (Exp6)

```bash
# Uses the Stage-2 winning selector (default prob_ascent → full 16-cell grid;
# class_divergence → 8 cells, *_target skipped). Keep --max-test identical across cells.
uv run python experiments/zeroshot_cf/exp6_context_ablation.py --dataset moons
uv run python experiments/zeroshot_cf/exp6_context_ablation.py --dataset heloc --max-test 30
```

## Guardrail checks (run before each commit)

```bash
git diff --name-only main..HEAD -- src/tabpfn          # must be empty (core untouched)
grep -rn "tabpfn_client" experiments/zeroshot_cf       # must find nothing
```
