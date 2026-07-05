# Commands

All runs are **fully offline** against the staged local TabPFN **v2** checkpoints. Models
load only via `from checkpoints import get_models`. Never import `tabpfn_client` or call the
cloud API. Run from the repo root. On the DGX/GB10 host used for the heavy stages this is
`/home/ofurman/pwr/CounterContEX`.

> **Artefact path**: all `results/...` artefacts land in `experiments/zeroshot_cf/results/`
> (the runners use exp2's `RESULTS_DIR = Path(__file__).parent / "results"`), **not** a
> repo-root `results/`. The report to extend is `experiments/zeroshot_cf/results/REPORT.md`.

## Host preflight (required before Stage 5+)

Run this once on the execution host before starting any pending stage. Dependency/vendor
provisioning may require network access once; the experiment runs after this preflight remain
offline. Do not proceed until all checks pass.

```bash
cd /home/ofurman/pwr/CounterContEX

# Install the experiment-only dependencies that are not part of the TabPFN package itself.
uv pip install -r experiments/zeroshot_cf/requirements.txt

# Restore the gitignored CEL dataset/config tree used by load_dataset().
uv run python experiments/zeroshot_cf/vendor_setup.py

export HF_HUB_OFFLINE=1
export TABPFN_MODEL_VERSION=v2
export TABPFN_LOCAL_CACHE=/home/ofurman/pwr/CounterContEX/experiments/zeroshot_cf/models
export TABPFN_DEVICE=cuda

# These exact files must already be present for offline execution.
test -s "$TABPFN_LOCAL_CACHE/tabpfn-v2-classifier-finetuned-zk73skhh.ckpt"
test -s "$TABPFN_LOCAL_CACHE/tabpfn-v2-regressor.ckpt"

# Import and data smoke tests. These catch missing tabpfn-extensions, cel, vendor data,
# checkpoints, and CUDA visibility before a multi-hour experiment starts.
uv run python - <<'PY'
from pathlib import Path
import torch
from experiments.zeroshot_cf.checkpoints import TABPFN_LOCAL_CACHE
from experiments.zeroshot_cf.data import load_dataset
from experiments.zeroshot_cf.sampler import ConditionalDensitySampler

print("cuda_available", torch.cuda.is_available())
assert torch.cuda.is_available(), "TABPFN_DEVICE=cuda requested but CUDA is unavailable"
print("sampler", ConditionalDensitySampler.__name__)
for name in [
    "tabpfn-v2-classifier-finetuned-zk73skhh.ckpt",
    "tabpfn-v2-regressor.ckpt",
]:
    path = Path(TABPFN_LOCAL_CACHE) / name
    assert path.is_file() and path.stat().st_size > 0, path
for dataset in ["moons", "heloc"]:
    bundle = load_dataset(dataset)
    print(dataset, bundle.X_train.shape, bundle.X_test.shape)
PY

uv run pytest experiments/zeroshot_cf/tests/test_context.py -q
```

If any command fails, fix provisioning first. Do **not** defer missing dependencies,
missing vendor data, missing v2 checkpoints, or CUDA unavailability to the stage backlog:
they block all experiment stages.

Spark note: the current plan does not call Spark. If the target host policy requires Spark,
verify `spark-submit --version` separately; otherwise Spark is not a prerequisite for these
Python/TabPFN runs.

## Environment (offline guarantee)

```bash
export HF_HUB_OFFLINE=1
export TABPFN_MODEL_VERSION=v2
export TABPFN_LOCAL_CACHE=/home/ofurman/pwr/CounterContEX/experiments/zeroshot_cf/models
export TABPFN_DEVICE=cuda
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

## Stage 5 — budget sweep (Exp7)

```bash
# Revisit-enabled loop; --stall-eps default 1e-6. Sweep budget at the Stage-4 config.
uv run python experiments/zeroshot_cf/exp7_budget_sweep.py --dataset moons
uv run python experiments/zeroshot_cf/exp7_budget_sweep.py --dataset heloc --max-test 30
# Spot-check the revisit loop directly:
uv run python experiments/zeroshot_cf/exp4_greedy_cf.py --dataset moons --selector prob_ascent --budget 32
```

## Stage 6 — MOONS trajectory plots (Exp8)

```bash
uv run python experiments/zeroshot_cf/exp8_moons_trajectories.py --max-test 30
# → results/figures/moons_trajectories.png (+ blocked-slice panels)
```

## Stage 7 — native-categorical dataset (Exp4 on the new dataset)

```bash
# <native_cat_name> is the adapted/synthetic dataset with one categorical variable per
# column (integer/category-coded), no one-hot expansion, and explicit TabPFN categorical indices.
uv run python experiments/zeroshot_cf/exp4_greedy_cf.py --dataset <native_cat_name> --selector prob_ascent --max-test 50
```

## Stage 8 — routing override (Exp9)

```bash
uv run python experiments/zeroshot_cf/exp9_routing_audit.py --dataset heloc --max-test 30
# baseline (--force-numeric-cols none) vs override, written to results/exp9_routing_summary.md
```

## Stage 9 — consolidated table

```bash
# Synthesis only — aggregates exp4/5/6/7/9 CSVs into results/summary_table.{md,csv}
uv run python experiments/zeroshot_cf/exp4_greedy_cf.py --help   # confirm proximity column present
```

## Guardrail checks (run before each commit)

```bash
git diff --name-only main..HEAD -- src/tabpfn          # must be empty (core untouched)
grep -rn "tabpfn_client" experiments/zeroshot_cf       # must find nothing
```
