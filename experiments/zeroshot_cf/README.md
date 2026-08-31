# TabICL counterfactual benchmark suite

This directory is the retained CounterContEX surface: the TabICL generator,
the Exp9 DiCoFlex benchmark, four comparison baselines, pinned CEL dataset
bootstrap, checkpoint staging, offline smoke checks, and Athena launchers.
Legacy TabPFN counterfactual experiments are intentionally out of scope.

## Setup

Create the suite-owned environment from a fresh checkout:

```bash
uv sync --project experiments/zeroshot_cf --python 3.12 --locked
```

Bootstrap the pinned CEL checkout and validate the four benchmark assets:

```bash
uv run --project experiments/zeroshot_cf \
  python experiments/zeroshot_cf/vendor_setup.py

uv run --project experiments/zeroshot_cf \
  python experiments/zeroshot_cf/vendor_setup.py --check
```

`vendor_setup.py` uses the pinned CEL revision
`3587f943826f6b087a0d198c8c4aa4373712c7ee` under
`experiments/zeroshot_cf/vendor/counterfactuals/`. It verifies four required
config/data pairs:

- `config/datasets/heloc.yaml` -> `data/heloc.csv`
- `config/datasets/bank_marketing.yaml` -> `data/bank_marketing.csv`
- `config/datasets/give_me_some_credit.yaml` -> `data/give_me_some_credit.csv`
- `config/datasets/lending_club.yaml` -> `data/lending_club.csv`

The generated vendor checkout is local-only and ignored by git. If an existing
checkout is at the wrong revision, setup stops with a recovery instruction;
rerun with `--repin` only when you intentionally want to rewrite that local
tree.

The CEL checkout can materialize dataset files under the suite vendor tree and
the repo-root `data/` area, depending on the upstream config. Review the
upstream dataset licenses and provenance before redistributing any copied raw
data.

## Checkpoints and offline commands

Stage the TabICLv2 checkpoint pair once on a machine with network access:

```bash
uv run --project experiments/zeroshot_cf \
  python -m experiments.zeroshot_cf.tabicl_checkpoints
```

The classifier and regressor checkpoints are stored under
`experiments/zeroshot_cf/models/tabicl/`. Runtime checksum verification stays
enabled and the offline smoke test refuses missing or mismatched files.

The following `--help` commands are the documentation sanity checks for the
locked environment. With `HF_HUB_OFFLINE=1`, they must stay free of network
access and checkpoint loading:

```bash
HF_HUB_OFFLINE=1 uv run --project experiments/zeroshot_cf python -m experiments.zeroshot_cf.exp8_tabicl_cf --help
HF_HUB_OFFLINE=1 uv run --project experiments/zeroshot_cf python -m experiments.zeroshot_cf.exp9_dicoflex_benchmark --help
HF_HUB_OFFLINE=1 uv run --project experiments/zeroshot_cf python -m experiments.zeroshot_cf.exp11_nice_nun_baseline --help
HF_HUB_OFFLINE=1 uv run --project experiments/zeroshot_cf python -m experiments.zeroshot_cf.exp12_optimization_baselines --help
HF_HUB_OFFLINE=1 uv run --project experiments/zeroshot_cf python -m experiments.zeroshot_cf.exp13_dice_baseline --help
HF_HUB_OFFLINE=1 uv run --project experiments/zeroshot_cf python -m experiments.zeroshot_cf.exp14_face_baseline --help
HF_HUB_OFFLINE=1 uv run --project experiments/zeroshot_cf python -m experiments.zeroshot_cf.tabicl_checkpoints --help
HF_HUB_OFFLINE=1 uv run --project experiments/zeroshot_cf python -m experiments.zeroshot_cf.tabicl_smoke_test --help
```

Run the real offline smoke after the checkpoints are staged:

```bash
HF_HUB_OFFLINE=1 uv run --project experiments/zeroshot_cf \
  python -m experiments.zeroshot_cf.tabicl_smoke_test
```

## Public generator API

The stable programmatic entry point is
`experiments.zeroshot_cf.generator.generate_counterfactual_batch()`. It accepts
`TabICLGeneratorInputs`, a `TabICLGeneratorConfig`, and an injected point
backend. `exp8_tabicl_cf.py` is the CLI compatibility adapter over that API.

Single-counterfactual mode keeps the original greedy validity search plus
optional plausibility refinement. Multi-counterfactual mode uses bounded-beam
search and DPP-based selection without padding fabricated rows when fewer valid
diverse outputs exist.

## Benchmark and baselines

Exp9 is the fixed four-dataset benchmark:

- `heloc`
- `bank_marketing`
- `give_me_some_credit`
- `lending_club`

The protocol uses one deterministic 64/16/20 train/validation/test split,
seed 42, classifier-prediction targets, atomic categorical edits, immutable
feature preservation, and a shared result schema across TabICL and the
baselines.

Current actionability is intentionally narrow: immutable columns may not change,
and one-hot categorical groups may change only atomically. The suite does not
encode directional, monotonic, or causal constraints.

Available retained CLI entry points:

- `python -m experiments.zeroshot_cf.exp8_tabicl_cf`
- `python -m experiments.zeroshot_cf.exp9_dicoflex_benchmark`
- `python -m experiments.zeroshot_cf.exp11_nice_nun_baseline`
- `python -m experiments.zeroshot_cf.exp12_optimization_baselines`
- `python -m experiments.zeroshot_cf.exp13_dice_baseline`
- `python -m experiments.zeroshot_cf.exp14_face_baseline`

Example offline runs:

```bash
HF_HUB_OFFLINE=1 uv run --project experiments/zeroshot_cf \
  python -m experiments.zeroshot_cf.exp8_tabicl_cf \
  --dataset heloc \
  --max-test 10 \
  --cache-dir experiments/zeroshot_cf/models/tabicl

HF_HUB_OFFLINE=1 uv run --project experiments/zeroshot_cf \
  python -m experiments.zeroshot_cf.exp9_dicoflex_benchmark \
  --dataset heloc \
  --tabicl-cache-dir experiments/zeroshot_cf/models/tabicl

uv run --project experiments/zeroshot_cf \
  pytest -q
```

## Outputs and environment variables

Local benchmark and baseline runs write under `experiments/zeroshot_cf/results/local/`.
Athena runs write under `experiments/zeroshot_cf/results/athena/`. Both trees
are ignored by git. Discriminator caches live under `experiments/zeroshot_cf/models/`.

Relevant environment variables:

- `TABICL_LOCAL_CACHE`: default checkpoint directory.
- `TABICL_DEVICE`: `auto`, `cpu`, `mps`, or `cuda`.
- `HF_HUB_OFFLINE=1`: enforce offline operation after staging.
- `ZEROSHOT_CF_MODELS_DIR`: override the discriminator cache/output model directory.

## Athena

See [athena/README.md](athena/README.md) for the four-case Exp9 Slurm launcher,
environment variables, and the aggregation command.
