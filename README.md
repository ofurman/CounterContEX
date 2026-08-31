# CounterContEX TabICL Suite

This repository tracks one supported surface: the retained TabICL
counterfactual generator, the Exp9 DiCoFlex benchmark, four comparison
baselines, pinned CEL bootstrap, checkpoint staging, and Athena launchers under
`experiments/zeroshot_cf/`.

The repository root is the authoritative entry point. Root `uv` commands
install the same locked suite described in
[`experiments/zeroshot_cf/README.md`](experiments/zeroshot_cf/README.md).

## Setup

Create the root workspace environment:

```bash
uv sync --locked
```

Bootstrap the pinned CEL checkout and validate the four benchmark assets:

```bash
uv run python experiments/zeroshot_cf/vendor_setup.py
uv run python experiments/zeroshot_cf/vendor_setup.py --check
```

The bootstrap pins CEL revision
`3587f943826f6b087a0d198c8c4aa4373712c7ee` into the ignored local checkout at
`experiments/zeroshot_cf/vendor/counterfactuals/`.

## Verification Commands

Run the retained tests from the repository root:

```bash
uv run pytest -q
```

The documented offline CLI sanity checks are:

```bash
HF_HUB_OFFLINE=1 uv run python -m experiments.zeroshot_cf.exp8_tabicl_cf --help
HF_HUB_OFFLINE=1 uv run python -m experiments.zeroshot_cf.exp9_dicoflex_benchmark --help
HF_HUB_OFFLINE=1 uv run python -m experiments.zeroshot_cf.exp11_nice_nun_baseline --help
HF_HUB_OFFLINE=1 uv run python -m experiments.zeroshot_cf.exp12_optimization_baselines --help
HF_HUB_OFFLINE=1 uv run python -m experiments.zeroshot_cf.exp13_dice_baseline --help
HF_HUB_OFFLINE=1 uv run python -m experiments.zeroshot_cf.exp14_face_baseline --help
HF_HUB_OFFLINE=1 uv run python -m experiments.zeroshot_cf.tabicl_checkpoints --help
HF_HUB_OFFLINE=1 uv run python -m experiments.zeroshot_cf.tabicl_smoke_test --help
```

Stage checkpoints once on a networked machine:

```bash
uv run python -m experiments.zeroshot_cf.tabicl_checkpoints
```

Then run the real offline smoke:

```bash
HF_HUB_OFFLINE=1 uv run python -m experiments.zeroshot_cf.tabicl_smoke_test
```

## Retained Entry Points

- `python -m experiments.zeroshot_cf.exp8_tabicl_cf`
- `python -m experiments.zeroshot_cf.exp9_dicoflex_benchmark`
- `python -m experiments.zeroshot_cf.exp11_nice_nun_baseline`
- `python -m experiments.zeroshot_cf.exp12_optimization_baselines`
- `python -m experiments.zeroshot_cf.exp13_dice_baseline`
- `python -m experiments.zeroshot_cf.exp14_face_baseline`

The stable programmatic API is
`experiments.zeroshot_cf.generator.generate_counterfactual_batch()`.

The numbered commands are compatibility shims. New benchmark automation uses
the typed matrix CLI:

```bash
uv run python -m experiments.zeroshot_cf.cli matrix \
  --config experiments/zeroshot_cf/configs/matrices/one_factual_compat.yaml \
  --resume
uv run python -m experiments.zeroshot_cf.cli aggregate \
  --config experiments/zeroshot_cf/configs/matrices/one_factual_compat.yaml
```

Canonical runs are content-addressed directories completed by a final
`COMPLETE` marker. Scientific configuration and resolved data, model, method,
backend, and checkpoint versions determine identity; output paths, devices,
hosts, scheduler limits, and resume settings are execution metadata. See the
[suite README](experiments/zeroshot_cf/README.md) for layer ownership, exact
metric denominators, artifact layout, extension recipes, and the full-reference
procedure.

## Repository Layout

- `experiments/zeroshot_cf/`: retained runtime, tests, docs, Athena launchers, and suite-local lockfile
- `data/`: local dataset files materialized by the pinned CEL loader
- `logs/`: local run artifacts
- `docs/plans/tabicl-generator-cleanup/`: execution record for the cleanup that produced this surface

## Operational Notes

- The benchmark contract is fixed to `heloc`, `bank_marketing`,
  `give_me_some_credit`, and `lending_club` with the deterministic 64/16/20
  split and seed 42.
- Checkpoints live under `experiments/zeroshot_cf/models/tabicl/` and are
  intentionally ignored by git.
- Athena launchers and the suite-local README remain under
  `experiments/zeroshot_cf/` for cluster execution.
- Athena submission defaults to a configurable ten-hour walltime because the
  measured DiCoFlex/Lending Club reference cell took 7.64 hours.
