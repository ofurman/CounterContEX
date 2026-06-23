# Commands

All runs are **fully offline** against the staged local TabPFN **v2** checkpoints. Models
load only via `from checkpoints import get_models`. Never import `tabpfn_client` or call the
cloud API. Run from the repo root (`/Users/ofurman/pwr/CounterContEX`).

> **Artefact path**: all `results/...` artefacts land in `experiments/zeroshot_cf/results/`
> (`RESULTS_DIR = Path(__file__).parent / "results"`), **not** a repo-root `results/`.
> The report to extend is `experiments/zeroshot_cf/results/REPORT.md`.

> **Env note** (`iterative-greedy-cf` Decision #10): if `uv run` fails to resolve deps in the
> offline sandbox, fall back to the sibling provisioned venv: `PYTHONPATH=<repo>` +
> `/Users/ofurman/pwr/TabPFN/.venv/bin/python`, `TABPFN_LOCAL_CACHE` → staged v2 checkpoints,
> and the gitignored `experiments/zeroshot_cf/vendor` symlink for `cel` configs. Heavy runs go
> on the remote DGX `gx10-bdc5` (Decision #8).

## Environment (offline guarantee)

```bash
export HF_HUB_OFFLINE=1
export TABPFN_MODEL_VERSION=v2
```

## Tests

```bash
uv run pytest experiments/zeroshot_cf/tests -q                    # full suite (prior + new)
uv run pytest experiments/zeroshot_cf/tests/test_score.py -q      # Stage 1
uv run pytest experiments/zeroshot_cf/tests/test_flow.py -q       # Stages 2, 4, 5
```

## Stage 1 — score-oracle validation

```bash
# the cosine-accuracy gate vs numerical KDE-score lives in the test
uv run pytest experiments/zeroshot_cf/tests/test_score.py -q -k cosine
```

## Stages 2–3 — flow core + headline test (Exp7)

```bash
# smoke (local CPU OK)
uv run python experiments/zeroshot_cf/exp7_flow_cf.py --dataset moons --mode flow --budget 2 --max-test 20
uv run python experiments/zeroshot_cf/exp7_flow_cf.py --dataset heloc --mode flow --context-strategy knn_both --max-context 256 --max-test 5

# paired baseline (same driver, --mode greedy) — keep --max-test identical to the flow cell
uv run python experiments/zeroshot_cf/exp7_flow_cf.py --dataset moons --mode greedy --max-test 20

# full headline (heavy → DGX): MOONS n=100, HELOC bounded + logged
uv run python experiments/zeroshot_cf/exp7_flow_cf.py --dataset moons --mode flow --budget 2
uv run python experiments/zeroshot_cf/exp7_flow_cf.py --dataset heloc --mode flow --context-strategy knn_both --max-context 256 --max-test 30
```

## Stage 4 — path-adaptive context ablation

```bash
# context_refit ∈ {0 static, 5 periodic, 1 every-step}; keep --max-test identical across cells
uv run python experiments/zeroshot_cf/exp7_flow_cf.py --dataset heloc --mode flow --context-strategy knn_both --context-refit 5 --max-test 30
```

## Stage 5 — annealing + mix ablation

```bash
# mix grid (α,β); n_samples>1 + noise>0 for the recourse distribution
uv run python experiments/zeroshot_cf/exp7_flow_cf.py --dataset moons --mode flow --alpha 1 --beta 0   # generative-only ≈ class_divergence
uv run python experiments/zeroshot_cf/exp7_flow_cf.py --dataset moons --mode flow --alpha 0 --beta 1   # discriminative-only ≈ prob_ascent
uv run python experiments/zeroshot_cf/exp7_flow_cf.py --dataset moons --mode flow --alpha 1 --beta 1   # dual
```

## Guardrail checks (run before each commit)

```bash
git diff --name-only iterative-greedy-cf..HEAD -- src/tabpfn   # must be empty (core untouched)
grep -rn "tabpfn_client" experiments/zeroshot_cf              # must find nothing
```
