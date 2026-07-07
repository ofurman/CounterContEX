# Commands

All runs are **fully offline** against the staged local TabPFN **v2** checkpoints. Models load only
via `from checkpoints import get_models`. Never import `tabpfn_client` or call the cloud API. Run from
the repo root. On the DGX/GB10 host used for the heavy Phase-B stages this is
`/home/ofurman/pwr/CounterContEX`.

> **Artefact path**: all `results/...` artefacts land in `experiments/zeroshot_cf/results/` (the
> runners use exp2's `RESULTS_DIR = Path(__file__).parent / "results"`), **not** a repo-root `results/`.
> The report to extend is `experiments/zeroshot_cf/results/REPORT.md`.
> Budget/context grids: reuse the predecessor plan's `docs/plans/iterative-greedy-cf/resources/grids.md`.

## Environment (offline guarantee)

```bash
export HF_HUB_OFFLINE=1
export TABPFN_MODEL_VERSION=v2
export TABPFN_LOCAL_CACHE=/home/ofurman/pwr/CounterContEX/experiments/zeroshot_cf/models
export TABPFN_DEVICE=cuda   # Phase-B (DGX) only; Phase-A code/tests run on CPU
```

## Host preflight (required before Phase B, Stages 4–7)

Run once on the DGX before any Phase-B stage. Provisioning may need network once; the experiment runs
after this remain offline. See memory `dgx-remote-experiments` for the full one-time provisioning recipe
(uv venv py3.13, cu128 torch, editable tabpfn, tabpfn-extensions, cel vendor, scp'd v2 checkpoints).

```bash
cd /home/ofurman/pwr/CounterContEX
uv pip install --no-config -r experiments/zeroshot_cf/requirements.txt
uv run python experiments/zeroshot_cf/vendor_setup.py   # restore gitignored CEL tree

export HF_HUB_OFFLINE=1 TABPFN_MODEL_VERSION=v2 TABPFN_DEVICE=cuda
export TABPFN_LOCAL_CACHE=/home/ofurman/pwr/CounterContEX/experiments/zeroshot_cf/models
test -s "$TABPFN_LOCAL_CACHE/tabpfn-v2-classifier-finetuned-zk73skhh.ckpt"
test -s "$TABPFN_LOCAL_CACHE/tabpfn-v2-regressor.ckpt"

uv run python - <<'PY'
import torch
from experiments.zeroshot_cf.data import load_dataset
from experiments.zeroshot_cf.sampler import ConditionalDensitySampler
assert torch.cuda.is_available(), "CUDA requested but unavailable"
for d in ["moons", "heloc"]:
    b = load_dataset(d); print(d, b.X_train.shape, b.X_test.shape)
PY

uv run pytest experiments/zeroshot_cf/tests/test_context.py -q
```

If any check fails, fix provisioning first — do **not** defer missing deps/vendor/checkpoints/CUDA to
the Backlog. `--no-config` on every `uv pip` call avoids the repo `pyproject.toml [tool.uv] exclude-newer`
parse error. Sync code without disturbing gitignored `vendor/`+`models/`: `git fetch && git reset --hard
origin/greedy-cf-fullscale`.

## Tests (Phase A + Stage 8)

```bash
uv run pytest experiments/zeroshot_cf/tests -q                    # FULL suite (Stage 8 requirement)
uv run pytest experiments/zeroshot_cf/tests/test_greedy.py -q     # Stages 1, 3
uv run pytest experiments/zeroshot_cf/tests/test_context.py -q    # Stage 2
```

## Detached DGX run pattern (survives the SSH session)

```bash
nohup bash -c 'set -e; <env exports>; uv run python experiments/zeroshot_cf/<exp>.py <args>; \
  touch ~/<exp>.DONE' > ~/<exp>.log 2>&1 &
# then poll ~/<exp>.DONE over SSH. Per-impute ~0.5 s warm (first point ~60 s CUDA kernel compile).
```

## Phase-B full-scale runs (HELOC n=200, MOONS full)

```bash
# Stage 4 — selector ablation (Exp5): identical --max-test across both selectors within a dataset
uv run python experiments/zeroshot_cf/exp5_selector_ablation.py --dataset moons
uv run python experiments/zeroshot_cf/exp5_selector_ablation.py --dataset heloc --max-test 200

# Stage 5 — context ablation (Exp6): 16-cell grid at the Stage-4 selector (the long pole; add --beam if needed)
uv run python experiments/zeroshot_cf/exp6_context_ablation.py --dataset moons
uv run python experiments/zeroshot_cf/exp6_context_ablation.py --dataset heloc --max-test 200

# Stage 6 — budget sweep (Exp7): ONLY after Stage 1's shortcut fix; regenerates both CSVs
uv run python experiments/zeroshot_cf/exp7_budget_sweep.py --dataset moons
uv run python experiments/zeroshot_cf/exp7_budget_sweep.py --dataset heloc --max-test 200

# Stage 7 — routing override (Exp9): needs Stage 3 --beam to be tractable at scale
uv run python experiments/zeroshot_cf/exp9_routing_audit.py --dataset heloc --max-test 200 --budget 17 --beam 6
# (baseline --force-numeric-cols none vs override 5,6,9,10,12 handled inside the driver)
```

Keep `--max-test`, temperature, `--n-permutations`, and `--beam` **identical across compared cells**
within a dataset. Log the effective n and beam for every run. Sizes above the pool are capped and the
`effective_size` recorded.

## Guardrail checks (run before each commit)

```bash
git diff --name-only <base>..HEAD -- src/tabpfn    # must be empty (core untouched); <base> = branch point from iterative-greedy-cf
grep -rn "tabpfn_client" experiments/zeroshot_cf   # must find nothing
```
