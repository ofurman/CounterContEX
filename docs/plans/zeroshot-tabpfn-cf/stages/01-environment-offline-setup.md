# Stage 1: Environment & Offline Checkpoint Setup

**Goal**: Stand up a fully-offline environment — TabPFN v2 checkpoints pre-staged locally, `tabpfn-extensions` and `cel` installed — and prove local sampling from the conditional density works with no network.
**Dependencies**: None

---

## Steps

1. **Create the experiment package and branch.**
   - Branch: `git checkout -b zeroshot-tabpfn-cf` (from `main`).
   - Files: `experiments/zeroshot_cf/__init__.py`, `experiments/zeroshot_cf/README.md` (stub: purpose + offline run instructions), `experiments/zeroshot_cf/requirements.txt`.
   - Use `uv` for all Python tooling (team convention). Prefer a dedicated venv: `uv venv` in repo root.

2. **Install dependencies (offline-capable).**
   - File: `experiments/zeroshot_cf/requirements.txt`
   - Pin: the local TabPFN core (`-e .` from repo root, v8.0.8), `tabpfn-extensions[all] @ git+https://github.com/PriorLabs/tabpfn-extensions.git`, and `cel` from `git+https://github.com/ofurman/counterfactuals.git` (or a local clone path if network-restricted — see note).
   - `cel` pulls a heavy TF/alibi stack for baseline CF methods we don't use. We only need `cel.datasets`, `cel.preprocessing`, `cel.metrics`, and a classifier model (`cel.models` LR/MLP). If the full install fails, install a minimal subset and defer the heavy optional deps to the Backlog.
   - Note: if `cel` is not cleanly pip-installable, clone `ofurman/counterfactuals` into `experiments/zeroshot_cf/vendor/counterfactuals` and install editable / add to `sys.path`. Record the path taken in Decisions.

3. **Pre-stage TabPFN v2 checkpoints for offline use.**
   - File: `experiments/zeroshot_cf/checkpoints.py`
   - Set `TABPFN_MODEL_CACHE_DIR` to a repo-local dir, e.g. `experiments/zeroshot_cf/models/` (gitignored).
   - On a machine with network, trigger a one-time download by constructing `TabPFNClassifier()` and `TabPFNRegressor()` once (they auto-download `tabpfn-v2-classifier.ckpt` / `tabpfn-v2-regressor.ckpt` from HuggingFace into the cache dir). Thereafter all runs are offline.
   - Provide `get_models(device="auto", n_estimators=4) -> (TabPFNClassifier, TabPFNRegressor)` loading from local cache (or explicit `model_path=`).
   - Device: on this Mac `device="auto"` selects MPS. Expose a `TABPFN_DEVICE` override so CPU fallback is trivial. (See `resources/api-reference.md` for env vars: `TABPFN_MPS_MEMORY_FRACTION`.)

4. **Add `.gitignore` entries.**
   - File: repo `.gitignore` (or `experiments/zeroshot_cf/.gitignore`)
   - Ignore: `experiments/zeroshot_cf/models/`, `experiments/zeroshot_cf/results/data/`, `experiments/zeroshot_cf/vendor/`. Keep committed: code, configs, `results/*.md` reports, small `results/*.png` plots.

5. **Smoke test: offline conditional-density sampling.**
   - File: `experiments/zeroshot_cf/smoke_test.py`
   - Fit a `TabPFNRegressor` on a tiny toy `(X[:, :2], y=X[:, 2])`, call `predict(X_query, output_type="full")`, assert the dict has `"criterion"` and `"logits"`, then `criterion.sample(logits, t=1.0)` returns a finite tensor of the expected shape.
   - Run with networking disabled (`HF_HUB_OFFLINE=1`) to prove no download occurs once checkpoints are staged.

---

## Verification

- [ ] `uv run python -c "import tabpfn, tabpfn_extensions, cel; print('ok')"` prints `ok`.
- [ ] `HF_HUB_OFFLINE=1 uv run python experiments/zeroshot_cf/smoke_test.py` runs with no network error and prints a sampled value + its shape.
- [ ] Checkpoint files exist under `experiments/zeroshot_cf/models/` and show as ignored in `git status`.
- [ ] `get_models()` returns a `(clf, reg)` pair on `device="auto"` without error.

---

## Commit

`chore(zeroshot-cf): offline TabPFN env, checkpoint staging, sampling smoke test`
