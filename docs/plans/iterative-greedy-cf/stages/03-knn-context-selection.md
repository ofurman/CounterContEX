# Stage 3: kNN / Context-Selection Support in the Sampler

**Goal**: Extend `ConditionalDensitySampler.set_context` so the context pool can be built by **random subsample** (current behaviour) or **nearest-neighbour** to a query point, drawn from either the **target class** or **both classes** — enabling the four context strategies the Stage-4 ablation needs. Default behaviour is unchanged.
**Dependencies**: Stage 1 DONE. Independent of Stage 2. Required by Stage 4.

---

## What exists today

`set_context` (`experiments/zeroshot_cf/sampler.py:93–157`) does exactly one thing: optionally filter `X_context` to `target_class` (`:127–132`), then **random** subsample to `max_context` (`:134–140`) via `rng.choice`. There is no relevance-based (nearest-neighbour) selection, and the class pool is binary (one class via `target_class`, or all rows if `target_class=None`).

The four context strategies the ablation requires map onto two new orthogonal choices:

| Strategy name | Class pool | Selection within pool |
|---------------|-----------|------------------------|
| `random_target` | target class only | random subsample (**= current behaviour** with `target_class` set) |
| `random_both`   | both classes | random subsample (**= current behaviour** with `target_class=None`) |
| `knn_target`    | target class only | `max_context` nearest neighbours to the query |
| `knn_both`      | both classes | `max_context` nearest neighbours to the query |

The two `random_*` strategies are already implemented — Stage 3 adds only the two `knn_*` strategies.

---

## Steps

1. **Add selection-method parameters to `set_context`.**
   - File: `experiments/zeroshot_cf/sampler.py`, `set_context` signature (≈ line 93).
   - Add `selection: str = "random"` (`"random"` | `"knn"`) and `query: Optional[np.ndarray] = None` (the factual point used as the kNN anchor; shape `(d,)` or `(1, d)`). Keep `target_class` and `max_context` as-is.
   - When `selection == "random"`: unchanged path (filter to `target_class` if set, then `rng.choice` subsample). Preserves byte-identical behaviour for existing callers (the default).
   - When `selection == "knn"`: require `query` (raise `ValueError` if `None`). After the optional `target_class` filter, if `len(X) > max_context`, select the `max_context` rows with the **smallest Euclidean distance** to `query` in the (already MinMax-[0,1]) feature space; sort the chosen indices for determinism. Compute distance over the original `d` features only (the appended Y column is added *after* selection). Slice `y` in lockstep.
   - Document in the docstring that `knn_target` ≡ (`target_class=<t>`, `selection="knn"`) and `knn_both` ≡ (`target_class=None`, `selection="knn"`).

2. **Keep the RNG / append-Y / fit tail unchanged.**
   - The categorical-Y append (`:144–151`), `model.fit` (`:155`), and RNG seeding (`:116–121`) are untouched — selection only changes *which rows* enter `X`/`y` before the append.

3. **Expose a tiny helper for distance (optional, for clarity).**
   - A module-level `_knn_indices(X, query, k) -> np.ndarray` keeps `set_context` readable and is unit-testable in isolation.

4. **Tests.**
   - New file: `experiments/zeroshot_cf/tests/test_context.py`. Unit-test the selection logic in **isolation** via the module-level `_knn_indices` / random-subsample helpers (pure numpy, **no model needed**); use the conftest `models` fixture only for any test that exercises the full `set_context → model.fit` path. There is no `FAST_TEST_MODE`.
   - (a) `selection="random"` with a fixed seed reproduces the **exact** indices the current implementation produces (regression guard — default behaviour unchanged; test the selection logic directly so it doesn't depend on a fitted model).
   - (b) `selection="knn"` returns the `k` rows closest to `query` on a small synthetic set with a hand-checkable nearest set; indices are sorted; `y` is sliced consistently.
   - (c) `knn` raises `ValueError` when `query` is `None`.
   - (d) `knn_target` draws only target-class rows; `knn_both` may draw either class (assert label composition on a constructed set).
   - (e) when `len(pool) <= max_context`, both methods return the whole pool (no selection needed).
   - All prior tests (13) still pass.

---

## Verification

- [ ] `uv run pytest experiments/zeroshot_cf/tests/test_context.py -q` passes.
- [ ] Regression test (a) confirms `selection="random"` indices are byte-identical to the pre-change behaviour for the same seed.
- [ ] `uv run pytest experiments/zeroshot_cf/tests -q` — full suite green (prior + new).
- [ ] `git diff --name-only main..HEAD -- src/tabpfn` is empty; no `tabpfn_client` import.

---

## Commit

`feat(greedy-cf): add kNN context selection (random|knn × target|both) to set_context`
