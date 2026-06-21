"""Unit tests for build_chain_dag and impute_masked with dag support (Stage 9).

Tests:
  1. build_chain_dag correctness: Y/immutables absent from keys (roots); each
     actionable's parents = [y_idx] + immutables + earlier actionables.
  2. build_chain_dag acyclicity: topological order exists (no cycles); Y and
     immutables precede all actionables.
  3. impute_masked with dag= returns correct shape and leaves non-masked
     columns byte-identical; all masked cells are filled (no NaN remains).
  4. All prior 8 tests still pass (import guard only; tests live in other modules).
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.zeroshot_cf.sampler import build_chain_dag, ConditionalDensitySampler

# The shared ``models`` fixture now lives in tests/conftest.py.


def _make_synthetic(n: int = 80, seed: int = 0):
    """3-feature dataset: x2 = x0 + x1 + noise, binary label."""
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(0, 1, n)
    x1 = rng.uniform(0, 1, n)
    x2 = x0 + x1 + 0.05 * rng.standard_normal(n)
    X = np.stack([x0, x1, x2], axis=1).astype(np.float64)
    y = (x2 > 1.0).astype(np.int64)
    return X, y


# ---------------------------------------------------------------------------
# Test 1: build_chain_dag — structural correctness
# ---------------------------------------------------------------------------

def test_build_chain_dag_structure():
    """Y and immutables absent from keys; each actionable's parents are correct."""
    y_idx = 5
    immutable_idx = [1, 3]
    ordered_actionable = [0, 2, 4]

    dag = build_chain_dag(ordered_actionable, immutable_idx, y_idx)

    # Y and immutables are roots → not in keys
    assert y_idx not in dag, "Y column must be a root (not in dag keys)"
    for im in immutable_idx:
        assert im not in dag, f"Immutable {im} must be a root (not in dag keys)"

    # Verify parent sets
    assert dag[0] == [y_idx] + immutable_idx + []
    assert dag[2] == [y_idx] + immutable_idx + [0]
    assert dag[4] == [y_idx] + immutable_idx + [0, 2]


def test_build_chain_dag_empty_immutables():
    """Works with no immutables (e.g. MOONS)."""
    y_idx = 2
    immutable_idx = []
    ordered_actionable = [0, 1]

    dag = build_chain_dag(ordered_actionable, immutable_idx, y_idx)

    assert dag[0] == [y_idx]
    assert dag[1] == [y_idx, 0]
    assert y_idx not in dag


def test_build_chain_dag_acyclic_topo_order():
    """The DAG must be acyclic and topological order places Y+immutables before actionables."""
    y_idx = 10
    immutable_idx = [7, 8]
    ordered_actionable = [0, 2, 4, 6]

    dag = build_chain_dag(ordered_actionable, immutable_idx, y_idx)

    # Build a simple topological sort to verify no cycles exist
    all_nodes = set(dag.keys()) | {y_idx} | set(immutable_idx)
    in_degree = {n: 0 for n in all_nodes}
    adj: dict = {n: [] for n in all_nodes}
    for node, parents in dag.items():
        for p in parents:
            adj[p].append(node)
            in_degree[node] = in_degree.get(node, 0) + 1

    # Kahn's algorithm
    from collections import deque
    queue = deque([n for n in all_nodes if in_degree[n] == 0])
    topo = []
    while queue:
        node = queue.popleft()
        topo.append(node)
        for child in adj[node]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    assert len(topo) == len(all_nodes), "Cycle detected — topological sort incomplete"

    # Y and immutables must appear before all actionables in topo order
    actionable_set = set(ordered_actionable)
    root_set = {y_idx} | set(immutable_idx)
    topo_pos = {n: i for i, n in enumerate(topo)}
    for r in root_set:
        for a in actionable_set:
            assert topo_pos[r] < topo_pos[a], (
                f"Root {r} must precede actionable {a} in topo order"
            )


# ---------------------------------------------------------------------------
# Test 2: impute_masked with dag — shape, no NaN, non-masked preserved
# ---------------------------------------------------------------------------

def test_impute_masked_with_dag_shape_and_invariants(models):
    """impute_masked(dag=dag) returns same shape as dag=None; non-masked cols preserved."""
    clf, reg = models
    X, y = _make_synthetic(n=80, seed=3)
    X_train, X_test = X[:60], X[60:]
    y_train = y[:60]

    # Treat col 2 as actionable, col 1 as immutable, col 0 as Y (appended)
    # For the sampler we use 2 original features (cols 0,1) so Y_idx=2
    # Actually let's use a full setup: 3 original features, append Y as col 3
    sampler = ConditionalDensitySampler(
        clf, reg,
        append_target=True,
        n_permutations=3,
        temperature=1.0,
        random_state=42,
    )
    sampler.set_context(X_train, y_context=y_train)

    mask_cols = [2]   # col 2 is the actionable to impute
    immutable_idx = [1]
    y_idx = X_train.shape[1]  # = 3 (Y appended at position 3 in augmented matrix)

    dag = build_chain_dag(mask_cols, immutable_idx, y_idx)

    # DAG path
    X_filled_dag = sampler.impute_masked(
        X_test, mask_cols=mask_cols, fixed_target=1, dag=dag
    )
    # Random path (baseline)
    X_filled_rand = sampler.impute_masked(
        X_test, mask_cols=mask_cols, fixed_target=1, dag=None
    )

    # Shape must match original
    assert X_filled_dag.shape == X_test.shape, (
        f"DAG path: shape {X_filled_dag.shape} != {X_test.shape}"
    )
    assert X_filled_rand.shape == X_test.shape

    # No NaN in output
    assert not np.any(np.isnan(X_filled_dag)), "DAG path: NaN in output"
    assert not np.any(np.isnan(X_filled_rand)), "Random path: NaN in output"

    # Non-masked columns must be byte-identical to input
    non_masked = [c for c in range(X_test.shape[1]) if c not in mask_cols]
    np.testing.assert_array_equal(
        X_filled_dag[:, non_masked], X_test[:, non_masked],
        err_msg="DAG path: non-masked columns altered"
    )
    np.testing.assert_array_equal(
        X_filled_rand[:, non_masked], X_test[:, non_masked],
        err_msg="Random path: non-masked columns altered"
    )


def test_impute_masked_dag_index_out_of_bounds(models):
    """DAG with an index beyond augmented matrix width should raise AssertionError."""
    clf, reg = models
    X, y = _make_synthetic(n=60, seed=4)
    X_train, X_test = X[:40], X[40:]
    y_train = y[:40]

    sampler = ConditionalDensitySampler(
        clf, reg,
        append_target=True,
        n_permutations=2,
        temperature=1e-9,
        random_state=0,
    )
    sampler.set_context(X_train, y_context=y_train)

    y_idx = X_train.shape[1]  # = 3; valid augmented cols are 0..3
    bad_dag = {2: [y_idx, 999]}  # 999 is out of bounds

    with pytest.raises(AssertionError, match="out of bounds"):
        sampler.impute_masked(X_test, mask_cols=[2], fixed_target=1, dag=bad_dag)
