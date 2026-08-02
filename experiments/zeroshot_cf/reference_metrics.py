"""Score saved Exp-4 counterfactual arrays with the **reference (dicoflex) metrics**.

These are verbatim ports of ``cel/metrics/dicoflex_metrics.py`` (commit ``b9715ef``,
branch ``origin/ofurman/CFN_baselines`` of the `counterfactuals` repo), the module
selected by ``cel/pipelines/conf/metrics/dicoflex.yaml`` and identified there as the
authoritative source of the paper's Table-1 columns:

    validity, proximity_l1_continuous, sparsity_categorical, eps_sparsity,
    lof_score_median_log, pairwise_diversity_mixed, number_of_instances

Why these and not ``cel_standard_metrics.py``: the reference formulas differ from the
generic cel registry in three substantive ways (documented in the reference repo's
``GOAL_table1_metric_config.md``, "Hypotheses ELIMINATED" item 1):

  * ``eps_sparsity`` thresholds the RELATIVE change ``|dx| / (|x| + 1e-8) > 0.05``,
    not the absolute change against ``0.05 * feature_range``.
  * ``sparsity_categorical`` averages over the ONE-HOT COLUMNS, not over the
    one-hot groups.
  * ``pairwise_diversity_mixed`` uses Euclidean on the continuous block and Hamming
    over raw one-hot columns.

All metrics except validity are computed on **valid counterfactuals only**, where
"valid" means ``y_cf_pred == y_target`` — matching ``MetricsOrchestrator`` in the
reference repo (``counterfactuals/metrics/orchestrator.py:96``).

## Protocol difference from the reference repo — read before comparing numbers

The reference pipelines set the target from the *true* label
(``y_target = np.abs(1 - y_test_origin)``, e.g.
``run_cchvae_traintest_pipeline.py:125``). This project **relabels**: the target is the
flip of the discriminator's own prediction on the factual
(``exp4_beam_search.py:144-145``, ``y_target = 1 - disc_model.predict(X_test)``).

Under relabelling the two validity conventions come apart:

  * ``validity_target``  = mean(y_cf_pred == y_target)  — did generation hit the class
    it aimed at. This is the meaningful number here.
  * ``validity_vs_true`` = mean(y_cf_pred != y_test)    — the cel registry's ``validity``
    (``counterfactuals/metrics/basic_metrics.py:65``). Under relabelling this reduces
    algebraically to the discriminator's test accuracy and carries no information about
    the generator. Reported only so the mismatch stays visible.

Both are emitted. Use ``validity_target``.

Usage:
  uv run python experiments/zeroshot_cf/reference_metrics.py
  uv run python experiments/zeroshot_cf/reference_metrics.py --min-n 100
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.spatial.distance import pdist
from sklearn.neighbors import LocalOutlierFactor

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

RESULTS_DIR = Path(__file__).parent / "results"
ARRAYS_DIR = RESULTS_DIR / "arrays"

EPS_SPARSITY_THRESHOLD = 0.05
EPS = 1e-8


# ---------------------------------------------------------------------------
# Reference metric formulas — verbatim ports of cel/metrics/dicoflex_metrics.py
# ---------------------------------------------------------------------------


def proximity_l1_continuous(
    X_test_valid: np.ndarray, X_cf_valid: np.ndarray, cont: List[int]
) -> float:
    """mean(|X_test[:, cont] - X_cf[:, cont]|) over valid CFs."""
    if X_test_valid.size == 0 or len(cont) == 0:
        return 0.0
    return float(np.abs(X_test_valid[:, cont] - X_cf_valid[:, cont]).mean())


def sparsity_categorical(
    X_test_valid: np.ndarray, X_cf_valid: np.ndarray, cat: List[int]
) -> float:
    """mean(X_test[:, cat] != X_cf[:, cat]) over one-hot COLUMNS, valid CFs."""
    if X_test_valid.size == 0 or len(cat) == 0:
        return 0.0
    return float((X_test_valid[:, cat] != X_cf_valid[:, cat]).astype(float).mean())


def eps_sparsity(
    X_test_valid: np.ndarray, X_cf_valid: np.ndarray, cont: List[int]
) -> float:
    """mean(|dx| / (|x| + 1e-8) > 0.05) over continuous features, valid CFs.

    Note this is a *relative* threshold, so it is sensitive to factual values near
    zero — on MinMax-scaled data a feature at x=0 makes any change count as
    significant. That is the reference behaviour and is kept deliberately.
    """
    if X_test_valid.size == 0 or len(cont) == 0:
        return 0.0
    rel = np.abs(X_test_valid[:, cont] - X_cf_valid[:, cont]) / (
        np.abs(X_test_valid[:, cont]) + EPS
    )
    return float((rel > EPS_SPARSITY_THRESHOLD).mean())


def sparsity_categorical_decoded(
    X_test_valid: np.ndarray,
    X_cf_valid: np.ndarray,
    cat: List[int],
    feature_names: Optional[List[str]] = None,
) -> Optional[float]:
    """DIAGNOSTIC, not a reference metric: categorical change rate after argmax decoding.

    ``sparsity_categorical`` above uses exact float inequality, which assumes the method
    emits discrete one-hots (DiCE, CCHVAE and DiCoFlex all do). Beam search emits a
    continuous relaxation — on Law, 0% of the CF categorical cells are exactly 0 or 1
    (values like 0.999 / 0.001) — so every column registers as changed and the reference
    metric saturates at 1.0 regardless of what the CF actually says.

    This variant decodes each one-hot group by argmax and asks whether the selected
    category changed. It is the number that answers "did the category flip", but it is
    NOT what the reference pipeline computes, so it must not be reported in the same
    column as published values.
    """
    if X_test_valid.size == 0 or len(cat) == 0:
        return None

    groups: Dict[str, List[int]] = {}
    for i in cat:
        base = feature_names[i].split("__")[0] if feature_names is not None else str(i)
        groups.setdefault(base, []).append(i)

    changed = np.zeros(len(X_test_valid), dtype=float)
    for cols in groups.values():
        f_arg = np.argmax(X_test_valid[:, cols], axis=1)
        c_arg = np.argmax(X_cf_valid[:, cols], axis=1)
        changed += (f_arg != c_arg).astype(float)
    return float((changed / len(groups)).mean())


def lof_score_median_log(X_cf_valid: np.ndarray, X_train: np.ndarray) -> float:
    """median(log(-lof.score_samples(X_cf_valid) + 1e-8)), LOF fit on X_train.

    LocalOutlierFactor is constructed with library defaults (n_neighbors=20) exactly
    as in the reference module. Lower is more plausible.
    """
    if X_cf_valid.size == 0:
        return 0.0
    lof = LocalOutlierFactor(novelty=True)
    lof.fit(X_train)
    return float(np.median(np.log(-lof.score_samples(X_cf_valid) + EPS)))


def pairwise_diversity_mixed(
    X_cf_valid: np.ndarray,
    X_test_valid: np.ndarray,
    cont: List[int],
    cat: List[int],
) -> Optional[float]:
    """Mean within-factual pairwise distance (Euclidean cont + Hamming*len(cat)).

    NOT COMPUTABLE FOR THIS PROJECT — always returns None. The metric measures spread
    across the K counterfactuals a method proposes for one factual; the reference
    runner feeds it ``cf_group_ids`` from ``result.extras``. Beam search here emits
    exactly one CF per factual, so there is no within-factual set to measure.

    Running the reference implementation verbatim does not return a clean 0 either: it
    groups by the *value* of the factual row, and HELOC's test split contains one block
    of 115 byte-identical rows (2092 rows, 1978 unique). Generation is deterministic, so
    those 115 produce identical CFs and the group contributes a pairwise distance of
    exactly 0 — which would be reported as perfect diversity. That is an artifact of
    duplicate factuals, not a property of the method, so we refuse the number outright.
    """
    return None


def _pairwise_diversity_mixed_reference(
    X_cf_valid: np.ndarray,
    X_test_valid: np.ndarray,
    cont: List[int],
    cat: List[int],
) -> Optional[float]:
    """Verbatim reference implementation, retained for provenance. See caveat above."""
    if X_cf_valid.size == 0:
        return None
    n_features = len(cont) + len(cat)
    if n_features == 0:
        return None

    groups: Dict[tuple, List[np.ndarray]] = {}
    for orig_row, cf_row in zip(X_test_valid, X_cf_valid):
        groups.setdefault(tuple(orig_row.tolist()), []).append(
            cf_row.astype(np.float32)
        )

    group_diversities: List[float] = []
    for cf_group in groups.values():
        K = len(cf_group)
        if K < 2:
            continue
        X_cf_group = np.vstack(cf_group)
        num_pairs = K * (K - 1) // 2
        d_cont = (
            pdist(X_cf_group[:, cont], metric="euclidean")
            if len(cont) > 0
            else np.zeros(num_pairs)
        )
        d_cat = (
            pdist(X_cf_group[:, cat], metric="hamming") * len(cat)
            if len(cat) > 0
            else np.zeros(num_pairs)
        )
        group_diversities.append(float(np.mean((d_cont + d_cat) / n_features)))

    if not group_diversities:
        return None
    return float(np.mean(group_diversities))


# ---------------------------------------------------------------------------


def score_cell(
    npz_path: Path,
    dataset_name: Optional[str] = None,
    tag: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute the reference Table-1 metrics for one saved (dataset, regime) cell.

    ``dataset_name``/``tag`` default to being parsed out of the filename; pass them
    explicitly for config-tagged Exp-7 sweep arrays, whose stems carry a
    ``__<run-id>`` suffix that this parse would mis-split.
    """
    from experiments.zeroshot_cf.data import load_dataset  # noqa: PLC0415
    from experiments.zeroshot_cf.discriminator import train_discriminator  # noqa: PLC0415

    if dataset_name is None or tag is None:
        stem = npz_path.stem.replace("exp4_", "").replace("_cfs", "")
        dataset_name, tag = stem.rsplit("_", 1)

    z = np.load(npz_path)
    X_cf_raw = z["X_cf"]
    # Score the delivered CF; clip the rare out-of-[0,1] cells into range, matching
    # cel_standard_metrics.py so the two scorers see identical inputs.
    X_cf = np.clip(X_cf_raw, 0.0, 1.0)
    X_test = z["X_test"]
    y_test = z["y_test"].astype(np.int64).squeeze()
    y_target = z["y_target"].astype(np.int64).squeeze()

    bundle = load_dataset(dataset_name)
    disc_model = train_discriminator(
        bundle.X_train, bundle.y_train, X_test, y_test, dataset_name
    )
    y_cf_pred = np.asarray(disc_model.predict(X_cf)).squeeze()

    cont = list(bundle.numerical_features_indices)
    cat = list(bundle.categorical_features_indices)

    # "valid" per the reference orchestrator: y_cf_pred == y_target
    valid_mask = y_cf_pred == y_target
    X_cf_valid = X_cf[valid_mask]
    X_test_valid = X_test[valid_mask]

    diversity = pairwise_diversity_mixed(X_cf_valid, X_test_valid, cont, cat)

    # Diagnostic: does the CF even lie on the one-hot simplex? The reference formulas
    # assume it does. Fraction of categorical cells that are exactly 0 or 1.
    onehot_exact = (
        float(np.isin(X_cf_valid[:, cat], [0.0, 1.0]).mean()) if len(cat) > 0 else None
    )

    return {
        "dataset": dataset_name,
        "set": tag,
        "number_of_instances": int(X_cf.shape[0]),
        "n_valid": int(valid_mask.sum()),
        # the meaningful validity under this project's relabelling protocol
        "validity_target": float((y_cf_pred == y_target).mean()),
        # cel-registry convention; degenerates to disc accuracy under relabelling
        "validity_vs_true": float((y_cf_pred != y_test).mean()),
        "disc_accuracy": float(
            (np.asarray(disc_model.predict(X_test)).squeeze() == y_test).mean()
        ),
        "proximity_l1_continuous": proximity_l1_continuous(
            X_test_valid, X_cf_valid, cont
        ),
        "sparsity_categorical": sparsity_categorical(X_test_valid, X_cf_valid, cat),
        "eps_sparsity": eps_sparsity(X_test_valid, X_cf_valid, cont),
        "lof_score_median_log": lof_score_median_log(X_cf_valid, bundle.X_train),
        "pairwise_diversity_mixed": diversity,
        # --- diagnostics, NOT reference metrics ---
        "sparsity_categorical_decoded": sparsity_categorical_decoded(
            X_test_valid, X_cf_valid, cat, bundle.feature_names
        ),
        "cf_onehot_cells_exactly_binary": onehot_exact,
        "n_continuous": len(cont),
        "n_categorical_onehot": len(cat),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-n", type=int, default=0)
    parser.add_argument(
        "--json", type=str, default=str(RESULTS_DIR / "exp4_reference_metrics.json")
    )
    args = parser.parse_args()

    paths = sorted(ARRAYS_DIR.glob("exp4_*_cfs.npz"))
    rows: List[Dict[str, Any]] = []
    for p in paths:
        n = int(np.load(p)["X_cf"].shape[0])
        if n < args.min_n:
            print(f"skip {p.name} (n={n} < {args.min_n})")
            continue
        print(f"\n=== {p.name} (n={n}) ===")
        row = score_cell(p)
        for k, v in row.items():
            if isinstance(v, float):
                print(f"  {k:28s} {v:.4f}")
            else:
                print(f"  {k:32s} {v if v is not None else 'N/A'}")
        rows.append(row)

    if rows:
        Path(args.json).write_text(json.dumps(rows, indent=2))
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
