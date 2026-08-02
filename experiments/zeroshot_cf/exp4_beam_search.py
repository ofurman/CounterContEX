"""Experiment 4: counterfactuals via task-guided beam search, in two regimes.

Both regimes use identical beam settings and differ **only** in whether the
immutable features are masked:

- **Set 1 — frozen immutables** (``--set frozen``): immutables are *observed*
  (held at the factual value); the beam generates only the actionable features.
  Directly comparable to the Exp 2/3 imputation baseline; ``true_actionability=1.0``.
- **Set 2 — from scratch** (``--set fromscratch``): *no* feature is masked; every
  feature is generated autoregressively conditioned only on Y=target. The factual
  enters solely via the per-feature proximity penalty; immutables drift (reported).

Context: all_classes (mandatory — a constant Y in context trips TabPFN's
constant-feature validator; Y must vary so the appended-Y conditioning works).
For MOONS (no immutables), Set 1 ≡ Set 2.

Outputs:
  results/exp4_<dataset>_<set>_metrics.csv   — per-dataset, per-regime metric row
  results/exp4_summary.md                    — combined two-regime table + notes

Usage:
  uv run python experiments/zeroshot_cf/exp4_beam_search.py --dataset all --set both
  uv run python experiments/zeroshot_cf/exp4_beam_search.py --dataset heloc \\
      --set fromscratch --beam-width 8 --lambda-actionable 1.0
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
# Raw generated CF arrays (gitignored) — lets metrics be recomputed without
# re-running the ~0.85 s/CF beam search.
ARRAYS_DIR = RESULTS_DIR / "arrays"

N_ESTIMATORS = 4
MAX_CONTEXT = 256
# Query points per batched beam-search call. Bounds the per-step predict batch
# (chunk x beam_width rows) and gives progress output on full-split runs.
#
# NOTE: results are *not* chunk-invariant. TabPFN's predictions depend on the
# composition of the predict batch, so changing chunk_size perturbs the generated
# CFs (verified: chunk=40 vs chunk=7 on law differ by up to ~1.0 on a one-hot
# column). Larger chunks are also faster per CF. The default is therefore set high
# enough that a whole target class is normally one chunk, preserving the
# single-call-per-class semantics of the earlier runs; lower it only if memory
# forces you to, and hold it fixed across runs you intend to compare.
DEFAULT_CHUNK = 4096

_DATASET_PARAMS = {
    "moons": {"max_test": 100},
    "heloc": {"max_test": 30},
    "law": {"max_test": 100},
}


def _actionable_order_by_coef(disc_model, actionable_idx: List[int]) -> List[int]:
    """Order actionable columns by descending |LR coefficient| (most class-informative
    first), so the strongest anchors are generated early in the chain."""
    coef = np.abs(disc_model._clf.coef_[0])
    act = np.asarray(actionable_idx)
    return act[np.argsort(-coef[act])].tolist()


def generate_counterfactuals_beam(
    dataset_name: str,
    beam_width: int = 8,
    n_candidates: int = 6,
    lambda_actionable: float = 1.0,
    lambda_immutable: float = 100.0,
    max_context: int = MAX_CONTEXT,
    max_test: Optional[int] = None,
    freeze_immutable: bool = False,
    chunk_size: int = DEFAULT_CHUNK,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Generate beam-search CFs for one dataset. Returns (X_test, y_test, X_cf, info).

    ``freeze_immutable=False`` (Set 2): every feature generated from scratch.
    ``freeze_immutable=True``  (Set 1): immutables observed (held at factual),
    only actionables generated — comparable to the Exp 2/3 imputation baseline.

    Query points are processed in chunks of ``chunk_size`` within each target class.
    The beam search is already batched across queries (one regressor fit + one
    batched predict per step for all query x beam rows), so chunking does not change
    results — each query's beams are independent and the context subsample is fixed
    by ``random_state``. It bounds the predict batch (and peak memory) on full-split
    runs and gives incremental progress output.
    """
    from experiments.zeroshot_cf.beam_search import (
        BeamConfig,
        build_generation_ordering,
        generate_cf_beam,
    )
    from experiments.zeroshot_cf.checkpoints import get_models
    from experiments.zeroshot_cf.data import get_actionable_immutable, load_dataset
    from experiments.zeroshot_cf.discriminator import train_discriminator

    params = _DATASET_PARAMS.get(dataset_name, {"max_test": 30})
    if max_test is not None and max_test < 0:
        MAX_TEST = None
    elif max_test is not None:
        MAX_TEST = max_test
    else:
        MAX_TEST = params["max_test"]

    mode = (
        "frozen-immutable (Set 1, ~Exp2 baseline)"
        if freeze_immutable
        else ("from-scratch (Set 2, no masking)")
    )
    print(f"\n=== Experiment 4 (beam) [{mode}]: {dataset_name.upper()} ===")
    print(
        f"  beam_width={beam_width}, n_candidates={n_candidates}, "
        f"lambda_actionable={lambda_actionable}, lambda_immutable={lambda_immutable}, "
        f"max_context={max_context}, freeze_immutable={freeze_immutable}"
    )

    bundle = load_dataset(dataset_name)
    X_train, y_train = bundle.X_train, bundle.y_train
    X_test = bundle.X_test[:MAX_TEST]
    y_test = bundle.y_test[:MAX_TEST]
    n, d = X_test.shape

    actionable_idx, immutable_idx = get_actionable_immutable(dataset_name, bundle)
    immut_note = "observed/held" if freeze_immutable else "generated (drift reported)"
    print(
        f"Features: {d} total, {len(actionable_idx)} actionable, "
        f"{len(immutable_idx)} immutable ({immut_note})"
    )
    print(f"Test set (capped): {n} points")

    disc_model = train_discriminator(X_train, y_train, X_test, y_test, dataset_name)
    y_pred = disc_model.predict(X_test)
    y_target = 1 - y_pred
    print(f"Target distribution: {np.bincount(y_target)}")

    actionable_order = _actionable_order_by_coef(disc_model, actionable_idx)
    ordering = build_generation_ordering(d, immutable_idx, actionable_order)
    print(
        f"  Generation order (immutables first, then |coef|-desc actionables): "
        f"{[bundle.feature_names[i] for i in ordering]}"
    )

    print("Loading TabPFN models …")
    _, reg = get_models(n_estimators=N_ESTIMATORS)

    X_cf = np.empty((n, d), dtype=np.float64)
    immutable_drift = np.full(n, np.nan)
    chosen_valid = np.zeros(n, dtype=bool)
    n_oob_fallback = 0

    for target_cls in np.unique(y_target):
        target_cls = int(target_cls)
        test_idx = np.where(y_target == target_cls)[0]
        X_batch = X_test[test_idx]
        if len(X_batch) == 0:
            continue
        print(f"\n  Target class {target_cls}: {len(X_batch)} test points")

        cfg = BeamConfig(
            beam_width=beam_width,
            n_candidates=n_candidates,
            lambda_actionable=lambda_actionable,
            lambda_immutable=lambda_immutable,
            max_context=max_context,
            random_state=42 + target_cls,
        )
        n_gen = len(actionable_idx) if freeze_immutable else d
        n_chunks = max(1, -(-len(X_batch) // chunk_size))  # ceil div
        t_class = time.perf_counter()

        for ci in range(n_chunks):
            lo, hi = ci * chunk_size, min((ci + 1) * chunk_size, len(X_batch))
            sub_idx = test_idx[lo:hi]
            t0 = time.perf_counter()
            X_cf_batch, aux = generate_cf_beam(
                reg,
                X_context=X_train,  # all classes — Y must vary in context
                y_context=y_train,
                X_factual=X_batch[lo:hi],
                target_class=target_cls,
                ordering=ordering,
                immutable_idx=immutable_idx,
                config=cfg,
                disc_model=disc_model,
                freeze_immutable=freeze_immutable,
            )
            dt = time.perf_counter() - t0
            print(
                f"    chunk {ci + 1}/{n_chunks}: {hi - lo} pts, {n_gen} gen. features "
                f"→ {dt:.2f}s ({dt / max(1, hi - lo):.3f}s/CF, "
                f"oob_fallback={aux['n_oob_fallback']})",
                flush=True,
            )

            X_cf[sub_idx] = X_cf_batch
            immutable_drift[sub_idx] = aux["immutable_drift"]
            chosen_valid[sub_idx] = aux["chosen_valid"]
            n_oob_fallback += aux["n_oob_fallback"]

        print(
            f"    class {target_cls} total: {len(X_batch)} pts in "
            f"{time.perf_counter() - t_class:.1f}s",
            flush=True,
        )

    return (
        X_test,
        y_test,
        X_cf,
        {
            "bundle": bundle,
            "y_pred": y_pred,
            "y_target": y_target,
            "actionable_idx": actionable_idx,
            "immutable_idx": immutable_idx,
            "ordering": ordering,
            "disc_model": disc_model,
            "immutable_drift": immutable_drift,
            "chosen_valid": chosen_valid,
            "n_oob_fallback": n_oob_fallback,
            "beam_width": beam_width,
            "n_candidates": n_candidates,
            "lambda_actionable": lambda_actionable,
            "lambda_immutable": lambda_immutable,
            "max_context": max_context,
            "freeze_immutable": freeze_immutable,
        },
    )


def evaluate_and_report_beam(
    dataset_name: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_cf: np.ndarray,
    info: Dict,
    write_csv: bool = True,
) -> Dict[str, float]:
    """Evaluate from-scratch CFs. Unlike Exp2, immutables are NOT asserted unchanged —
    their drift is reported instead (true_actionability is informational, not a gate)."""
    from experiments.zeroshot_cf.metrics_harness import compute_metrics, print_metrics

    bundle = info["bundle"]
    immutable_idx = info["immutable_idx"]
    disc_model = info["disc_model"]
    y_target = info["y_target"]

    oob_mask = (X_cf < 0.0) | (X_cf > 1.0)
    frac_oob = float(oob_mask.any(axis=1).mean())
    print(
        f"\n  Out-of-[0,1] fraction (pre-clip): {frac_oob:.3f} "
        f"({int(oob_mask.any(axis=1).sum())}/{len(X_cf)})"
    )

    X_cf_clipped = np.clip(X_cf, 0.0, 1.0)

    drift = info["immutable_drift"]
    mean_drift = float(np.nanmean(drift)) if immutable_idx else 0.0
    max_drift = float(np.nanmax(drift)) if immutable_idx else 0.0
    if immutable_idx:
        print(
            f"  Immutable soft-freeze drift (mean|Δ| over {len(immutable_idx)} cols): "
            f"mean={mean_drift:.4f}, max={max_drift:.4f}"
        )

    metrics = compute_metrics(
        disc_model=disc_model,
        X_cf=X_cf_clipped,
        X_test=X_test,
        X_train=bundle.X_train,
        y_test=y_test,
        y_target=y_target,
        immutable_idx=immutable_idx,
        X_cf_lof=X_cf,
        categorical_idx=bundle.categorical_features_indices,
        feature_names=bundle.feature_names,
    )
    metrics["frac_oob"] = frac_oob
    metrics["immutable_drift_mean"] = mean_drift
    metrics["immutable_drift_max"] = max_drift
    metrics["n_oob_fallback"] = int(info.get("n_oob_fallback", 0))
    print_metrics(metrics, prefix=dataset_name)

    if write_csv:
        csv_path = RESULTS_DIR / f"exp4_{dataset_name}_metrics.csv"
        row = {"dataset": dataset_name, **metrics}
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)
        print(f"\n  Wrote {csv_path}")

    return metrics


# The two report regimes. Each (tag, freeze_immutable) pair becomes one "set".
_SETS = [
    ("frozen", True, "Set 1 — frozen immutables (actionable, ~Exp2 baseline)"),
    ("fromscratch", False, "Set 2 — from scratch (no masking, nothing frozen)"),
]


def run_dataset(dataset_name: str, tag: str, freeze_immutable: bool, **kwargs) -> Dict:
    X_test, y_test, X_cf, info = generate_counterfactuals_beam(
        dataset_name, freeze_immutable=freeze_immutable, **kwargs
    )

    # Persist the raw generated arrays. Generation is the expensive step (~0.85 s/CF
    # on HELOC), so saving them means any new metric can be computed later without
    # re-running the search.
    ARRAYS_DIR.mkdir(parents=True, exist_ok=True)
    npz_path = ARRAYS_DIR / f"exp4_{dataset_name}_{tag}_cfs.npz"
    np.savez_compressed(
        npz_path,
        X_cf=X_cf,
        X_test=X_test,
        y_test=y_test,
        y_target=info["y_target"],
        y_pred=info["y_pred"],
        immutable_idx=np.asarray(info["immutable_idx"], dtype=np.int64),
        chosen_valid=info["chosen_valid"],
        immutable_drift=info["immutable_drift"],
    )
    print(f"  Saved arrays → {npz_path}")

    metrics = evaluate_and_report_beam(
        dataset_name, X_test, y_test, X_cf, info, write_csv=False
    )
    row = {
        "dataset": dataset_name,
        "set": tag,
        "freeze_immutable": freeze_immutable,
        **metrics,
    }
    csv_path = RESULTS_DIR / f"exp4_{dataset_name}_{tag}_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    print(f"\n  Wrote {csv_path}")
    return row


def write_summary(all_rows: List[Dict], settings: Dict) -> None:
    lines = [
        "# Experiment 4: Counterfactuals via Task-Guided Beam Search",
        "",
        "Two regimes, identical beam settings — they differ **only** in whether the "
        "immutable features are masked:",
        "",
        "- **Set 1 (frozen immutables)** — immutables are *observed* (held at the "
        "factual value); the beam generates only the actionable features. Directly "
        "comparable to the Exp 2/3 imputation baseline; `true_actionability = 1.0`.",
        "- **Set 2 (from scratch)** — *no* feature is masked; every feature is "
        "generated, conditioned only on `Y=target`. The factual enters only via the "
        "proximity penalty.",
        "",
        f"Settings: beam_width={settings['beam_width']}, "
        f"n_candidates={settings['n_candidates']}, "
        f"lambda_actionable={settings['lambda_actionable']}, "
        f"max_context={settings['max_context']}, context_type=all_classes. "
        "(For MOONS and LAW, which have no immutables, Set 1 ≡ Set 2.)",
        "",
        "## Metrics",
        "",
        "| Dataset | Set | n | Validity | LOF | Prox L2 | Prox L1 | L0 | Sparsity(tol) "
        "| OOB frac | Immut drift | True-action |",
        "|---------|-----|---|---------|-----|--------|--------|----|--------------"
        "|---------|------------|------------|",
    ]
    for m in all_rows:
        lines.append(
            f"| {m['dataset']} "
            f"| {m['set']} "
            f"| {int(m.get('n_total', 0))} "
            f"| {m.get('validity', float('nan')):.3f} "
            f"| {m.get('lof_scores_cf', float('nan')):.3f} "
            f"| {m.get('proximity_l2_jaccard', float('nan')):.4f} "
            f"| {m.get('proximity_l1', float('nan')):.4f} "
            f"| {m.get('l0_count_mean', float('nan')):.2f} "
            f"| {m.get('sparsity_tol', float('nan')):.3f} "
            f"| {m.get('frac_oob', float('nan')):.3f} "
            f"| {m.get('immutable_drift_mean', float('nan')):.4f} "
            f"| {m.get('true_actionability', float('nan')):.3f} |"
        )

    cat_rows = [m for m in all_rows if "cat_change_rate" in m]
    if cat_rows:
        lines += [
            "",
            "### Categorical-aware distances (datasets with one-hot columns)",
            "",
            "Pure L2 over one-hot columns overstates distance — a single category "
            "flip contributes ~1.41 to L2 on its own. These split the continuous "
            "part from the decoded categorical part.",
            "",
            "| Dataset | Set | Prox L2 (continuous only) | Categorical change rate | "
            "# categorical features |",
            "|---------|-----|--------------------------|-------------------------|"
            "------------------------|",
        ]
        for m in cat_rows:
            lines.append(
                f"| {m['dataset']} | {m['set']} "
                f"| {m.get('proximity_l2_continuous', float('nan')):.4f} "
                f"| {m.get('cat_change_rate', float('nan')):.3f} "
                f"| {int(m.get('n_categorical_features', 0))} |"
            )
    lines += [
        "",
        "## Notes",
        "",
        "- `validity`: fraction whose discriminator class == target (higher = better).",
        "- `lof_scores_cf`: mean negative-LOF plausibility on unclipped CFs (lower = better).",
        "- `proximity_l2_jaccard` / `proximity_l1`: mean L2 / L1 to factual on *valid* "
        "CFs (lower = closer).",
        "- `L0`: mean number of features changed by more than 1e-3; `Sparsity(tol)` is "
        "that as a fraction. The exact-equality `sparsity` metric saturates at 1.0 for "
        "continuously generated CFs and is not reported here.",
        "- `frac_oob`: fraction of CF rows with a feature outside [0,1] before clipping; "
        "the hard [0,1] candidate rejection keeps this at 0.",
        "- **Set 2** generates immutables too, so `true_actionability` < 1.0 and "
        "`immutable_drift` reports how far they wandered.",
        "",
        "Full comparison vs. Exp 2 (imputation baseline) is in `results/REPORT.md §8`.",
    ]
    out = RESULTS_DIR / "exp4_summary.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 4: beam-search CFs (frozen-immutable vs from-scratch)"
    )
    parser.add_argument(
        "--dataset",
        default="moons",
        help="Comma-separated: any of moons, heloc, law (or 'all' = moons,heloc,law). "
        "e.g. --dataset heloc,law",
    )
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--n-candidates", type=int, default=6)
    parser.add_argument("--lambda-actionable", type=float, default=1.0)
    parser.add_argument("--max-context", type=int, default=MAX_CONTEXT)
    parser.add_argument(
        "--max-test",
        type=int,
        default=None,
        help="Test points to evaluate. Default per-dataset cap (moons=100, heloc=30); "
        "-1 for the full stratified split.",
    )
    parser.add_argument(
        "--set",
        choices=["frozen", "fromscratch", "both"],
        default="both",
        help="Which regime(s) to run (default: both).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK,
        help=f"Query points per batched beam call (default {DEFAULT_CHUNK}). "
        "Bounds the per-step predict batch; results are chunk-invariant.",
    )
    args = parser.parse_args()
    _ALL = ["moons", "heloc", "law"]
    datasets: List[str] = []
    for tok in (t.strip() for t in args.dataset.split(",")):
        if tok == "all":
            datasets.extend(d for d in _ALL if d not in datasets)
        elif tok in _ALL:
            if tok not in datasets:
                datasets.append(tok)
        else:
            parser.error(f"unknown dataset {tok!r}; choose from {_ALL} or 'all'")
    sets = _SETS if args.set == "both" else [s for s in _SETS if s[0] == args.set]

    settings = dict(
        beam_width=args.beam_width,
        n_candidates=args.n_candidates,
        lambda_actionable=args.lambda_actionable,
        max_context=args.max_context,
    )
    kwargs = dict(
        beam_width=args.beam_width,
        n_candidates=args.n_candidates,
        lambda_actionable=args.lambda_actionable,
        # Set 2 (from scratch): immutables generated with the same λ as actionables
        # (no special freeze). Set 1 ignores lambda_immutable (immutables observed).
        lambda_immutable=args.lambda_actionable,
        max_context=args.max_context,
        max_test=args.max_test,
        chunk_size=args.chunk_size,
    )
    all_rows = []
    for tag, freeze, label in sets:
        print(f"\n########## {label} ##########")
        for ds in datasets:
            all_rows.append(run_dataset(ds, tag, freeze, **kwargs))

    write_summary(all_rows, settings=settings)
    print("\nExperiment 4 done.")


if __name__ == "__main__":
    main()
