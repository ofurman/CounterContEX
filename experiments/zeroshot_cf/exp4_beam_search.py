"""Experiment 4: from-scratch counterfactuals via task-guided beam search.

Every feature of each counterfactual is generated autoregressively, conditioned
only on Y=target — the factual is *never observed*, entering solely through a
per-feature proximity penalty. Immutable columns are soft-frozen with a large λ
(still generated, but strongly pulled to the factual value).

Contrast with Exp 2/3 (imputation: freeze immutables, mask only actionables).
Because immutables are now generated, ``true_actionability`` is no longer 1.0 by
construction — immutable *drift* is reported as a first-class metric instead.

Context: all_classes (mandatory — a constant Y in context trips TabPFN's
constant-feature validator; Y must vary so the appended-Y conditioning works).

Outputs:
  results/exp4_<dataset>_metrics.csv   — per-dataset metric row
  results/exp4_summary.md              — aggregate table + notes

Usage:
  uv run python experiments/zeroshot_cf/exp4_beam_search.py --dataset moons
  uv run python experiments/zeroshot_cf/exp4_beam_search.py --dataset heloc \\
      --beam-width 8 --n-candidates 6 --lambda-actionable 1.0 --lambda-immutable 50
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

N_ESTIMATORS = 4
MAX_CONTEXT = 256

_DATASET_PARAMS = {
    "moons": {"max_test": 100},
    "heloc": {"max_test": 30},
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Generate from-scratch CFs for one dataset. Returns (X_test, y_test, X_cf, info)."""
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

    print(f"\n=== Experiment 4 (beam, from scratch): {dataset_name.upper()} ===")
    print(
        f"  beam_width={beam_width}, n_candidates={n_candidates}, "
        f"lambda_actionable={lambda_actionable}, lambda_immutable={lambda_immutable}, "
        f"max_context={max_context}"
    )

    bundle = load_dataset(dataset_name)
    X_train, y_train = bundle.X_train, bundle.y_train
    X_test = bundle.X_test[:MAX_TEST]
    y_test = bundle.y_test[:MAX_TEST]
    n, d = X_test.shape

    actionable_idx, immutable_idx = get_actionable_immutable(dataset_name, bundle)
    print(
        f"Features: {d} total, {len(actionable_idx)} actionable, "
        f"{len(immutable_idx)} immutable (soft-frozen); all generated from scratch"
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
        t0 = time.perf_counter()
        X_cf_batch, aux = generate_cf_beam(
            reg,
            X_context=X_train,  # all classes — Y must vary in context
            y_context=y_train,
            X_factual=X_batch,
            target_class=target_cls,
            ordering=ordering,
            immutable_idx=immutable_idx,
            config=cfg,
            disc_model=disc_model,
        )
        print(
            f"    beam search: {len(X_batch)} pts, {d} features "
            f"→ {time.perf_counter() - t0:.2f}s "
            f"(oob_fallback={aux['n_oob_fallback']})"
        )

        X_cf[test_idx] = X_cf_batch
        immutable_drift[test_idx] = aux["immutable_drift"]
        chosen_valid[test_idx] = aux["chosen_valid"]
        n_oob_fallback += aux["n_oob_fallback"]

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


def run_dataset(dataset_name: str, **kwargs) -> Dict[str, float]:
    X_test, y_test, X_cf, info = generate_counterfactuals_beam(dataset_name, **kwargs)
    return evaluate_and_report_beam(dataset_name, X_test, y_test, X_cf, info)


def write_summary(all_metrics: List[Dict], settings: Dict) -> None:
    lines = [
        "# Experiment 4: From-Scratch Counterfactuals via Task-Guided Beam Search",
        "",
        f"Settings: beam_width={settings['beam_width']}, "
        f"n_candidates={settings['n_candidates']}, "
        f"lambda_actionable={settings['lambda_actionable']}, "
        f"lambda_immutable={settings['lambda_immutable']}, "
        f"max_context={settings['max_context']}, context_type=all_classes",
        "",
        "## Metrics",
        "",
        "| Dataset | Validity | LOF | Proximity L2 | OOB frac | Immut drift (mean) | "
        "True-action |",
        "|---------|---------|-----|-------------|---------|-------------------|"
        "------------|",
    ]
    for m in all_metrics:
        lines.append(
            f"| {m['dataset']} "
            f"| {m.get('validity', float('nan')):.3f} "
            f"| {m.get('lof_scores_cf', float('nan')):.3f} "
            f"| {m.get('proximity_l2_jaccard', float('nan')):.4f} "
            f"| {m.get('frac_oob', float('nan')):.3f} "
            f"| {m.get('immutable_drift_mean', float('nan')):.4f} "
            f"| {m.get('true_actionability', float('nan')):.3f} |"
        )
    lines += [
        "",
        "## Notes",
        "",
        "- **From scratch**: every feature is generated autoregressively conditioned "
        "only on Y=target; the factual enters solely via the per-feature proximity "
        "penalty `λ·|f − factual|`.",
        "- Immutables are soft-frozen (large `lambda_immutable`); they are still "
        "generated, so `true_actionability` < 1.0 is expected and `immutable_drift` "
        "(mean |Δ| over immutable columns) quantifies how far they wandered.",
        "- `validity`: fraction whose discriminator class == target (higher = better).",
        "- `lof_scores_cf`: mean negative-LOF plausibility on unclipped CFs (lower = better).",
        "- `proximity_l2_jaccard`: mean L2 to factual on *valid* CFs (lower = closer).",
        "- `frac_oob`: fraction of CF rows with a feature outside [0,1] before clipping. "
        "Hard [0,1] candidate rejection during search should keep this low.",
        "",
        "Comparison vs. Exp 2 (imputation baseline) is recorded in `results/REPORT.md`.",
    ]
    out = RESULTS_DIR / "exp4_summary.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 4: from-scratch beam-search CFs"
    )
    parser.add_argument("--dataset", choices=["moons", "heloc", "all"], default="moons")
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--n-candidates", type=int, default=6)
    parser.add_argument("--lambda-actionable", type=float, default=1.0)
    parser.add_argument("--lambda-immutable", type=float, default=100.0)
    parser.add_argument("--max-context", type=int, default=MAX_CONTEXT)
    parser.add_argument(
        "--max-test",
        type=int,
        default=None,
        help="Test points to evaluate. Default per-dataset cap (moons=100, heloc=30); "
        "-1 for the full stratified split.",
    )
    args = parser.parse_args()
    datasets = ["moons", "heloc"] if args.dataset == "all" else [args.dataset]

    kwargs = dict(
        beam_width=args.beam_width,
        n_candidates=args.n_candidates,
        lambda_actionable=args.lambda_actionable,
        lambda_immutable=args.lambda_immutable,
        max_context=args.max_context,
        max_test=args.max_test,
    )
    all_metrics = []
    for ds in datasets:
        m = run_dataset(ds, **kwargs)
        all_metrics.append({"dataset": ds, **m})

    write_summary(all_metrics, settings=kwargs)
    print("\nExperiment 4 done.")


if __name__ == "__main__":
    main()
