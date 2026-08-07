"""Experiment 2: Counterfactual generation via conditional density estimation.

For each factual test point x with predicted class c:
  - Set target = 1 - c (binary class flip).
  - Build context from target-class train rows.
  - NaN-mask actionable features, fix appended Y=target, impute → x_cf.
  - Assemble the full CF matrix and evaluate the 6-metric suite.

Batching: points are grouped by target class so context is fit only twice per
dataset (once per target class), not once per point.

Outputs:
  results/exp2_<dataset>_metrics.csv   — per-dataset metric row
  results/exp2_examples.md             — human-readable factual vs. CF examples
  results/exp2_summary.md              — aggregate table + notes

Usage:
  uv run python experiments/zeroshot_cf/exp2_counterfactuals.py --dataset moons
  uv run python experiments/zeroshot_cf/exp2_counterfactuals.py --dataset heloc
  uv run python experiments/zeroshot_cf/exp2_counterfactuals.py --dataset all

  # MOONS recommended config (Stage 6 sweep best):
  uv run python experiments/zeroshot_cf/exp2_counterfactuals.py \\
      --dataset moons --temperature 0.5 --context-type all_classes
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MAX_CONTEXT = 256
N_PERMUTATIONS = 5
TEMPERATURE = 1.0  # generation temperature (vs 1e-9 MAP used in Exp 1)
N_ESTIMATORS = 4

_DATASET_PARAMS = {
    "moons": {"max_test": 100},
    "heloc": {"max_test": 50},
}


def _impute_range(sampler, X_src, X_out, mask_cols, fixed_target, dag, s, e, failed):
    """Impute rows [s:e); on a numerical failure (float32 overflow in the
    autoregressive chain at t>0) bisect down to the offending row(s). Rows that
    fail even individually keep their factual values (→ counted invalid) and their
    batch-relative index is appended to ``failed``."""
    try:
        X_out[s:e] = sampler.impute_masked(
            X_src[s:e], mask_cols=mask_cols, fixed_target=fixed_target, dag=dag
        )
    except Exception as ex:  # noqa: BLE001 — isolate per-row numerical blow-ups
        if e - s <= 1:
            failed.append(s)
            print(f"    [robust] row {s} failed ({type(ex).__name__}); left as factual")
            return
        mid = (s + e) // 2
        print(f"    [robust] chunk [{s}:{e}] failed ({type(ex).__name__}); bisecting")
        _impute_range(sampler, X_src, X_out, mask_cols, fixed_target, dag, s, mid, failed)
        _impute_range(sampler, X_src, X_out, mask_cols, fixed_target, dag, mid, e, failed)


def robust_impute(sampler, X_batch, mask_cols, fixed_target, dag, chunk=256):
    """Chunked, failure-isolating wrapper around ``impute_masked`` for large test
    sets. Per-row results are identical to a single call (rows are conditionally
    independent given the fixed context); chunking only localizes the rare row
    whose sampled value overflows float32 when fed to the next column in the chain.
    Returns (X_cf_batch, failed_idx) with any non-finite survivors mapped to an
    out-of-range sentinel so they are counted as OOB and then clipped downstream."""
    n = len(X_batch)
    out = np.asarray(X_batch, dtype=np.float64).copy()
    failed: list = []
    for s in range(0, n, chunk):
        _impute_range(sampler, X_batch, out, mask_cols, fixed_target, dag,
                      s, min(s + chunk, n), failed)
    out = np.nan_to_num(out, nan=1e6, posinf=1e6, neginf=-1e6)
    return out, failed


def generate_counterfactuals(
    dataset_name: str,
    temperature: float = TEMPERATURE,
    n_permutations: int = N_PERMUTATIONS,
    max_context: int = MAX_CONTEXT,
    context_type: str = "target_only",
    ordering: str = "random",
    actionable_set: str = "full",
    reduced_k: int = 6,
    max_test: Optional[int] = None,
    base_seed: int = 42,
    disc_type: str = "lr",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Generate CFs for one dataset. Returns (X_test, y_test, X_cf, info_dict).

    Args:
        ordering: 'random' (default: random permutations) or 'dag' (Y → immutables
                  → actionable chain via explicit DAG).
        actionable_set: 'full' (all actionable features) or 'reduced' (top-reduced_k
                        actionable features by |LR coef|). HELOC only; for MOONS,
                        'reduced' is treated as 'full' (only 2 actionable features).
        reduced_k: Number of top actionable features to mask when actionable_set='reduced'.
        max_test: Override for the number of test points (default: from _DATASET_PARAMS).
        base_seed: Base random_state for the per-target-class sampler (default: 42,
                   matching the previously-hardcoded value). Vary this across repeated
                   calls to draw independent CFs per test point (e.g. for a diversity
                   metric) — each call is otherwise fully deterministic.
        disc_type: Validity-oracle classifier — 'lr' (default) or 'mlp'. Forwarded to
                   discriminator.train_discriminator; cached separately per disc_type.
    """
    from experiments.zeroshot_cf.checkpoints import get_models
    from experiments.zeroshot_cf.data import get_actionable_immutable, load_dataset
    from experiments.zeroshot_cf.discriminator import train_discriminator
    from experiments.zeroshot_cf.sampler import ConditionalDensitySampler, build_chain_dag

    params = _DATASET_PARAMS.get(dataset_name, {"max_test": 50})
    if max_test is not None and max_test < 0:
        MAX_TEST = None  # full test split (no cap)
    elif max_test is not None:
        MAX_TEST = max_test
    else:
        MAX_TEST = params["max_test"]

    print(f"\n=== Experiment 2: {dataset_name.upper()} ===")
    print(f"  temperature={temperature}, n_permutations={n_permutations}, "
          f"max_context={max_context}, context_type={context_type}, "
          f"ordering={ordering}, actionable_set={actionable_set}")
    bundle = load_dataset(dataset_name)
    X_train = bundle.X_train
    y_train = bundle.y_train
    X_test = bundle.X_test[:MAX_TEST]
    y_test = bundle.y_test[:MAX_TEST]
    n = len(X_test)

    actionable_idx, immutable_idx = get_actionable_immutable(dataset_name, bundle)
    print(f"Features: {X_train.shape[1]} total, "
          f"{len(actionable_idx)} actionable, {len(immutable_idx)} immutable")
    print(f"Test set (capped): {n} points")

    disc_model = train_discriminator(
        X_train, y_train, X_test, y_test, dataset_name, disc_type=disc_type
    )

    # Determine target class for each test point: flip the predicted class
    y_pred = disc_model.predict(X_test)
    y_target = 1 - y_pred  # binary flip
    print(f"Target distribution: {np.bincount(y_target)}")

    # Determine which actionable features to mask (full or reduced)
    frozen_cols: List[int] = []
    if actionable_set == "reduced":
        if dataset_name == "moons":
            print("  [INFO] reduced==full for MOONS (only 2 actionable features). Using full.")
            mask_cols = list(actionable_idx)
        else:
            # Top-reduced_k actionable features by |LR discriminator coef|
            coef = disc_model._clf.coef_[0]
            act_array = np.array(actionable_idx)
            top_local = np.argsort(-np.abs(coef[act_array]))[:reduced_k]
            mask_cols = sorted(act_array[top_local].tolist())
            frozen_cols = sorted([i for i in actionable_idx if i not in mask_cols])
            print(f"  Reduced mask cols (top-{reduced_k} by |coef|): "
                  f"{[bundle.feature_names[i] for i in mask_cols]}")
            print(f"  Frozen non-immutable cols: "
                  f"{[bundle.feature_names[i] for i in frozen_cols]}")
    else:
        mask_cols = list(actionable_idx)

    # Build DAG if chain ordering requested (augmented index space: Y appended last)
    dag = None
    if ordering == "dag":
        y_idx = X_train.shape[1]  # Y appended at this position in augmented matrix
        dag = build_chain_dag(mask_cols, immutable_idx, y_idx)
        print(f"  DAG ordering: y_idx={y_idx}, immutable={immutable_idx}, "
              f"actionable chain={mask_cols}")

    print("Loading TabPFN models …")
    clf, reg = get_models(n_estimators=N_ESTIMATORS)

    # Build output array; initialize with factual values (immutables stay as-is)
    X_cf = X_test.copy()
    n_failed = 0

    # Process each target class in one batch
    for target_cls in np.unique(y_target):
        target_cls = int(target_cls)
        test_mask = y_target == target_cls
        test_idx = np.where(test_mask)[0]
        X_batch = X_test[test_mask]
        n_batch = len(X_batch)
        if n_batch == 0:
            continue

        print(f"\n  Target class {target_cls}: {n_batch} test points")
        sampler = ConditionalDensitySampler(
            clf=clf,
            reg=reg,
            append_target=True,
            n_permutations=n_permutations,
            temperature=temperature,
            random_state=base_seed + target_cls,
        )
        ctx_target = target_cls if context_type == "target_only" else None
        sampler.set_context(
            X_train,
            y_context=y_train,
            target_class=ctx_target,
            max_context=max_context,
        )

        X_cf_batch, failed = robust_impute(
            sampler,
            X_batch,
            mask_cols=mask_cols,
            fixed_target=target_cls,
            dag=dag,
        )
        X_cf[test_idx] = X_cf_batch
        n_failed += len(failed)

    if n_failed:
        print(f"\n  [robust] {n_failed}/{len(X_test)} rows failed imputation "
              f"(left as factual -> counted invalid).")

    return X_test, y_test, X_cf, {
        "n_failed": n_failed,
        "bundle": bundle,
        "y_pred": y_pred,
        "y_target": y_target,
        "actionable_idx": actionable_idx,
        "immutable_idx": immutable_idx,
        "mask_cols": mask_cols,
        "frozen_cols": frozen_cols,
        "disc_model": disc_model,
        "temperature": temperature,
        "n_permutations": n_permutations,
        "max_context": max_context,
        "context_type": context_type,
        "ordering": ordering,
        "actionable_set": actionable_set,
    }


def evaluate_and_report(
    dataset_name: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_cf: np.ndarray,
    info: Dict,
    write_csv: bool = True,
) -> Dict[str, float]:
    from experiments.zeroshot_cf.metrics_harness import compute_metrics, print_metrics
    from experiments.zeroshot_cf.data import load_dataset

    bundle = info["bundle"]
    immutable_idx = info["immutable_idx"]
    frozen_cols = info.get("frozen_cols", [])
    disc_model = info["disc_model"]
    y_target = info["y_target"]

    # --- Out-of-bounds fraction (before clipping) ---
    oob_mask = (X_cf < 0.0) | (X_cf > 1.0)
    frac_oob = float(oob_mask.any(axis=1).mean())
    print(f"\n  Out-of-[0,1] fraction of CFs: {frac_oob:.3f} "
          f"({int(oob_mask.any(axis=1).sum())}/{len(X_cf)} points)")

    # Clip to [0,1] — features are MinMax-scaled; out-of-range values are
    # TabPFN extrapolation artefacts. Document the clip fraction above.
    X_cf_clipped = np.clip(X_cf, 0.0, 1.0)

    # --- Immutable + frozen column check (must be exactly preserved by construction) ---
    check_cols = list(immutable_idx) + list(frozen_cols)
    if check_cols:
        cols = np.asarray(check_cols)
        max_dev = float(np.abs(X_cf[:, cols] - X_test[:, cols]).max())
        print(f"  Max immutable/frozen-column deviation: {max_dev:.2e} "
              f"({len(immutable_idx)} immutable, {len(frozen_cols)} frozen)")
        assert max_dev < 1e-9, (
            f"Immutable/frozen columns drifted: max_dev={max_dev} — "
            "these features must be preserved exactly by construction"
        )

    # --- Compute metrics ---
    metrics = compute_metrics(
        disc_model=disc_model,
        X_cf=X_cf_clipped,
        X_test=X_test,
        X_train=bundle.X_train,
        y_test=y_test,
        y_target=y_target,
        immutable_idx=immutable_idx,
        X_cf_lof=X_cf,  # use unclipped array for LOF to preserve true geometry
    )
    metrics["frac_oob"] = frac_oob
    metrics["n_failed"] = int(info.get("n_failed", 0))
    print_metrics(metrics, prefix=dataset_name)

    if write_csv:
        # --- Write per-dataset CSV ---
        csv_path = RESULTS_DIR / f"exp2_{dataset_name}_metrics.csv"
        row = {"dataset": dataset_name, **metrics}
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)
        print(f"\n  Wrote {csv_path}")

    return metrics


def write_examples(
    dataset_name: str,
    X_test: np.ndarray,
    X_cf: np.ndarray,
    info: Dict,
    n_examples: int = 5,
) -> None:
    """Write human-readable CF examples in original feature space."""
    bundle = info["bundle"]
    feat_names = bundle.feature_names
    y_pred = info["y_pred"]
    y_target = info["y_target"]
    disc_model = info["disc_model"]

    X_cf_clipped = np.clip(X_cf, 0.0, 1.0)
    y_cf_pred = disc_model.predict(X_cf_clipped)

    X_test_orig = bundle.inverse_transform(X_test)
    X_cf_orig = bundle.inverse_transform(X_cf_clipped)

    temperature = info.get("temperature", TEMPERATURE)
    n_permutations = info.get("n_permutations", N_PERMUTATIONS)
    max_context = info.get("max_context", MAX_CONTEXT)

    lines = [
        f"# Experiment 2: CF Examples — {dataset_name.upper()}",
        "",
        f"Temperature: {temperature}, n_permutations: {n_permutations}, "
        f"max_context: {max_context}",
        "",
    ]

    # Pick first n_examples where the CF is valid (predicted class == y_target)
    valid_mask = y_cf_pred == y_target
    valid_idxs = np.where(valid_mask)[0][:n_examples]
    invalid_idxs = np.where(~valid_mask)[0][:max(0, n_examples - len(valid_idxs))]
    idxs = np.concatenate([valid_idxs, invalid_idxs])

    for rank, i in enumerate(idxs[:n_examples]):
        status = "VALID" if valid_mask[i] else "INVALID"
        lines += [
            f"## Example {rank + 1} (idx={i}, {status})",
            f"Factual class: {y_pred[i]}, CF target: {y_target[i]}, "
            f"CF predicted: {y_cf_pred[i]}",
            "",
            f"| Feature | Factual | Counterfactual | Delta |",
            f"|---------|---------|---------------|-------|",
        ]
        for j, fname in enumerate(feat_names):
            fval = X_test_orig[i, j]
            cfval = X_cf_orig[i, j]
            delta = cfval - fval
            changed = "*" if abs(delta) > 1e-6 else ""
            lines.append(f"| {fname} | {fval:.4g} | {cfval:.4g} | {delta:+.4g} {changed} |")
        lines += ["", "---", ""]

    out_path = RESULTS_DIR / "exp2_examples.md"
    with open(out_path, "a") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Wrote examples to {out_path}")


def run_dataset(
    dataset_name: str,
    temperature: float = TEMPERATURE,
    n_permutations: int = N_PERMUTATIONS,
    max_context: int = MAX_CONTEXT,
    context_type: str = "target_only",
    ordering: str = "random",
    actionable_set: str = "full",
    reduced_k: int = 6,
    max_test: Optional[int] = None,
) -> Dict[str, float]:
    X_test, y_test, X_cf, info = generate_counterfactuals(
        dataset_name,
        temperature=temperature,
        n_permutations=n_permutations,
        max_context=max_context,
        context_type=context_type,
        ordering=ordering,
        actionable_set=actionable_set,
        reduced_k=reduced_k,
        max_test=max_test,
    )
    metrics = evaluate_and_report(dataset_name, X_test, y_test, X_cf, info)
    write_examples(dataset_name, X_test, X_cf, info)
    return metrics


def write_summary(all_metrics: List[Dict], temperature: float = TEMPERATURE,
                  n_permutations: int = N_PERMUTATIONS, max_context: int = MAX_CONTEXT,
                  context_type: str = "target_only", ordering: str = "random",
                  actionable_set: str = "full") -> None:
    lines = [
        "# Experiment 2: Counterfactual Generation — Summary",
        "",
        f"Settings: temperature={temperature}, n_permutations={n_permutations}, "
        f"max_context={max_context}, context_type={context_type}, "
        f"ordering={ordering}, actionable_set={actionable_set}",
        "",
        "## Metrics",
        "",
        "| Dataset | Validity | LOF | Sparsity | True-action | Proximity L2 | OOB frac |",
        "|---------|---------|-----|---------|------------|-------------|---------|",
    ]
    for m in all_metrics:
        lines.append(
            f"| {m['dataset']} "
            f"| {m.get('validity', float('nan')):.3f} "
            f"| {m.get('lof_scores_cf', float('nan')):.3f} "
            f"| {m.get('sparsity', float('nan')):.3f} "
            f"| {m.get('true_actionability', float('nan')):.3f} "
            f"| {m.get('proximity_l2_jaccard', float('nan')):.4f} "
            f"| {m.get('frac_oob', float('nan')):.3f} |"
        )
    lines += [
        "",
        "## Notes",
        "",
        "- `validity`: fraction of CFs whose discriminator-predicted class differs from factual (higher = better).",
        "- `lof_scores_cf`: mean negative-LOF plausibility score on X_cf (lower = more plausible).",
        "- `sparsity`: mean fraction of feature entries changed (lower = sparser).",
        "- `true_actionability`: fraction of CFs where immutable columns are exactly preserved (must be 1.0 by construction).",
        "- `proximity_l2_jaccard`: mean per-instance L2 distance on *valid* CFs (lower = closer to factual).",
        "- `frac_oob`: fraction of CF rows with at least one feature outside [0,1] before clipping.",
        "",
        "## Interpretation",
        "",
        "- CFs generated by conditioning on the target class via the Y-as-column trick.",
        "- Immutable features frozen by construction; only actionable features are imputed.",
        "- Out-of-range values are TabPFN extrapolation artefacts clipped to [0,1] before metric computation.",
        "- Low validity is expected for out-of-the-box generation; Stage 6 explores refinement levers.",
    ]
    summary_path = RESULTS_DIR / "exp2_summary.md"
    summary_path.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {summary_path}")


def main() -> None:
    from experiments.zeroshot_cf.local_data import LOCAL_DATASET_NAMES

    parser = argparse.ArgumentParser(description="Experiment 2: counterfactual generation")
    parser.add_argument(
        "--dataset",
        choices=["moons", "heloc", "all", *sorted(LOCAL_DATASET_NAMES)],
        default="moons",
        help="Dataset to run (default: moons). 'all' runs moons+heloc; local "
             "(CETGFN-ported) datasets run individually.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=TEMPERATURE,
        help=f"Sampling temperature (default: {TEMPERATURE}). Use 0 or 1e-9 for near-MAP.",
    )
    parser.add_argument(
        "--context-type",
        choices=["target_only", "all_classes"],
        default="target_only",
        help="Context selection strategy: 'target_only' filters context to the target class "
             "(default); 'all_classes' uses the full training set.",
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=N_PERMUTATIONS,
        help=f"Number of imputation permutations (default: {N_PERMUTATIONS}).",
    )
    parser.add_argument(
        "--max-context",
        type=int,
        default=MAX_CONTEXT,
        help=f"Max context rows passed to TabPFNUnsupervisedModel (default: {MAX_CONTEXT}).",
    )
    parser.add_argument(
        "--ordering",
        choices=["random", "dag"],
        default="random",
        help="Feature ordering: 'random' uses random permutations (default); 'dag' uses "
             "an explicit chain DAG (Y → immutables → actionable features in index order).",
    )
    parser.add_argument(
        "--actionable-set",
        choices=["full", "reduced"],
        default="full",
        help="Actionable feature set: 'full' masks all actionable features (default); "
             "'reduced' masks only the top-reduced_k features by |LR coef| (HELOC only).",
    )
    parser.add_argument(
        "--reduced-k",
        type=int,
        default=6,
        help="Number of top actionable features to mask when --actionable-set=reduced (default: 6).",
    )
    parser.add_argument(
        "--max-test",
        type=int,
        default=None,
        help="Number of test points to evaluate. Default: per-dataset cap "
             "(moons=100, heloc=50). Use -1 for the FULL stratified test split "
             "(moons=200, heloc=2092).",
    )
    args = parser.parse_args()
    datasets = ["moons", "heloc"] if args.dataset == "all" else [args.dataset]

    # Clear examples file before appending
    examples_path = RESULTS_DIR / "exp2_examples.md"
    if examples_path.exists() and args.dataset in ("all", datasets[0]):
        examples_path.unlink()

    all_metrics = []
    for ds in datasets:
        m = run_dataset(
            ds,
            temperature=args.temperature,
            n_permutations=args.n_permutations,
            max_context=args.max_context,
            context_type=args.context_type,
            ordering=args.ordering,
            actionable_set=args.actionable_set,
            reduced_k=args.reduced_k,
            max_test=args.max_test,
        )
        all_metrics.append({"dataset": ds, **m})

    write_summary(
        all_metrics,
        temperature=args.temperature,
        n_permutations=args.n_permutations,
        max_context=args.max_context,
        context_type=args.context_type,
        ordering=args.ordering,
        actionable_set=args.actionable_set,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
