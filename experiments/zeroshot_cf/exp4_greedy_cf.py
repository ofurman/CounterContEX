"""Experiment 4: Iterative greedy counterfactual generation.

For each factual test point x with predicted class c:
  - Set target = 1 - c (binary class flip).
  - Run the iterative greedy loop (greedy.greedy_counterfactual): change one
    actionable feature at a time, conditioned class-conditionally on all the
    rest, and stop at the discriminator's flip.
  - Assemble the full CF matrix and evaluate the metric suite, plus greedy-
    specific L0 / steps / failure-rate keys.

Context fitting is batched by target class (like exp2). For ``prob_ascent`` the
context is the target class (``target_only``); for ``class_divergence`` it is the
full training set (``all_classes``) because the class-divergence selector needs a
non-constant Y to contrast Y=target vs Y=current.

Outputs (under experiments/zeroshot_cf/results/):
  results/exp4_greedy_<dataset>_metrics.csv  — per-dataset metric row
  results/exp4_examples.md                   — factual vs CF + recourse path

Usage:
  uv run python experiments/zeroshot_cf/exp4_greedy_cf.py --dataset moons --selector prob_ascent
  uv run python experiments/zeroshot_cf/exp4_greedy_cf.py --dataset heloc --selector prob_ascent --max-test 50
  uv run python experiments/zeroshot_cf/exp4_greedy_cf.py --dataset moons --selector class_divergence
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

MAX_CONTEXT = 256
N_PERMUTATIONS = 3
TEMPERATURE = 1e-9  # near-MAP committed value (deterministic single-column commit)
N_ESTIMATORS = 4
TAU = 0.5

_DATASET_PARAMS = {
    "moons": {"max_test": 100},
    "heloc": {"max_test": 50},
}


def generate_counterfactuals(
    dataset_name: str,
    selector: str = "prob_ascent",
    tau: float = TAU,
    budget: Optional[int] = None,
    temperature: float = TEMPERATURE,
    n_permutations: int = N_PERMUTATIONS,
    max_context: int = MAX_CONTEXT,
    max_test: Optional[int] = None,
    base_seed: int = 42,
    disc_type: str = "lr",
    allow_repeats: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Generate greedy CFs for one dataset. Returns (X_test, y_test, X_cf, info).

    Args:
        base_seed: Base random_state for the per-target-class sampler (default:
            42, matching the previously-hardcoded value). At the default
            near-MAP temperature (1e-9) the committed values are already close
            to deterministic, so varying this mainly changes context subsampling,
            not the generated values.
        disc_type: Validity-oracle classifier — 'lr' (default) or 'mlp'. Forwarded
            to discriminator.train_discriminator; cached separately per disc_type.
        allow_repeats: If True, a feature can be re-picked and re-committed at
            a later step instead of being excluded once changed (see
            greedy.greedy_counterfactual's docstring). Pair with a larger
            ``budget`` (e.g. 2x actionable count) — otherwise the loop still
            stops after the same number of steps and repeats have no room to
            occur.
    """
    from experiments.zeroshot_cf.checkpoints import get_models
    from experiments.zeroshot_cf.data import get_actionable_immutable, load_dataset
    from experiments.zeroshot_cf.discriminator import train_discriminator
    from experiments.zeroshot_cf.greedy import greedy_counterfactual
    from experiments.zeroshot_cf.sampler import ConditionalDensitySampler

    params = _DATASET_PARAMS.get(dataset_name, {"max_test": 50})
    if max_test is not None and max_test < 0:
        MAX_TEST = None
    elif max_test is not None:
        MAX_TEST = max_test
    else:
        MAX_TEST = params["max_test"]

    # Strategy 2 (class_divergence) requires a both-classes context pool.
    if selector == "class_divergence":
        context_type = "all_classes"
    else:
        context_type = "target_only"

    print(f"\n=== Experiment 4 (greedy): {dataset_name.upper()} ===")
    print(f"  selector={selector}, context_type={context_type}, tau={tau}, "
          f"budget={budget}, temperature={temperature}, "
          f"n_permutations={n_permutations}, max_context={max_context}")

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

    y_pred = disc_model.predict(X_test)
    y_target = 1 - y_pred
    print(f"Target distribution: {np.bincount(y_target)}")

    print("Loading TabPFN models …")
    clf, reg = get_models(n_estimators=N_ESTIMATORS)

    eff_budget = budget if budget is not None else len(actionable_idx)

    X_cf = X_test.copy()
    changed_per_point: List[List[int]] = [[] for _ in range(n)]
    flipped_per_point: List[bool] = [False] * n
    steps_per_point: List[int] = [0] * n
    l0_per_point: List[int] = [0] * n

    # Batch by target class so context is fit at most twice.
    for target_cls in np.unique(y_target):
        target_cls = int(target_cls)
        test_mask = y_target == target_cls
        test_idx = np.where(test_mask)[0]
        n_batch = len(test_idx)
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
            X_train, y_context=y_train, target_class=ctx_target, max_context=max_context
        )

        t0 = time.perf_counter()
        for k, i in enumerate(test_idx):
            x_cf, changed, gi = greedy_counterfactual(
                sampler,
                disc_model,
                X_test[i],
                target_cls,
                actionable_idx,
                selector,
                tau=tau,
                budget=eff_budget,
                temperature=temperature,
                allow_repeats=allow_repeats,
            )
            X_cf[i] = x_cf
            changed_per_point[i] = changed
            flipped_per_point[i] = gi["flipped"]
            steps_per_point[i] = gi["steps"]
            l0_per_point[i] = gi["l0"]
            if k == 0:
                per_pt = time.perf_counter() - t0
                print(f"    [timing] first point: {per_pt:.1f}s "
                      f"(~{per_pt * n_batch / 60:.1f} min est. for this batch)")

    return X_test, y_test, X_cf, {
        "bundle": bundle,
        "y_pred": y_pred,
        "y_target": y_target,
        "actionable_idx": actionable_idx,
        "immutable_idx": immutable_idx,
        "disc_model": disc_model,
        "selector": selector,
        "context_type": context_type,
        "tau": tau,
        "budget": eff_budget,
        "temperature": temperature,
        "n_permutations": n_permutations,
        "max_context": max_context,
        "allow_repeats": allow_repeats,
        "changed_per_point": changed_per_point,
        "flipped_per_point": flipped_per_point,
        "steps_per_point": steps_per_point,
        "l0_per_point": l0_per_point,
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

    bundle = info["bundle"]
    immutable_idx = info["immutable_idx"]
    disc_model = info["disc_model"]
    y_target = info["y_target"]

    # --- Out-of-bounds fraction (before clipping) — same recipe as exp2 ---
    oob_mask = (X_cf < 0.0) | (X_cf > 1.0)
    frac_oob = float(oob_mask.any(axis=1).mean())
    print(f"\n  Out-of-[0,1] fraction of CFs: {frac_oob:.3f} "
          f"({int(oob_mask.any(axis=1).sum())}/{len(X_cf)} points)")

    X_cf_clipped = np.clip(X_cf, 0.0, 1.0)

    # --- Immutable column check (preserved exactly by construction) ---
    if immutable_idx:
        cols = np.asarray(immutable_idx)
        max_dev = float(np.abs(X_cf[:, cols] - X_test[:, cols]).max())
        print(f"  Max immutable-column deviation: {max_dev:.2e}")
        assert max_dev < 1e-9, (
            f"Immutable columns drifted: max_dev={max_dev} — must be preserved exactly."
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

    # --- Greedy-specific keys ---
    # l0 = distinct features changed (correct even with allow_repeats=True, where a
    # feature can be re-picked, so len(changed_per_point[i]) alone would overcount).
    flipped = np.asarray(info["flipped_per_point"], dtype=bool)
    l0_all = np.asarray(info["l0_per_point"], dtype=float)
    steps_all = np.asarray(info["steps_per_point"], dtype=float)

    # L0 / steps reported over VALID (flipped) CFs (matches the success criterion);
    # failure_rate over all points.
    if flipped.any():
        l0_valid = l0_all[flipped]
        steps_valid = steps_all[flipped]
        metrics["l0_count_mean"] = float(l0_valid.mean())
        metrics["l0_count_median"] = float(np.median(l0_valid))
        metrics["l0_count_max"] = float(l0_valid.max())
        metrics["steps_mean"] = float(steps_valid.mean())
        metrics["steps_median"] = float(np.median(steps_valid))
        metrics["steps_max"] = float(steps_valid.max())
    else:
        for key in ("l0_count_mean", "l0_count_median", "l0_count_max",
                    "steps_mean", "steps_median", "steps_max"):
            metrics[key] = float("nan")
    metrics["failure_rate"] = float((~flipped).mean())
    metrics["n_actionable"] = int(len(info["actionable_idx"]))

    print_metrics(metrics, prefix=dataset_name)

    if write_csv:
        csv_path = RESULTS_DIR / f"exp4_greedy_{dataset_name}_metrics.csv"
        row = {"dataset": dataset_name, "selector": info["selector"], **metrics}
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
    """Write human-readable greedy CF examples in original feature space, with
    the ordered list of changed features (the recourse path)."""
    bundle = info["bundle"]
    feat_names = bundle.feature_names
    y_pred = info["y_pred"]
    y_target = info["y_target"]
    disc_model = info["disc_model"]
    changed_per_point = info["changed_per_point"]
    flipped_per_point = info["flipped_per_point"]

    X_cf_clipped = np.clip(X_cf, 0.0, 1.0)
    y_cf_pred = disc_model.predict(X_cf_clipped)

    X_test_orig = bundle.inverse_transform(X_test)
    X_cf_orig = bundle.inverse_transform(X_cf_clipped)

    lines = [
        f"# Experiment 4 (greedy): CF Examples — {dataset_name.upper()}",
        "",
        f"Selector: {info['selector']}, context: {info['context_type']}, "
        f"tau: {info['tau']}, temperature: {info['temperature']}, "
        f"n_permutations: {info['n_permutations']}, max_context: {info['max_context']}",
        "",
    ]

    valid_mask = np.asarray(flipped_per_point, dtype=bool)
    valid_idxs = np.where(valid_mask)[0][:n_examples]
    invalid_idxs = np.where(~valid_mask)[0][:max(0, n_examples - len(valid_idxs))]
    idxs = np.concatenate([valid_idxs, invalid_idxs]).astype(int)

    for rank, i in enumerate(idxs[:n_examples]):
        i = int(i)
        status = "VALID (flipped)" if valid_mask[i] else "INVALID (budget exhausted)"
        recourse = changed_per_point[i]
        recourse_names = [feat_names[j] for j in recourse]
        lines += [
            f"## Example {rank + 1} (idx={i}, {status})",
            f"Factual class: {y_pred[i]}, CF target: {y_target[i]}, "
            f"CF predicted: {y_cf_pred[i]}",
            f"L0 (features changed): {len(recourse)}",
            f"Recourse path (ordered): {recourse_names}",
            "",
            "| Feature | Factual | Counterfactual | Delta |",
            "|---------|---------|---------------|-------|",
        ]
        for j, fname in enumerate(feat_names):
            fval = X_test_orig[i, j]
            cfval = X_cf_orig[i, j]
            delta = cfval - fval
            changed = "*" if abs(delta) > 1e-6 else ""
            lines.append(f"| {fname} | {fval:.4g} | {cfval:.4g} | {delta:+.4g} {changed} |")
        lines += ["", "---", ""]

    out_path = RESULTS_DIR / "exp4_examples.md"
    with open(out_path, "a") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Wrote examples to {out_path}")


def run_dataset(
    dataset_name: str,
    selector: str = "prob_ascent",
    tau: float = TAU,
    budget: Optional[int] = None,
    temperature: float = TEMPERATURE,
    n_permutations: int = N_PERMUTATIONS,
    max_context: int = MAX_CONTEXT,
    max_test: Optional[int] = None,
    allow_repeats: bool = False,
) -> Dict[str, float]:
    X_test, y_test, X_cf, info = generate_counterfactuals(
        dataset_name,
        selector=selector,
        tau=tau,
        budget=budget,
        temperature=temperature,
        n_permutations=n_permutations,
        max_context=max_context,
        max_test=max_test,
        allow_repeats=allow_repeats,
    )
    metrics = evaluate_and_report(dataset_name, X_test, y_test, X_cf, info)
    write_examples(dataset_name, X_test, X_cf, info)
    return metrics


def main() -> None:
    from experiments.zeroshot_cf.local_data import LOCAL_DATASET_NAMES

    parser = argparse.ArgumentParser(description="Experiment 4: iterative greedy CF")
    parser.add_argument(
        "--dataset",
        choices=["moons", "heloc", "all", *sorted(LOCAL_DATASET_NAMES)],
        default="moons",
        help="'all' runs moons+heloc; local (CETGFN-ported) datasets run individually.",
    )
    parser.add_argument(
        "--selector",
        choices=["prob_ascent", "class_divergence"],
        default="prob_ascent",
        help="Candidate-selection strategy (default: prob_ascent).",
    )
    parser.add_argument("--tau", type=float, default=TAU,
                        help=f"Flip probability threshold (default: {TAU} ≡ hard flip).")
    parser.add_argument("--budget", type=int, default=None,
                        help="Max features to change (default: |actionable|).")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE,
                        help=f"Committed-value temperature (default: {TEMPERATURE} ≈ MAP).")
    parser.add_argument("--n-permutations", type=int, default=N_PERMUTATIONS,
                        help=f"Imputation permutations (default: {N_PERMUTATIONS}).")
    parser.add_argument("--max-context", type=int, default=MAX_CONTEXT,
                        help=f"Max context rows (default: {MAX_CONTEXT}).")
    parser.add_argument("--max-test", type=int, default=None,
                        help="Number of test points (default: moons=100, heloc=50; "
                             "-1 for full split).")
    parser.add_argument(
        "--allow-repeats",
        action="store_true",
        help="Let a feature be re-picked and re-committed at a later step "
             "instead of being excluded once changed. Pair with a larger "
             "--budget (e.g. 2x actionable count) or the loop still stops "
             "after the same number of steps.",
    )
    args = parser.parse_args()

    datasets = ["moons", "heloc"] if args.dataset == "all" else [args.dataset]

    # Clear examples file before appending.
    examples_path = RESULTS_DIR / "exp4_examples.md"
    if examples_path.exists() and args.dataset in ("all", datasets[0]):
        examples_path.unlink()

    for ds in datasets:
        run_dataset(
            ds,
            selector=args.selector,
            tau=args.tau,
            budget=args.budget,
            temperature=args.temperature,
            n_permutations=args.n_permutations,
            max_context=args.max_context,
            max_test=args.max_test,
            allow_repeats=args.allow_repeats,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
