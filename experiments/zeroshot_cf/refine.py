"""Stage 6: Inference refinement sweep.

Runs a grid of inference configurations (temperature, n_permutations,
context strategy) for each dataset using the Experiment 2 CF pipeline.
No retraining — only inference knobs are varied.

Budget note:
  MOONS: ~2s per impute call → 6 configs × 2 target classes = 12 calls ≈ 24s
  HELOC: ~3.5 min per impute call (17 features × n_perm=3) → 3 configs ×
         2 target classes = 6 calls ≈ 21 min.
  Any skipped grid points are explicitly noted in the output.

Usage:
  uv run python experiments/zeroshot_cf/refine.py --dataset moons
  uv run python experiments/zeroshot_cf/refine.py --dataset heloc
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

CONFIGS_DIR = Path(__file__).parent / "configs"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MAX_CONTEXT_DEFAULT = 256
N_ESTIMATORS = 4


def run_config(
    dataset_name: str,
    cfg: Dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    actionable_idx: List[int],
    immutable_idx: List[int],
    disc_model,
    clf,
    reg,
) -> Dict:
    """Run one sweep configuration; return metrics dict."""
    from experiments.zeroshot_cf.metrics_harness import compute_metrics
    from experiments.zeroshot_cf.sampler import ConditionalDensitySampler

    temperature = float(cfg["temperature"])
    n_permutations = int(cfg["n_permutations"])
    context_type = cfg["context_type"]
    max_context = int(cfg.get("max_context", MAX_CONTEXT_DEFAULT))
    cfg_id = cfg["id"]

    print(f"\n  Config '{cfg_id}': t={temperature}, n_perm={n_permutations}, "
          f"ctx={context_type}, max_ctx={max_context}")

    y_pred = disc_model.predict(X_test)
    y_target = 1 - y_pred
    X_cf = X_test.copy()

    for target_cls in np.unique(y_target):
        target_cls = int(target_cls)
        test_mask = y_target == target_cls
        test_idx = np.where(test_mask)[0]
        X_batch = X_test[test_mask]
        if len(X_batch) == 0:
            continue

        sampler = ConditionalDensitySampler(
            clf=clf,
            reg=reg,
            append_target=True,
            n_permutations=n_permutations,
            temperature=temperature,
            random_state=42 + target_cls,
        )

        if context_type == "target_class_only":
            sampler.set_context(
                X_train, y_context=y_train,
                target_class=target_cls, max_context=max_context,
            )
        elif context_type == "all_classes":
            sampler.set_context(
                X_train, y_context=y_train,
                target_class=None, max_context=max_context,
            )
        else:
            raise ValueError(f"Unknown context_type: {context_type!r}")

        X_cf_batch = sampler.impute_masked(
            X_batch, mask_cols=actionable_idx, fixed_target=target_cls,
        )
        X_cf[test_idx] = X_cf_batch

    oob_mask = (X_cf < 0.0) | (X_cf > 1.0)
    frac_oob = float(oob_mask.any(axis=1).mean())
    X_cf_clipped = np.clip(X_cf, 0.0, 1.0)

    metrics = compute_metrics(
        disc_model=disc_model,
        X_cf=X_cf_clipped,
        X_test=X_test,
        X_train=X_train,
        y_test=y_test,
        y_target=y_target,
        immutable_idx=immutable_idx,
        X_cf_lof=X_cf,  # use unclipped array for LOF to preserve true geometry
    )
    metrics["frac_oob"] = frac_oob
    metrics["config_id"] = cfg_id
    metrics["dataset"] = dataset_name
    metrics["temperature"] = temperature
    metrics["n_permutations"] = n_permutations
    metrics["context_type"] = context_type
    metrics["max_context"] = max_context

    print(f"    validity={metrics['validity']:.3f}  "
          f"lof={metrics['lof_scores_cf']:.3f}  "
          f"oob={frac_oob:.3f}  "
          f"prox={metrics['proximity_l2_jaccard']:.4f}")

    return metrics


def run_sweep(dataset_name: str) -> List[Dict]:
    """Run all configs for a dataset; return list of metric dicts."""
    from experiments.zeroshot_cf.checkpoints import get_models
    from experiments.zeroshot_cf.data import get_actionable_immutable, load_dataset
    from experiments.zeroshot_cf.discriminator import train_discriminator

    sweep_cfg_path = CONFIGS_DIR / "sweep.yaml"
    with open(sweep_cfg_path) as f:
        sweep_cfg = yaml.safe_load(f)

    ds_cfg = sweep_cfg.get(dataset_name)
    if ds_cfg is None:
        print(f"No sweep config for dataset '{dataset_name}' — skipping.")
        return []

    MAX_TEST = int(ds_cfg.get("max_test", 50))
    configs = ds_cfg.get("configs", [])

    print(f"\n=== Refinement Sweep: {dataset_name.upper()} "
          f"({len(configs)} configs, max_test={MAX_TEST}) ===")

    bundle = load_dataset(dataset_name)
    X_train = bundle.X_train
    y_train = bundle.y_train
    X_test = bundle.X_test[:MAX_TEST]
    y_test = bundle.y_test[:MAX_TEST]

    actionable_idx, immutable_idx = get_actionable_immutable(dataset_name, bundle)
    disc_model = train_discriminator(X_train, y_train, X_test, y_test, dataset_name)

    print("Loading TabPFN models …")
    clf, reg = get_models(n_estimators=N_ESTIMATORS)

    all_metrics = []
    for i, cfg in enumerate(configs, 1):
        print(f"\n  [{i}/{len(configs)}] ", end="")
        metrics = run_config(
            dataset_name, cfg, X_train, y_train, X_test, y_test,
            actionable_idx, immutable_idx, disc_model, clf, reg,
        )
        all_metrics.append(metrics)

    # Write sweep CSV
    if all_metrics:
        sweep_path = RESULTS_DIR / f"exp2_sweep_{dataset_name}.csv"
        fieldnames = list(all_metrics[0].keys())
        with open(sweep_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_metrics)
        print(f"\nWrote {sweep_path}")

    return all_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 6: Inference refinement sweep")
    parser.add_argument(
        "--dataset", choices=["moons", "heloc", "all"], default="moons"
    )
    args = parser.parse_args()
    datasets = ["moons", "heloc"] if args.dataset == "all" else [args.dataset]

    for ds in datasets:
        run_sweep(ds)

    print("\nDone.")


if __name__ == "__main__":
    main()
