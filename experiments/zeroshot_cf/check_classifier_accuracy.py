"""Report accuracy for the classifier checkpoint(s) shared by every cell.

Every benchmark cell for a dataset -- any method, any seed -- is scored
against the exact same cached classifier (see warm_classifier_cache.py and
orchestration/runner.py's default case loader). ``train_discriminator``
(discriminator.py) only prints accuracy the first time it trains a fresh
checkpoint; later cache hits are silent, and even that first printout is
measured against the *validation* split, not the true held-out test set.

This script loads each checkpoint via the same code path used at benchmark
time -- so it is guaranteed to be the identical classifier, never a
retrain -- and reports accuracy/precision/recall/F1/ROC-AUC on the train,
validation, and test splits.

Usage:
    uv run --project experiments/zeroshot_cf \\
        python -m experiments.zeroshot_cf.check_classifier_accuracy
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
from experiments.zeroshot_cf.evaluation import EvaluationSpec
from experiments.zeroshot_cf.orchestration.runner import _default_case_loader
from experiments.zeroshot_cf.orchestration.spec import (
    DatasetSpec,
    MethodSpec,
    ProtocolSpec,
    RunSpec,
    TargetModelSpec,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

DEFAULT_DATASETS = ("heloc", "bank_marketing", "give_me_some_credit", "lending_club")


def _metrics(oracle: Any, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    if len(X) == 0:
        return {}
    clf = getattr(oracle, "_clf", oracle)
    classes = np.asarray(clf.classes_)
    positive = classes[-1]
    predictions = oracle.predict(X)
    metrics = {
        "n": float(len(X)),
        "accuracy": accuracy_score(y, predictions),
        "precision": precision_score(
            y, predictions, pos_label=positive, zero_division=0
        ),
        "recall": recall_score(y, predictions, pos_label=positive, zero_division=0),
        "f1": f1_score(y, predictions, pos_label=positive, zero_division=0),
    }
    positive_index = int(np.where(classes == positive)[0][0])
    proba = oracle.predict_proba(X)[:, positive_index]
    metrics["roc_auc"] = roc_auc_score(y, proba)
    return metrics


def check(dataset: str, *, max_test: int) -> None:
    spec = RunSpec(
        dataset=DatasetSpec(dataset),
        protocol=ProtocolSpec(max_test=max_test, test_selection="stratified"),
        target_model=TargetModelSpec(
            name="retained_logistic_regression",
            params={"C": 1.0, "max_iter": 1000, "seed": 42},
        ),
        # Unused by the case loader -- only dataset and target_model matter
        # for which classifier gets loaded.
        method=MethodSpec(name="nice"),
        evaluation=EvaluationSpec(),
        seed=42,
    )
    loaded = _default_case_loader(spec)
    oracle = loaded.runtime_context["oracle"]
    prepared = loaded.case.dataset

    print(f"\n=== {dataset} ===")
    for split_name, X, y in (
        ("train", prepared.X_train, prepared.y_train),
        ("validation", prepared.X_validation, prepared.y_validation),
        ("test", prepared.X_test, prepared.y_test),
    ):
        metrics = _metrics(oracle, X, y)
        if not metrics:
            print(f"  {split_name:<10} (empty)")
            continue
        parts = " ".join(
            f"{key}={value:.4f}" for key, value in metrics.items() if key != "n"
        )
        print(f"  {split_name:<10} n={int(metrics['n']):<6} {parts}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        default=None,
        help="restrict to one dataset (repeatable); default: all four",
    )
    parser.add_argument("--max-test", type=int, default=1000)
    args = parser.parse_args()
    for dataset in args.datasets or list(DEFAULT_DATASETS):
        check(dataset, max_test=args.max_test)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
