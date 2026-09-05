"""Pretrain and disk-cache the shared classifier checkpoint for every dataset.

Every benchmark cell for a given dataset -- whichever counterfactual method or
seed produced it -- must be scored against the exact same classifier ("target
model"). ``train_discriminator`` (discriminator.py) already caches its fitted
classifier to ``$ZEROSHOT_CF_MODELS_DIR/disc_<dataset>_lr.pkl`` and reuses it
on later calls, but concurrent Slurm array tasks that all miss the cache at
the same time would race to train and overwrite that file.

Running this script once, synchronously, before an array is submitted removes
that race: every task afterwards only ever reads the checkpoint this script
wrote, so every method and every seed is guaranteed to share the identical
classifier.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml
from experiments.zeroshot_cf.evaluation import EvaluationSpec
from experiments.zeroshot_cf.orchestration.runner import _default_case_loader
from experiments.zeroshot_cf.orchestration.spec import (
    DatasetSpec,
    MethodSpec,
    ProtocolSpec,
    RunSpec,
    TargetModelSpec,
)

_DEFAULT_CONFIG = (
    Path(__file__).resolve().parent
    / "configs"
    / "matrices"
    / "full_reference_3seeds.yaml"
)


def _dataset_name(entry: Any) -> str:
    return entry if isinstance(entry, str) else str(entry["name"])


def warm(dataset: str, *, max_test: int | None) -> None:
    spec = RunSpec(
        dataset=DatasetSpec(dataset),
        protocol=ProtocolSpec(max_test=max_test, test_selection="stratified"),
        target_model=TargetModelSpec(
            name="retained_logistic_regression",
            params={"C": 1.0, "max_iter": 1000, "seed": 42},
        ),
        # The case loader ignores the method entirely -- only the dataset and
        # target_model determine the cached classifier -- so any registered
        # method name is a valid, unused placeholder here.
        method=MethodSpec(name="nice"),
        evaluation=EvaluationSpec(),
        seed=42,
    )
    loaded = _default_case_loader(spec)
    fingerprint = loaded.case.protocol["target_model_fingerprint"]
    print(f"[warm_classifier_cache] {dataset}: checkpoint fingerprint {fingerprint}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(_DEFAULT_CONFIG),
        help="matrix config to read the dataset list from",
    )
    parser.add_argument("--max-test", type=int, default=1000)
    args = parser.parse_args()

    payload = yaml.safe_load(Path(args.config).read_text())
    datasets = sorted({_dataset_name(entry) for entry in payload["datasets"]})
    for dataset in datasets:
        warm(dataset, max_test=args.max_test)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
