#  Copyright (c) Prior Labs GmbH 2026.
# ruff: noqa: T201

"""Probe target-conditioned TabICL completion for all-missing HELOC rows.

HELOC encodes a completely unavailable credit record as -9 in every feature;
after the experiment's MinMax scaling this becomes an all-zero row.  Such a
row has no observed individual attributes from which to construct an
actionability-preserving counterfactual.  This small diagnostic masks those
sentinel fields, draws complete target-conditioned profiles, and reports
whether any are both classifier-valid and locally plausible.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.neighbors import LocalOutlierFactor

from experiments.zeroshot_cf.data import load_dataset
from experiments.zeroshot_cf.discriminator import train_discriminator
from experiments.zeroshot_cf.exp8_tabicl_cf import empirical_confidence_grid
from experiments.zeroshot_cf.tabicl_checkpoints import TABICL_DEVICE
from experiments.zeroshot_cf.tabicl_sampler import TabICLConditionalDensitySampler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--n-estimators", type=int, default=1)
    parser.add_argument("--tabicl-cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    bundle = load_dataset("heloc")
    model = train_discriminator(
        bundle.X_train,
        bundle.y_train,
        bundle.X_test,
        bundle.y_test,
        "heloc",
    )
    factual = np.zeros(bundle.X_train.shape[1], dtype=np.float64)
    source_prediction = int(model.predict(factual[None])[0])
    target = 1 - source_prediction
    context_labels = model.predict(bundle.X_train)
    target_confidence = model.predict_proba(bundle.X_train)[:, target]

    sampler = TabICLConditionalDensitySampler(
        n_estimators=args.n_estimators,
        temperature=args.temperature,
        random_state=42,
        device=TABICL_DEVICE,
        cache_dir=args.tabicl_cache_dir,
        context_update="replace",
        numerical_point_estimate="mode",
    )
    sampler.set_context(
        bundle.X_train,
        y_context=context_labels,
        confidence_context=target_confidence,
        max_context=512,
        selection="knn",
        query=factual,
    )
    confidences = empirical_confidence_grid(
        sampler.selected_confidences_,
        sampler.selected_labels_,
        target,
        (0.1, 0.25, 0.5, 0.75, 0.9),
    )

    query_batch = np.repeat(factual[None], args.n_samples, axis=0)
    completions = []
    completion_confidences = []
    for confidence in confidences:
        completed = sampler.impute_masked(
            query_batch,
            range(bundle.X_train.shape[1]),
            fixed_target=target,
            fixed_confidence=confidence,
        )
        completions.append(np.clip(completed, 0.0, 1.0))
        completion_confidences.extend([confidence] * len(completed))
    candidates = np.concatenate(completions)

    probabilities = model.predict_proba(candidates)[:, target]
    predictions = model.predict(candidates)
    valid = (predictions == target) & (probabilities >= 0.5)
    lof_model = LocalOutlierFactor(n_neighbors=20, novelty=True).fit(bundle.X_train)
    lof = -lof_model.score_samples(candidates)
    eligible = np.flatnonzero(valid)
    best = None if len(eligible) == 0 else int(eligible[np.argmin(lof[eligible])])

    result = {
        "n_candidates": int(len(candidates)),
        "n_valid": int(valid.sum()),
        "validity_fraction": float(valid.mean()),
        "source_prediction": source_prediction,
        "target": target,
        "confidence_grid": list(confidences),
        "best_index": best,
        "best_lof": None if best is None else float(lof[best]),
        "best_target_probability": (
            None if best is None else float(probabilities[best])
        ),
        "best_confidence": (
            None if best is None else float(completion_confidences[best])
        ),
        "best_candidate": None if best is None else candidates[best].tolist(),
        "valid_lof_quantiles": (
            None
            if len(eligible) == 0
            else {
                str(q): float(np.quantile(lof[eligible], q))
                for q in (0.0, 0.25, 0.5, 0.75, 1.0)
            }
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
