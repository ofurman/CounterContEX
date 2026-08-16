"""Validation-calibrated row plausibility from a TabICL classifier."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from experiments.zeroshot_cf.tabicl_checkpoints import require_checkpoints
from sklearn.model_selection import train_test_split
from tabicl import TabICLClassifier


def one_vs_rest_logit(probabilities: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Return the target-class log-odds represented by softmax probabilities.

    TabICL does not expose raw classifier logits through its public API. The
    one-vs-rest logit is available from ``predict_proba`` and preserves the
    ordering of the target-class logit margin. Quantile calibration is invariant
    to this monotone transformation.
    """
    matrix = np.asarray(probabilities, dtype=np.float64)
    target_positions = np.asarray(positions, dtype=int)
    if matrix.ndim != 2:
        raise ValueError(f"probabilities must be 2D, got shape {matrix.shape}")
    if target_positions.shape != (len(matrix),):
        raise ValueError("positions must contain one class index per row")
    if np.any((target_positions < 0) | (target_positions >= matrix.shape[1])):
        raise ValueError("positions contain an invalid class index")
    target_probability = matrix[np.arange(len(matrix)), target_positions]
    eps = np.finfo(np.float64).eps
    target_probability = np.clip(target_probability, eps, 1.0 - eps)
    return np.log(target_probability) - np.log1p(-target_probability)


def _stratified_context_indices(
    labels: np.ndarray,
    context_size: int,
    random_state: int,
) -> np.ndarray:
    """Select one fixed class-stratified TabICL reference context."""
    y = np.asarray(labels)
    if y.ndim != 1 or len(y) == 0:
        raise ValueError("labels must be a non-empty 1D array")
    if context_size < len(np.unique(y)):
        raise ValueError("context_size must be at least the number of classes")
    if context_size >= len(y):
        return np.arange(len(y))
    selected, _ = train_test_split(
        np.arange(len(y)),
        train_size=context_size,
        random_state=random_state,
        stratify=y,
    )
    return np.sort(selected)


@dataclass
class ValidationCalibratedTabICLRowScorer:
    """Score complete rows with a fixed-context TabICL class-logit margin.

    A single TabICL classifier is fitted in context on a stratified subset of
    the training rows. For every class, the lower-tail threshold is calibrated
    from validation rows assigned to that class by the labeling model used by
    the counterfactual generator. Candidate rows are evaluated in batches.
    """

    estimator: Any
    thresholds: dict[int, float]
    calibration_counts: dict[int, int]
    calibration_scores: dict[int, np.ndarray]
    threshold_quantile: float
    context_size: int
    prediction_batch_size: int = 512

    @classmethod
    def fit(
        cls,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_validation: np.ndarray,
        y_validation: np.ndarray,
        *,
        cache_dir: Path | None,
        device: str,
        n_estimators: int,
        context_size: int = 512,
        threshold_quantile: float = 0.10,
        prediction_batch_size: int = 512,
        random_state: int = 42,
    ) -> "ValidationCalibratedTabICLRowScorer":
        """Fit the fixed context and calibrate one threshold per class."""
        if not 0.0 < threshold_quantile < 1.0:
            raise ValueError("threshold_quantile must be in (0, 1)")
        if prediction_batch_size < 1:
            raise ValueError("prediction_batch_size must be positive")
        train = np.asarray(X_train)
        train_labels = np.asarray(y_train)
        validation = np.asarray(X_validation)
        validation_labels = np.asarray(y_validation)
        if train.ndim != 2 or validation.ndim != 2:
            raise ValueError("training and validation matrices must be 2D")
        if train.shape[1] != validation.shape[1]:
            raise ValueError("training and validation feature counts differ")
        if train_labels.shape != (len(train),):
            raise ValueError("y_train must contain one label per training row")
        if validation_labels.shape != (len(validation),):
            raise ValueError("y_validation must contain one label per validation row")

        selected = _stratified_context_indices(
            train_labels,
            context_size,
            random_state,
        )
        classifier_path, _ = require_checkpoints(cache_dir)
        estimator = TabICLClassifier(
            n_estimators=n_estimators,
            model_path=classifier_path,
            allow_auto_download=False,
            device=device,
            kv_cache=True,
            random_state=random_state,
        )
        estimator.fit(train[selected], train_labels[selected])
        scorer = cls(
            estimator=estimator,
            thresholds={},
            calibration_counts={},
            calibration_scores={},
            threshold_quantile=threshold_quantile,
            context_size=len(selected),
            prediction_batch_size=prediction_batch_size,
        )
        validation_scores = scorer.score_targets(validation, validation_labels)
        for target_class in np.asarray(estimator.classes_):
            class_key = int(target_class)
            class_scores = validation_scores[validation_labels == target_class]
            if len(class_scores) == 0:
                raise ValueError(
                    f"validation data contain no rows for class {class_key}"
                )
            scorer.thresholds[class_key] = float(
                np.quantile(class_scores, threshold_quantile)
            )
            scorer.calibration_counts[class_key] = int(len(class_scores))
            scorer.calibration_scores[class_key] = np.sort(class_scores)
        return scorer

    @property
    def classes_(self) -> np.ndarray:
        return np.asarray(self.estimator.classes_)

    def _predict_proba(self, rows: np.ndarray) -> np.ndarray:
        matrix = np.atleast_2d(np.asarray(rows))
        chunks = [
            np.asarray(self.estimator.predict_proba(matrix[start:stop]))
            for start in range(0, len(matrix), self.prediction_batch_size)
            for stop in [min(start + self.prediction_batch_size, len(matrix))]
        ]
        return np.concatenate(chunks, axis=0)

    def _positions(self, targets: np.ndarray) -> np.ndarray:
        classes = self.classes_
        positions = np.empty(len(targets), dtype=int)
        for i, target in enumerate(targets):
            matches = np.flatnonzero(classes == target)
            if len(matches) != 1:
                raise ValueError(f"class {target!r} is absent from TabICL classes")
            positions[i] = int(matches[0])
        return positions

    def score_rows(self, rows: np.ndarray, target_class: int) -> np.ndarray:
        """Return row-level target-class logit margins for one class."""
        matrix = np.atleast_2d(np.asarray(rows))
        targets = np.full(len(matrix), target_class)
        probabilities = self._predict_proba(matrix)
        return one_vs_rest_logit(probabilities, self._positions(targets))

    def score_targets(self, rows: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Return row-level margins when each row may have a different target."""
        matrix = np.atleast_2d(np.asarray(rows))
        target_array = np.asarray(targets)
        if target_array.shape != (len(matrix),):
            raise ValueError("targets must contain one class per row")
        probabilities = self._predict_proba(matrix)
        return one_vs_rest_logit(probabilities, self._positions(target_array))

    def threshold(self, target_class: int) -> float:
        """Return the validation-calibrated lower-tail threshold for a class."""
        try:
            return self.thresholds[int(target_class)]
        except KeyError as exc:
            raise ValueError(
                f"class {target_class!r} has no calibrated threshold"
            ) from exc

    def percentile_rows(self, rows: np.ndarray, target_class: int) -> np.ndarray:
        """Map candidate margins to their target-class validation percentiles."""
        scores = self.score_rows(rows, target_class)
        try:
            calibration = self.calibration_scores[int(target_class)]
        except KeyError as exc:
            raise ValueError(
                f"class {target_class!r} has no calibration distribution"
            ) from exc
        return np.searchsorted(calibration, scores, side="right") / len(calibration)
