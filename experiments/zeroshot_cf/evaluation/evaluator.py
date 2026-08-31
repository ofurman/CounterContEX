"""Method-blind evaluation over canonical generation results."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from experiments.zeroshot_cf.core.contracts import BenchmarkCase, GenerationResult
from experiments.zeroshot_cf.core.validation import target_probabilities
from experiments.zeroshot_cf.evaluation.metrics import (
    common_candidate_metrics,
    evaluate_diverse_candidate_sets,
    prepare_novelty_models,
)
from experiments.zeroshot_cf.evaluation.result import (
    METRIC_SCHEMA_VERSION,
    ArrayOutput,
    CandidateOutput,
    EvaluationReport,
    EvaluationSpec,
    PointOutput,
    SummaryOutput,
)


class Evaluator:
    """Prepare reusable metric state for one benchmark case."""

    def prepare(self, case: BenchmarkCase, spec: EvaluationSpec) -> PreparedEvaluator:
        lof, isolation = prepare_novelty_models(
            case.dataset.X_train,
            lof_n_neighbors=spec.lof_n_neighbors,
            isolation_forest_estimators=spec.isolation_forest_estimators,
        )
        return PreparedEvaluator(case=case, spec=spec, lof=lof, isolation=isolation)


@dataclass(frozen=True)
class PreparedEvaluator:
    case: BenchmarkCase
    spec: EvaluationSpec
    lof: object
    isolation: object

    def evaluate(self, result: GenerationResult) -> EvaluationReport:
        """Derive common metrics solely from the case, spec, and candidates."""
        factuals = self.case.factuals.values
        result.validate_for_factuals(factuals)
        n_factuals, k, _ = result.candidates.shape
        if n_factuals != len(factuals):
            raise ValueError("generation result factual count does not match case")
        if self.spec.primary_rank >= k:
            raise ValueError("primary_rank is outside the candidate rank dimension")

        predictions = np.full((n_factuals, k), None, dtype=object)
        probabilities = np.full((n_factuals, k), np.nan, dtype=np.float64)
        if result.available.any():
            flat_candidates = result.candidates[result.available]
            flat_targets = np.broadcast_to(
                self.case.targets[:, None], result.available.shape
            )[result.available]
            predicted = np.asarray(self.case.oracle.predict(flat_candidates)).reshape(
                -1
            )
            if len(predicted) != len(flat_candidates):
                raise ValueError("oracle predict row count does not match candidates")
            probability = target_probabilities(
                self.case.oracle, flat_candidates, flat_targets
            )
            predictions[result.available] = predicted
            probabilities[result.available] = probability

        targets = np.broadcast_to(self.case.targets[:, None], result.available.shape)
        class_success = result.available & (predictions == targets)
        threshold_success = class_success & (
            probabilities >= self.spec.probability_threshold
        )
        returned = int(result.available.sum())
        requested = int(result.available.size)
        primary_available = result.available[:, self.spec.primary_rank]
        primary_class = class_success[:, self.spec.primary_rank]
        primary_threshold = threshold_success[:, self.spec.primary_rank]

        summary: dict[str, int | float] = {
            "n_factuals": n_factuals,
            "n_requested_slots": requested,
            "n_returned_candidates": returned,
            "coverage": float(result.available.any(axis=1).mean()),
            "validity_returned_class": (
                float(class_success.sum() / returned) if returned else float("nan")
            ),
            "validity_returned_threshold": (
                float(threshold_success.sum() / returned) if returned else float("nan")
            ),
            "valid_success_rate_class_per_requested_slot": float(
                class_success.sum() / requested
            ),
            "valid_success_rate_threshold_per_requested_slot": float(
                threshold_success.sum() / requested
            ),
            "valid_success_rate_class_per_factual": float(
                class_success.any(axis=1).mean()
            ),
            "valid_success_rate_threshold_per_factual": float(
                threshold_success.any(axis=1).mean()
            ),
            "primary_coverage": float(primary_available.mean()),
            "primary_validity_returned_class": (
                float(primary_class.sum() / primary_available.sum())
                if primary_available.any()
                else float("nan")
            ),
            "primary_validity_returned_threshold": (
                float(primary_threshold.sum() / primary_available.sum())
                if primary_available.any()
                else float("nan")
            ),
        }
        metric_summary, metric_arrays = common_candidate_metrics(
            candidates=result.candidates,
            factuals=factuals,
            available=result.available,
            class_success=class_success,
            numerical=self.case.dataset.schema.numerical,
            categorical_groups=self.case.dataset.schema.categorical_groups,
            immutable=self.case.dataset.schema.immutable,
            sparsity_epsilon=self.spec.sparsity_epsilon,
            lof=self.lof,
            isolation=self.isolation,
        )
        summary.update(metric_summary)
        summary.update(
            evaluate_diverse_candidate_sets(
                factuals=factuals,
                candidates=result.candidates,
                available=result.available,
                class_success=class_success,
                threshold_success=threshold_success,
                numerical=self.case.dataset.schema.numerical,
                categorical_groups=self.case.dataset.schema.categorical_groups,
            )
        )
        summary["lof_scores_test"] = float((-self.lof.score_samples(factuals)).mean())
        summary["isolation_forest_scores_test"] = float(
            self.isolation.decision_function(factuals).mean()
        )

        points: list[PointOutput] = []
        candidates: list[CandidateOutput] = []
        for point in range(n_factuals):
            rank = self.spec.primary_rank
            points.append(
                PointOutput(
                    point=point,
                    values={
                        "factual_label": self.case.factuals.true_labels[point].item(),
                        "factual_prediction": self.case.factual_predictions[
                            point
                        ].item(),
                        "target": self.case.targets[point].item(),
                        "available": bool(primary_available[point]),
                        "candidate_prediction": (
                            predictions[point, rank]
                            if primary_available[point]
                            else None
                        ),
                        "target_probability": (
                            float(probabilities[point, rank])
                            if primary_available[point]
                            else None
                        ),
                        "valid_class": bool(primary_class[point]),
                        "valid_threshold": bool(primary_threshold[point]),
                    },
                )
            )
            for candidate_rank in range(k):
                available = bool(result.available[point, candidate_rank])
                candidates.append(
                    CandidateOutput(
                        point=point,
                        rank=candidate_rank,
                        values={
                            "available": available,
                            "prediction": (
                                predictions[point, candidate_rank]
                                if available
                                else None
                            ),
                            "target_probability": (
                                float(probabilities[point, candidate_rank])
                                if available
                                else None
                            ),
                            "valid_class": bool(class_success[point, candidate_rank]),
                            "valid_threshold": bool(
                                threshold_success[point, candidate_rank]
                            ),
                        },
                    )
                )

        arrays = {
            "common.candidates": result.candidates,
            "common.available": result.available,
            "common.target_probabilities": probabilities,
            "common.class_success": class_success,
            "common.threshold_success": threshold_success,
            **metric_arrays,
            **result.artifacts,
        }
        return EvaluationReport(
            schema_version=METRIC_SCHEMA_VERSION,
            summary=SummaryOutput(METRIC_SCHEMA_VERSION, summary),
            points=tuple(points),
            candidates=tuple(candidates),
            arrays=ArrayOutput(METRIC_SCHEMA_VERSION, arrays),
            metadata={
                "case_id": self.case.case_id,
                "probability_threshold": self.spec.probability_threshold,
                "primary_rank": self.spec.primary_rank,
            },
        )
