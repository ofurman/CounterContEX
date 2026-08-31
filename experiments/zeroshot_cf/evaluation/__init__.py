"""Method-independent counterfactual evaluation."""

from experiments.zeroshot_cf.evaluation.evaluator import Evaluator, PreparedEvaluator
from experiments.zeroshot_cf.evaluation.result import (
    METRIC_SCHEMA_VERSION,
    EvaluationReport,
    EvaluationSpec,
)

__all__ = [
    "METRIC_SCHEMA_VERSION",
    "EvaluationReport",
    "EvaluationSpec",
    "Evaluator",
    "PreparedEvaluator",
]
