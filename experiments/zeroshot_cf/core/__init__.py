"""Provider-neutral contracts for the counterfactual benchmark."""

from experiments.zeroshot_cf.core.contracts import (
    BenchmarkCase,
    DatasetProvenance,
    FactualSelection,
    FeatureDomains,
    FeatureSchema,
    GenerationRequest,
    GenerationResult,
    MethodContext,
    Predictor,
    PreparedDataset,
)

__all__ = [
    "BenchmarkCase",
    "DatasetProvenance",
    "FactualSelection",
    "FeatureDomains",
    "FeatureSchema",
    "GenerationRequest",
    "GenerationResult",
    "MethodContext",
    "Predictor",
    "PreparedDataset",
]
