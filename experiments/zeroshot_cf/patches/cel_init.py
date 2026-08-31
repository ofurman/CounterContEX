"""Patched cel/__init__.py for vendor/counterfactuals.

Applied after cloning ofurman/counterfactuals to vendor/counterfactuals/.
Reason: cel/__init__.py imports heavy CF methods (nflows, alibi, tensorflow)
that are unavailable under Python 3.13. We only need cel.datasets, cel.metrics,
cel.preprocessing, and optionally cel.models (LR/MLP discriminator).
"""

__version__ = "0.1.0"

# Core modules always available (no heavy deps)
from cel.datasets import MethodDataset
from cel.metrics import CFMetrics, evaluate_cf

__all__ = [
    "__version__",
    "MethodDataset",
    "CFMetrics",
    "evaluate_cf",
]

# Optional heavy imports (CF methods, generative models, normalizing flows)
try:
    from cel.cf_methods import (
        PPCEF,
        BaseCounterfactualMethod,
        ExplanationResult,
    )
    __all__ += ["PPCEF", "BaseCounterfactualMethod", "ExplanationResult"]
except ImportError:
    pass

try:
    from cel.losses import BinaryDiscLoss, MulticlassDiscLoss
    __all__ += ["BinaryDiscLoss", "MulticlassDiscLoss"]
except ImportError:
    pass

try:
    from cel.models import MaskedAutoregressiveFlow, MLPClassifier
    __all__ += ["MaskedAutoregressiveFlow", "MLPClassifier"]
except ImportError:
    pass
