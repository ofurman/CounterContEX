from . import (
    mlp,
    flexible_categorical,
    prior_bag,
    counterfactual,
    counterfactual_prior,
)

# Optional imports — these require additional dependencies
try:
    from . import fast_gp
except ImportError:
    pass

try:
    from . import differentiable_prior
except ImportError:
    pass
