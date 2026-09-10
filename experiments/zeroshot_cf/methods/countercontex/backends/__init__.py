"""Proposal-backend contracts and retained adapters for CounterContEx."""

from experiments.zeroshot_cf.methods.countercontex.backends.base import (
    CategoryProposals,
    NumericalDistribution,
    PreparedBackend,
    ProposalBackend,
    ProposalCapabilities,
    ProposalSession,
    validate_backend_capabilities,
)
from experiments.zeroshot_cf.methods.countercontex.backends.empirical import (
    EmpiricalBackend,
)

__all__ = [
    "CategoryProposals",
    "EmpiricalBackend",
    "NumericalDistribution",
    "PreparedBackend",
    "ProposalBackend",
    "ProposalCapabilities",
    "ProposalSession",
    "validate_backend_capabilities",
]
