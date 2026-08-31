"""Proposal-backend contracts and retained adapters for DiCoFlex."""

from experiments.zeroshot_cf.methods.dicoflex.backends.base import (
    CategoryProposals,
    PreparedBackend,
    ProposalBackend,
    ProposalCapabilities,
    ProposalSession,
    validate_backend_capabilities,
)
from experiments.zeroshot_cf.methods.dicoflex.backends.empirical import EmpiricalBackend

__all__ = [
    "CategoryProposals",
    "EmpiricalBackend",
    "PreparedBackend",
    "ProposalBackend",
    "ProposalCapabilities",
    "ProposalSession",
    "validate_backend_capabilities",
]
