"""Make the TabPFN repo root importable as the 'experiments' package root,
and host the shared ``models`` fixture used across the sampler/ordering/greedy/
context test modules."""

from __future__ import annotations

import sys
from pathlib import Path

# Add repo root so `from experiments.zeroshot_cf.*` imports resolve.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest  # noqa: E402


@pytest.fixture(scope="module")
def models():
    """Real local v2 TabPFN checkpoints (small n_estimators for test speed)."""
    from experiments.zeroshot_cf.checkpoints import get_models

    clf, reg = get_models(n_estimators=2)
    return clf, reg
