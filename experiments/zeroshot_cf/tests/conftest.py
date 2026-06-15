"""Make the TabPFN repo root importable as the 'experiments' package root."""

from __future__ import annotations

import sys
from pathlib import Path

# Add repo root so `from experiments.zeroshot_cf.*` imports resolve.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
