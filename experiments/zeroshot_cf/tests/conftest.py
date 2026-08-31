"""Make the repo root importable as the ``experiments`` package root."""

from __future__ import annotations

import sys
from pathlib import Path

# Add repo root so `from experiments.zeroshot_cf.*` imports resolve.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import the pinned vendored CEL checkout directly for focused tests. The full
# editable package drags in optional dependencies that this suite does not use.
_CEL_VENDOR_ROOT = REPO_ROOT / "experiments" / "zeroshot_cf" / "vendor" / "counterfactuals"
if str(_CEL_VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(_CEL_VENDOR_ROOT))

RETAINED_FOCUSED_TESTS = (
    "experiments/zeroshot_cf/tests/test_candidate_domains.py",
    "experiments/zeroshot_cf/tests/test_data_cleaning.py",
    "experiments/zeroshot_cf/tests/test_diverse_search.py",
    "experiments/zeroshot_cf/tests/test_exp9_benchmark.py",
    "experiments/zeroshot_cf/tests/test_exp11_nice_nun_baseline.py",
    "experiments/zeroshot_cf/tests/test_exp12_optimization_baselines.py",
    "experiments/zeroshot_cf/tests/test_exp13_dice_baseline.py",
    "experiments/zeroshot_cf/tests/test_exp14_face_baseline.py",
    "experiments/zeroshot_cf/tests/test_generator.py",
    "experiments/zeroshot_cf/tests/test_grouped_categorical.py",
    "experiments/zeroshot_cf/tests/test_metrics_harness.py",
    "experiments/zeroshot_cf/tests/test_mixed_distance.py",
    "experiments/zeroshot_cf/tests/test_tabicl_backend.py",
    "experiments/zeroshot_cf/tests/test_tabicl_plausibility.py",
    "experiments/zeroshot_cf/tests/test_tabicl_checkpoints.py",
    "experiments/zeroshot_cf/tests/test_dataset_contract.py",
)
