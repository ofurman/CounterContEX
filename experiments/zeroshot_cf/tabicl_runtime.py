"""Public compatibility exports for the retained TabICL benchmark runtime."""

from experiments.zeroshot_cf.orchestration.tabicl_runtime_compat import (
    TabICLBenchmarkRun,
    run_tabicl_benchmark,
)

__all__ = ["TabICLBenchmarkRun", "run_tabicl_benchmark"]
