#  Copyright (c) Prior Labs GmbH 2026.

"""Minimal real-model smoke test for staged TabICLv2 checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def run_smoke(*, cache_dir: Path | None, device: str) -> None:
    from experiments.zeroshot_cf.tabicl_sampler import TabICLConditionalDensitySampler

    X = np.asarray(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.2, 0.8],
            [0.8, 0.2],
            [0.3, 0.7],
            [0.7, 0.3],
        ],
        dtype=np.float32,
    )
    y = np.asarray([0, 0, 1, 1, 0, 1, 0, 1], dtype=np.int64)
    sampler = TabICLConditionalDensitySampler(
        n_estimators=1,
        batch_size=1,
        cache_dir=cache_dir,
        device=device,
    )
    sampler.set_context(X, y_context=y, max_context=len(X))
    value = sampler.sample_candidates(X[[0]], [0], fixed_target=1)
    if value.shape != (1,) or not np.isfinite(value[0]):
        raise RuntimeError(f"TabICL smoke test returned an invalid value: {value}")
    print(f"TabICL smoke test passed; imputed feature 0 = {value[0]:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a minimal offline TabICL checkpoint smoke test."
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional checkpoint directory override.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device passed to the TabICL sampler (default: cpu).",
    )
    args = parser.parse_args()
    run_smoke(cache_dir=args.cache_dir, device=args.device)


if __name__ == "__main__":
    main()
