#  Copyright (c) Prior Labs GmbH 2026.

"""Minimal real-model smoke test for staged TabICLv2 checkpoints."""

from __future__ import annotations

import numpy as np
from experiments.zeroshot_cf.tabicl_sampler import TabICLConditionalDensitySampler


def main() -> None:
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
        device="cpu",
    )
    sampler.set_context(X, y_context=y, max_context=len(X))
    value = sampler.sample_candidates(X[[0]], [0], fixed_target=1)
    if value.shape != (1,) or not np.isfinite(value[0]):
        raise RuntimeError(f"TabICL smoke test returned an invalid value: {value}")
    print(f"TabICL smoke test passed; imputed feature 0 = {value[0]:.6f}")


if __name__ == "__main__":
    main()
