#  Copyright (c) Prior Labs GmbH 2026.

"""Tests for the official DiCE mixed-data adapter."""

from __future__ import annotations

import numpy as np
from experiments.zeroshot_cf.data import OneHotActionGroup
from experiments.zeroshot_cf.exp13_dice_baseline import DiceMixedAdapter


def test_dice_codec_round_trip_preserves_atomic_categories() -> None:
    codec = DiceMixedAdapter(
        n_features=5,
        scalar_columns=(0, 4),
        groups=(OneHotActionGroup("job", (1, 2, 3)),),
        scalar_names=("income", "age"),
    )
    matrix = np.array(
        [
            [0.2, 1.0, 0.0, 0.0, 0.7],
            [0.4, 0.0, 0.0, 1.0, 0.8],
        ]
    )

    frame = codec.encode(matrix)
    decoded = codec.decode(frame)

    assert list(frame) == ["income", "age", "job"]
    assert frame["job"].tolist() == ["0", "2"]
    np.testing.assert_array_equal(decoded, matrix)
