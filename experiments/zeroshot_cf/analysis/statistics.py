"""Paired tests and rank statistics used by the paper analysis."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import wilcoxon


@dataclass(frozen=True)
class SignificanceResult:
    comparison: str
    statistic: float
    corrected_p: float
    n: int
    mean_difference: float
    significant: bool
    below_noise_floor: bool


def holm_wilcoxon(
    comparisons: Mapping[str, tuple[Sequence[float], Sequence[float]]],
    *,
    noise_floor: float,
    alpha: float = 0.05,
) -> tuple[SignificanceResult, ...]:
    """Run paired Wilcoxon tests and Holm-correct all comparisons together."""
    measured: list[tuple[str, float, float, int, float]] = []
    for name, (left, right) in comparisons.items():
        x = np.asarray(left, dtype=float)
        y = np.asarray(right, dtype=float)
        if x.shape != y.shape or x.ndim != 1:
            raise ValueError("paired inputs must be equal-length vectors")
        finite = np.isfinite(x) & np.isfinite(y)
        differences = x[finite] - y[finite]
        if not len(differences):
            raise ValueError("paired inputs contain no finite observations")
        if np.all(differences == 0):
            statistic, p_value = 0.0, 1.0
        else:
            statistic, p_value = wilcoxon(differences)
        measured.append(
            (
                name,
                float(statistic),
                float(p_value),
                len(differences),
                float(np.mean(differences)),
            )
        )
    corrected: dict[str, float] = {}
    running = 0.0
    count = len(measured)
    for rank, item in enumerate(sorted(measured, key=lambda value: value[2])):
        adjusted = min(1.0, item[2] * (count - rank))
        running = max(running, adjusted)
        corrected[item[0]] = running
    return tuple(
        SignificanceResult(
            comparison=name,
            statistic=statistic,
            corrected_p=corrected[name],
            n=n,
            mean_difference=difference,
            significant=corrected[name] < alpha,
            below_noise_floor=abs(difference) < noise_floor,
        )
        for name, statistic, _p, n, difference in measured
    )
