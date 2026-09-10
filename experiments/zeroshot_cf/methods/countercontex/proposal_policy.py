"""Deterministic, backend-neutral proposal decoding and accounting primitives."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Literal

import numpy as np

NUMERICAL_BIN_COUNT = 256
_MIN_OPEN_UNIFORM = 2.0**-53
_MAX_OPEN_UNIFORM = 1.0 - _MIN_OPEN_UNIFORM

DISPOSITION_NO_OP = 0
DISPOSITION_UNIQUE = 1
DISPOSITION_DUPLICATE = 2

EqualMassPolicy = Literal["iid", "top-k", "top-p"]
CategoricalPolicy = Literal["iid", "greedy", "top-k", "top-p"]


@dataclass(frozen=True)
class DecodedNumericalProposals:
    """Raw numerical values and the exact q coordinate of every draw."""

    values: np.ndarray
    quantiles: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64)
        quantiles = np.asarray(self.quantiles, dtype=np.float64)
        if values.ndim != 2 or quantiles.shape != values.shape:
            raise ValueError(
                "decoded numerical values and quantiles must share a 2D shape"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("decoded numerical values must be finite")
        finite_quantiles = quantiles[np.isfinite(quantiles)]
        if np.any(np.isinf(quantiles)) or np.any(
            (finite_quantiles <= 0.0) | (finite_quantiles >= 1.0)
        ):
            raise ValueError(
                "decoded quantiles must be NaN or lie strictly inside (0, 1)"
            )
        object.__setattr__(self, "values", _readonly(values, dtype=np.float64))
        object.__setattr__(
            self, "quantiles", _readonly(quantiles, dtype=np.float64)
        )


@dataclass(frozen=True)
class DecodedCategoricalProposals:
    """Raw category draws, including repeated categories and the current value."""

    categories: np.ndarray
    probabilities: np.ndarray
    support_size: int

    def __post_init__(self) -> None:
        categories = np.asarray(self.categories, dtype=np.int64)
        probabilities = np.asarray(self.probabilities, dtype=np.float64)
        if categories.ndim != 1 or probabilities.shape != categories.shape:
            raise ValueError("decoded categorical arrays must share a 1D shape")
        if len(categories) == 0 or np.any(categories < 0):
            raise ValueError("decoded categorical proposals must be non-empty")
        if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0.0):
            raise ValueError("decoded categorical probabilities must be non-negative")
        if not isinstance(self.support_size, int) or self.support_size < 1:
            raise ValueError("support_size must be a positive integer")
        object.__setattr__(self, "categories", _readonly(categories, dtype=np.int64))
        object.__setattr__(
            self, "probabilities", _readonly(probabilities, dtype=np.float64)
        )


def _readonly(array: np.ndarray, *, dtype: np.dtype | type) -> np.ndarray:
    result = np.array(array, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class SamplingKey:
    """Complete stable identity for one proposal-policy uniform."""

    run_seed: int
    factual_source_index: int
    state_depth: int
    parent_id: str
    action_unit_id: str
    draw_index: int
    stream: str

    def __post_init__(self) -> None:
        integer_fields = (
            self.run_seed,
            self.factual_source_index,
            self.state_depth,
            self.draw_index,
        )
        if any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in integer_fields
        ):
            raise ValueError("sampling-key integer fields must be integers")
        if any(value < 0 for value in integer_fields):
            raise ValueError("sampling-key integer fields must be non-negative")
        if not self.parent_id or not self.action_unit_id or not self.stream:
            raise ValueError("sampling-key string fields must be non-empty")

    def canonical_bytes(self) -> bytes:
        """Serialize without process-dependent hashes or mapping order."""
        payload = [
            self.run_seed,
            self.factual_source_index,
            self.state_depth,
            self.parent_id,
            self.action_unit_id,
            self.draw_index,
            self.stream,
        ]
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )


def keyed_uniform(key: SamplingKey) -> float:
    """Map a stable SHA-256 digest to an open-interval IEEE-754 uniform."""
    digest = hashlib.sha256(key.canonical_bytes()).digest()
    mantissa = int.from_bytes(digest[:8], "big") >> 11
    return (mantissa + 0.5) / 2.0**53


@dataclass(frozen=True)
class BinTruncation:
    """Density ordering and retained support for equal-q-mass bins."""

    log_densities: np.ndarray
    density_order: np.ndarray
    retained: np.ndarray

    def __post_init__(self) -> None:
        scores = np.asarray(self.log_densities, dtype=np.float64)
        order = np.asarray(self.density_order, dtype=np.int64)
        retained = np.asarray(self.retained, dtype=np.bool_)
        if scores.ndim != 1 or len(scores) == 0:
            raise ValueError(
                "bin log densities must be a non-empty one-dimensional array"
            )
        if np.any(np.isnan(scores)) or np.any(np.isposinf(scores)):
            raise ValueError(
                "bin log densities must not contain NaN or positive infinity"
            )
        if order.shape != scores.shape or retained.shape != scores.shape:
            raise ValueError("bin ordering and mask must match log-density shape")
        if not np.array_equal(np.sort(order), np.arange(len(scores))):
            raise ValueError("density order must be a permutation of bin indices")
        if not np.any(retained):
            raise ValueError("at least one equal-mass bin must be retained")
        object.__setattr__(self, "log_densities", _readonly(scores, dtype=np.float64))
        object.__setattr__(self, "density_order", _readonly(order, dtype=np.int64))
        object.__setattr__(self, "retained", _readonly(retained, dtype=np.bool_))

    @property
    def bin_count(self) -> int:
        return len(self.log_densities)

    @property
    def retained_count(self) -> int:
        return int(np.count_nonzero(self.retained))


def equal_mass_bin_truncation(
    log_densities: np.ndarray,
    *,
    policy: EqualMassPolicy,
    k: int | None = None,
    p: float | None = None,
) -> BinTruncation:
    """Select equal-mass quantile bins with stable lower-index tie breaking."""
    scores = np.asarray(log_densities, dtype=np.float64)
    if scores.ndim != 1 or len(scores) == 0:
        raise ValueError("bin log densities must be a non-empty one-dimensional array")
    if np.any(np.isnan(scores)) or np.any(np.isposinf(scores)):
        raise ValueError("bin log densities must not contain NaN or positive infinity")

    indices = np.arange(len(scores), dtype=np.int64)
    order = np.lexsort((indices, -scores))
    if policy == "iid":
        retained_count = len(scores)
    elif policy == "top-k":
        if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= len(scores):
            raise ValueError("top-k requires an integer k within the bin support")
        retained_count = k
    elif policy == "top-p":
        if p is None or not np.isfinite(p) or not 0.0 < p <= 1.0:
            raise ValueError("top-p requires finite p in (0, 1]")
        retained_count = int(np.ceil(p * len(scores)))
    else:
        raise ValueError(f"unknown equal-mass policy: {policy}")

    retained = np.zeros(len(scores), dtype=np.bool_)
    retained[order[:retained_count]] = True
    return BinTruncation(scores, order, retained)


@dataclass(frozen=True)
class EqualMassSamples:
    """Selected bin indices and their within-bin quantile coordinates."""

    bin_indices: np.ndarray
    within_bin_uniforms: np.ndarray
    quantiles: np.ndarray

    def __post_init__(self) -> None:
        bins = np.asarray(self.bin_indices, dtype=np.int64)
        within = np.asarray(self.within_bin_uniforms, dtype=np.float64)
        quantiles = np.asarray(self.quantiles, dtype=np.float64)
        if (
            bins.ndim != 1
            or within.shape != bins.shape
            or quantiles.shape != bins.shape
        ):
            raise ValueError("sample arrays must be equally sized and one-dimensional")
        object.__setattr__(self, "bin_indices", _readonly(bins, dtype=np.int64))
        object.__setattr__(
            self, "within_bin_uniforms", _readonly(within, dtype=np.float64)
        )
        object.__setattr__(self, "quantiles", _readonly(quantiles, dtype=np.float64))


def sample_equal_mass_bins(
    truncation: BinTruncation,
    *,
    selection_uniforms: np.ndarray,
    within_bin_uniforms: np.ndarray,
) -> EqualMassSamples:
    """Draw from retained bins while preserving their equal original q masses."""
    selection = np.asarray(selection_uniforms, dtype=np.float64)
    within = np.asarray(within_bin_uniforms, dtype=np.float64)
    if selection.ndim != 1 or within.shape != selection.shape:
        raise ValueError(
            "selection and within-bin uniforms must share a one-dimensional shape"
        )
    if not np.all(np.isfinite(selection)) or not np.all(np.isfinite(within)):
        raise ValueError("sampling uniforms must be finite")
    if np.any((selection < 0.0) | (selection > 1.0)):
        raise ValueError("selection uniforms must lie in [0, 1]")
    if np.any((within < 0.0) | (within > 1.0)):
        raise ValueError("within-bin uniforms must lie in [0, 1]")

    retained_indices = np.flatnonzero(truncation.retained)
    selection_clipped = np.minimum(selection, np.nextafter(1.0, 0.0))
    positions = np.floor(selection_clipped * len(retained_indices)).astype(np.int64)
    bins = retained_indices[positions]
    within_clipped = np.clip(within, _MIN_OPEN_UNIFORM, _MAX_OPEN_UNIFORM)
    quantiles = (bins + within_clipped) / truncation.bin_count
    quantiles = np.minimum(quantiles, np.nextafter(1.0, 0.0))
    return EqualMassSamples(bins, within_clipped, quantiles)


@dataclass(frozen=True)
class CategoricalTruncation:
    """Stable categorical ordering and normalized retained q mass."""

    order: np.ndarray
    retained: np.ndarray
    normalized_probabilities: np.ndarray

    def __post_init__(self) -> None:
        order = np.asarray(self.order, dtype=np.int64)
        retained = np.asarray(self.retained, dtype=np.bool_)
        probabilities = np.asarray(self.normalized_probabilities, dtype=np.float64)
        if (
            order.ndim != 1
            or retained.shape != order.shape
            or probabilities.shape != order.shape
        ):
            raise ValueError(
                "categorical truncation arrays must share a one-dimensional shape"
            )
        if not np.isclose(probabilities.sum(), 1.0):
            raise ValueError("normalized categorical probabilities must sum to one")
        object.__setattr__(self, "order", _readonly(order, dtype=np.int64))
        object.__setattr__(self, "retained", _readonly(retained, dtype=np.bool_))
        object.__setattr__(
            self,
            "normalized_probabilities",
            _readonly(probabilities, dtype=np.float64),
        )


def categorical_truncation(
    probabilities: np.ndarray,
    *,
    policy: CategoricalPolicy,
    k: int | None = None,
    p: float | None = None,
) -> CategoricalTruncation:
    """Retain exact categorical q mass with canonical support-order ties."""
    q = np.asarray(probabilities, dtype=np.float64)
    if q.ndim != 1 or len(q) == 0:
        raise ValueError(
            "categorical probabilities must be non-empty and one-dimensional"
        )
    if not np.all(np.isfinite(q)) or np.any(q < 0.0) or not np.isclose(q.sum(), 1.0):
        raise ValueError(
            "categorical probabilities must be finite, non-negative, and sum to one"
        )

    indices = np.arange(len(q), dtype=np.int64)
    order = np.lexsort((indices, -q))
    if policy == "iid":
        retained_count = len(q)
    elif policy == "greedy":
        retained_count = 1
    elif policy == "top-k":
        if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= len(q):
            raise ValueError("top-k requires an integer k within categorical support")
        retained_count = k
    elif policy == "top-p":
        if p is None or not np.isfinite(p) or not 0.0 < p <= 1.0:
            raise ValueError("top-p requires finite p in (0, 1]")
        retained_count = int(np.searchsorted(np.cumsum(q[order]), p, side="left") + 1)
    else:
        raise ValueError(f"unknown categorical policy: {policy}")

    retained = np.zeros(len(q), dtype=np.bool_)
    retained[order[:retained_count]] = True
    normalized = np.where(retained, q, 0.0)
    normalized /= normalized.sum()
    return CategoricalTruncation(order, retained, normalized)


@dataclass(frozen=True)
class NormalizedWeights:
    """Log-space normalized proposal weights and effective sample size."""

    weights: np.ndarray
    log_normalizer: float
    effective_sample_size: float

    def __post_init__(self) -> None:
        weights = np.asarray(self.weights, dtype=np.float64)
        if weights.ndim != 1 or not np.all(np.isfinite(weights)):
            raise ValueError("normalized weights must be finite and one-dimensional")
        if np.any(weights < 0.0) or not np.isclose(weights.sum(), 1.0):
            raise ValueError("normalized weights must be non-negative and sum to one")
        if not np.isfinite(self.log_normalizer):
            raise ValueError("log normalizer must be finite")
        expected_ess = 1.0 / float(np.square(weights).sum())
        if not np.isfinite(self.effective_sample_size) or not np.isclose(
            self.effective_sample_size, expected_ess
        ):
            raise ValueError("effective sample size must match normalized weights")
        object.__setattr__(self, "weights", _readonly(weights, dtype=np.float64))


def normalize_log_weights(
    log_base_mass: np.ndarray,
    classifier_probabilities: np.ndarray,
    *,
    beta: float,
    epsilon: float = 1e-12,
) -> NormalizedWeights:
    """Normalize log(q) + beta*log(s), or only beta*log(s) for q draws."""
    base = np.asarray(log_base_mass, dtype=np.float64)
    scores = np.asarray(classifier_probabilities, dtype=np.float64)
    if base.ndim != 1 or scores.ndim != 1 or base.shape != scores.shape:
        raise ValueError(
            "base masses and classifier probabilities must have the same shape"
        )
    if len(base) == 0:
        raise ValueError("at least one proposal weight is required")
    if np.any(np.isnan(base)) or np.any(np.isposinf(base)):
        raise ValueError("log base masses must not contain NaN or positive infinity")
    if not np.all(np.isfinite(scores)):
        raise ValueError("classifier probabilities must be finite")
    if np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError("classifier probabilities must lie in [0, 1]")
    if not np.isfinite(beta) or beta < 0.0:
        raise ValueError("beta must be finite and non-negative")
    if not np.isfinite(epsilon) or not 0.0 < epsilon <= 1.0:
        raise ValueError("epsilon must be finite and lie in (0, 1]")
    if not np.any(np.isfinite(base)):
        raise ValueError("at least one base mass must be positive")

    log_weights = base.copy()
    if beta != 0.0:
        log_weights += beta * np.log(np.maximum(scores, epsilon))
    maximum = float(np.max(log_weights))
    shifted = np.exp(log_weights - maximum)
    normalizer = float(shifted.sum())
    weights = shifted / normalizer
    log_normalizer = maximum + float(np.log(normalizer))
    ess = 1.0 / float(np.square(weights).sum())
    return NormalizedWeights(weights, log_normalizer, ess)


def weighted_resample(weights: np.ndarray, uniforms: np.ndarray) -> np.ndarray:
    """Standard inverse-CDF resampling with replacement and no refill."""
    probabilities = np.asarray(weights, dtype=np.float64)
    draws = np.asarray(uniforms, dtype=np.float64)
    if probabilities.ndim != 1 or len(probabilities) == 0:
        raise ValueError("weights must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0.0):
        raise ValueError("weights must be finite and non-negative")
    if not np.isclose(probabilities.sum(), 1.0):
        raise ValueError("weights must sum to one")
    if draws.ndim != 1 or not np.all(np.isfinite(draws)):
        raise ValueError("uniforms must be finite and one-dimensional")
    if np.any((draws < 0.0) | (draws > 1.0)):
        raise ValueError("uniforms must lie in [0, 1]")

    clipped = np.minimum(draws, np.nextafter(1.0, 0.0))
    cumulative = np.cumsum(probabilities)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, clipped, side="right")


@dataclass(frozen=True)
class ProposalBudget:
    """Raw, projected, and classifier-work counts for one proposal batch."""

    raw_count: int
    projected_count: int
    unique_classifier_count: int
    no_op_count: int
    duplicate_count: int

    def __post_init__(self) -> None:
        counts = (
            self.raw_count,
            self.projected_count,
            self.unique_classifier_count,
            self.no_op_count,
            self.duplicate_count,
        )
        if any(
            not isinstance(value, int) or isinstance(value, bool) for value in counts
        ):
            raise ValueError("proposal-budget counts must be integers")
        if any(value < 0 for value in counts):
            raise ValueError("proposal-budget counts must be non-negative")
        if self.raw_count != self.projected_count:
            raise ValueError("every raw proposal must have one projected row")
        if (
            self.no_op_count + self.duplicate_count + self.unique_classifier_count
            != self.raw_count
        ):
            raise ValueError("terminal proposal counts must partition the raw budget")


@dataclass(frozen=True)
class ProjectionAccounting:
    """Terminal disposition and classifier-cache linkage for every raw draw."""

    projected_rows: np.ndarray
    unique_classifier_rows: np.ndarray
    unique_index_by_raw: np.ndarray
    terminal_dispositions: np.ndarray
    budget: ProposalBudget

    def __post_init__(self) -> None:
        projected = np.asarray(self.projected_rows, dtype=np.float64)
        unique = np.asarray(self.unique_classifier_rows, dtype=np.float64)
        reverse = np.asarray(self.unique_index_by_raw, dtype=np.int64)
        dispositions = np.asarray(self.terminal_dispositions, dtype=np.int8)
        if projected.ndim != 2 or unique.ndim != 2:
            raise ValueError(
                "projected and unique classifier rows must be two-dimensional"
            )
        if unique.shape[1:] != projected.shape[1:]:
            raise ValueError(
                "projected and unique classifier rows must share feature width"
            )
        if reverse.shape != (len(projected),) or dispositions.shape != reverse.shape:
            raise ValueError("every raw row must have one reverse link and disposition")
        allowed = np.array(
            [DISPOSITION_NO_OP, DISPOSITION_UNIQUE, DISPOSITION_DUPLICATE],
            dtype=np.int8,
        )
        if not np.all(np.isin(dispositions, allowed)):
            raise ValueError("every raw row must have a known terminal disposition")
        no_op = dispositions == DISPOSITION_NO_OP
        changed = ~no_op
        if np.any(reverse[no_op] != -1):
            raise ValueError("no-op rows must not reference classifier rows")
        if np.any((reverse[changed] < 0) | (reverse[changed] >= len(unique))):
            raise ValueError("changed rows must reference one unique classifier row")
        observed = ProposalBudget(
            raw_count=len(projected),
            projected_count=len(projected),
            unique_classifier_count=len(unique),
            no_op_count=int(np.count_nonzero(no_op)),
            duplicate_count=int(
                np.count_nonzero(dispositions == DISPOSITION_DUPLICATE)
            ),
        )
        if self.budget != observed:
            raise ValueError("proposal budget does not match row dispositions")
        object.__setattr__(
            self, "projected_rows", _readonly(projected, dtype=np.float64)
        )
        object.__setattr__(
            self, "unique_classifier_rows", _readonly(unique, dtype=np.float64)
        )
        object.__setattr__(
            self, "unique_index_by_raw", _readonly(reverse, dtype=np.int64)
        )
        object.__setattr__(
            self,
            "terminal_dispositions",
            _readonly(dispositions, dtype=np.int8),
        )


def account_projected_rows(
    factual: np.ndarray,
    projected_rows: np.ndarray,
) -> ProjectionAccounting:
    """Account no-ops and first-seen changed rows after legal projection."""
    factual_row = np.asarray(factual, dtype=np.float64)
    projected = np.asarray(projected_rows, dtype=np.float64)
    if (
        factual_row.ndim != 1
        or projected.ndim != 2
        or projected.shape[1:] != factual_row.shape
    ):
        raise ValueError("factual and projected rows must have compatible shapes")
    if not np.all(np.isfinite(factual_row)) or not np.all(np.isfinite(projected)):
        raise ValueError("factual and projected rows must be finite")

    unique_rows: list[np.ndarray] = []
    reverse = np.full(len(projected), -1, dtype=np.int64)
    dispositions = np.empty(len(projected), dtype=np.int8)
    for raw_index, row in enumerate(projected):
        if np.array_equal(row, factual_row):
            dispositions[raw_index] = DISPOSITION_NO_OP
            continue
        match = next(
            (
                index
                for index, unique_row in enumerate(unique_rows)
                if np.array_equal(row, unique_row)
            ),
            None,
        )
        if match is None:
            match = len(unique_rows)
            unique_rows.append(row.copy())
            dispositions[raw_index] = DISPOSITION_UNIQUE
        else:
            dispositions[raw_index] = DISPOSITION_DUPLICATE
        reverse[raw_index] = match

    if unique_rows:
        unique = np.stack(unique_rows)
    else:
        unique = np.empty((0, factual_row.size), dtype=np.float64)
    budget = ProposalBudget(
        raw_count=len(projected),
        projected_count=len(projected),
        unique_classifier_count=len(unique),
        no_op_count=int(np.count_nonzero(dispositions == DISPOSITION_NO_OP)),
        duplicate_count=int(np.count_nonzero(dispositions == DISPOSITION_DUPLICATE)),
    )
    return ProjectionAccounting(projected, unique, reverse, dispositions, budget)
