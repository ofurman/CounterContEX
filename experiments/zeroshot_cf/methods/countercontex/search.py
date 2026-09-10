"""Backend-neutral adapter from CounterContEx sessions to the retained search core."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field, replace

import numpy as np
from experiments.zeroshot_cf.generator import (
    DiscriminatorProtocol,
    TabICLGeneratorInputs,
    TabICLGeneratorPointBackend,
    TabICLGeneratorResult,
    generate_counterfactual_batch,
)
from experiments.zeroshot_cf.methods.countercontex.backends.base import (
    PreparedBackend,
    ProposalSession,
    validate_backend_capabilities,
)
from experiments.zeroshot_cf.methods.countercontex.config import CounterContExConfig
from experiments.zeroshot_cf.methods.countercontex.proposal_policy import (
    DISPOSITION_DUPLICATE,
    DISPOSITION_NO_OP,
    DISPOSITION_UNIQUE,
    NUMERICAL_BIN_COUNT,
    DecodedCategoricalProposals,
    DecodedNumericalProposals,
    SamplingKey,
    account_projected_rows,
    categorical_truncation,
    equal_mass_bin_truncation,
    keyed_uniform,
    normalize_log_weights,
    sample_equal_mass_bins,
    weighted_resample,
)


def _parent_id(row: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(row, dtype=np.float64)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


@dataclass
class _ProposalAccounting:
    raw_records: list[tuple[int, str, str, int, bytes | None]] = field(
        default_factory=list
    )
    classifier_rows_by_key: dict[bytes, tuple[np.ndarray, float, int]] = field(
        default_factory=dict
    )
    baseline_rows_by_key: dict[bytes, tuple[np.ndarray, float, int]] = field(
        default_factory=dict
    )
    proposal_row_keys: set[bytes] = field(default_factory=set)
    tabicl_calls: int = 0
    tabicl_rows: int = 0
    classifier_calls: int = 0
    first_validity_classifier_calls: int | None = None
    first_validity_classifier_rows: int | None = None
    action_traces: list[_ActionTrace] = field(default_factory=list)
    guidance_traces: list[_GuidanceTrace] = field(default_factory=list)

    @property
    def raw_count(self) -> int:
        return len(self.raw_records)

    @property
    def projected_count(self) -> int:
        return len(self.raw_records)

    @property
    def unique_count(self) -> int:
        return len(self.proposal_row_keys)

    @property
    def no_op_count(self) -> int:
        return sum(record[-1] is None for record in self.raw_records)

    @property
    def duplicate_count(self) -> int:
        return self.raw_count - self.no_op_count - self.unique_count

    @property
    def classifier_rows(self) -> int:
        return len(self.classifier_rows_by_key | self.baseline_rows_by_key)

    def terminal_arrays(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ordered = sorted(self.raw_records)
        known_rows = self.classifier_rows_by_key | self.baseline_rows_by_key
        missing = {
            record[-1]
            for record in ordered
            if record[-1] is not None and record[-1] not in known_rows
        }
        if missing:
            raise RuntimeError(
                "changed proposal rows were not linked to classifier scores"
            )
        classifier_order = {
            row_key: index
            for index, row_key in enumerate(sorted(known_rows))
        }
        seen: set[bytes] = set(self.baseline_rows_by_key)
        dispositions: list[int] = []
        reverse: list[int] = []
        for *_, row_key in ordered:
            if row_key is None:
                dispositions.append(DISPOSITION_NO_OP)
                reverse.append(-1)
            elif row_key in seen:
                dispositions.append(DISPOSITION_DUPLICATE)
                reverse.append(classifier_order[row_key])
            else:
                seen.add(row_key)
                dispositions.append(DISPOSITION_UNIQUE)
                reverse.append(classifier_order[row_key])
        ordered_scored = [known_rows[key] for key in sorted(known_rows)]
        scored_rows = np.stack([item[0] for item in ordered_scored])
        target_probabilities = np.asarray(
            [item[1] for item in ordered_scored], dtype=np.float64
        )
        return (
            np.asarray(dispositions, dtype=np.int8),
            np.asarray(reverse, dtype=np.int64),
            scored_rows,
            target_probabilities,
        )


@dataclass(frozen=True)
class _ActionTrace:
    key: tuple[int, str, str]
    action_type: int
    support_values: np.ndarray
    support_log_probabilities: np.ndarray
    support_order: np.ndarray
    support_retained: np.ndarray
    selection_uniforms: np.ndarray
    within_bin_uniforms: np.ndarray
    selected_quantiles: np.ndarray
    selected_values: np.ndarray
    selected_probabilities: np.ndarray
    selected_log_probabilities: np.ndarray


@dataclass(frozen=True)
class _GuidanceTrace:
    key: tuple[int, str, str]
    pool_id: str
    policy: str
    beta: float
    raw_pool_size: int
    unique_pool_size: int
    log_normalizer: float
    effective_sample_size: float
    selection_uniforms: np.ndarray
    raw_target_probabilities: np.ndarray
    raw_weights: np.ndarray
    selected_pool_indices: np.ndarray


def _classifier_classes(discriminator: DiscriminatorProtocol) -> np.ndarray:
    return np.asarray(discriminator.classes_)


class _SessionSampler:
    """Legacy sampler surface implemented only through a proposal session."""

    def __init__(
        self,
        session: ProposalSession,
        *,
        config: CounterContExConfig | None = None,
        seed: int = 0,
        factual_source_index: int = 0,
        target: int = 0,
        discriminator: DiscriminatorProtocol | None = None,
    ) -> None:
        self._session = session
        self._config = CounterContExConfig() if config is None else config
        self._seed = seed
        self._factual_source_index = factual_source_index
        self._target = int(target)
        self._discriminator = discriminator
        self.accounting = _ProposalAccounting()

    def sample_candidates(
        self,
        X_query: np.ndarray,
        candidate_cols: Sequence[int],
        *,
        sample_temperature: float,
        fixed_target: int,
        fixed_confidence: float | None = None,
    ) -> np.ndarray:
        del fixed_target
        return self._session.propose_numerical(
            X_query,
            candidate_cols,
            quantiles=None,
            confidence=fixed_confidence,
            temperature=sample_temperature,
        )

    def sample_candidate_grid(
        self,
        X_query: np.ndarray,
        candidate_cols: Sequence[int],
        *,
        quantiles: Sequence[float],
        fixed_target: int,
        confidences: Sequence[float] | None = None,
    ) -> np.ndarray:
        del fixed_target
        anchors: tuple[float | None, ...] = (
            (None,) if confidences is None else tuple(float(v) for v in confidences)
        )
        grids = [
            np.asarray(
                self._session.propose_numerical(
                    X_query,
                    candidate_cols,
                    quantiles=quantiles,
                    confidence=anchor,
                    temperature=0.0,
                ),
                dtype=np.float64,
            )
            for anchor in anchors
        ]
        return grids[0] if confidences is None else np.stack(grids, axis=1)

    def sample_candidates_batch(
        self,
        X_queries: np.ndarray,
        candidate_cols: Sequence[int],
        *,
        sample_temperature: float,
        fixed_target: int,
        fixed_confidence: float | Sequence[float] | np.ndarray | None = None,
    ) -> np.ndarray:
        del fixed_target
        return np.asarray(
            self._session.propose_numerical_batch(
                X_queries,
                candidate_cols,
                quantiles=None,
                confidences=fixed_confidence,
                temperature=sample_temperature,
            ),
            dtype=np.float64,
        )

    def sample_candidate_grid_batch(
        self,
        X_queries: np.ndarray,
        candidate_cols: Sequence[int],
        *,
        quantiles: Sequence[float],
        fixed_target: int,
        confidences: Sequence[float] | None = None,
    ) -> np.ndarray:
        del fixed_target
        return np.asarray(
            self._session.propose_numerical_batch(
                X_queries,
                candidate_cols,
                quantiles=quantiles,
                confidences=confidences,
                temperature=0.0,
            ),
            dtype=np.float64,
        )

    def _uniforms(
        self,
        *,
        row: np.ndarray,
        column: int | str,
        state_depth: int,
        count: int,
        stream: str,
    ) -> np.ndarray:
        parent = _parent_id(row)
        return np.asarray(
            [
                keyed_uniform(
                    SamplingKey(
                        run_seed=self._seed,
                        factual_source_index=self._factual_source_index,
                        state_depth=state_depth,
                        parent_id=parent,
                        action_unit_id=f"{column}:target-{self._target}",
                        draw_index=index,
                        stream=stream,
                    )
                )
                for index in range(count)
            ],
            dtype=np.float64,
        )

    def _distribution_values(
        self,
        row: np.ndarray,
        column: int,
        quantiles: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        distribution = self._session.numerical_distribution_batch(
            np.asarray(row, dtype=np.float64).reshape(1, -1),
            [column],
            quantiles=quantiles,
            confidences=None,
        )
        self.accounting.tabicl_calls += 1
        self.accounting.tabicl_rows += int(distribution.values.size)
        return (
            np.asarray(distribution.values, dtype=np.float64).reshape(-1),
            np.asarray(distribution.log_probabilities, dtype=np.float64).reshape(-1),
        )

    def _record_action_trace(
        self,
        *,
        row: np.ndarray,
        action_unit_id: str,
        state_depth: int,
        action_type: int,
        support_values: np.ndarray,
        support_log_probabilities: np.ndarray,
        support_order: np.ndarray,
        support_retained: np.ndarray,
        selection_uniforms: np.ndarray,
        within_bin_uniforms: np.ndarray,
        selected_quantiles: np.ndarray,
        selected_values: np.ndarray,
        selected_probabilities: np.ndarray,
        selected_log_probabilities: np.ndarray,
    ) -> None:
        self.accounting.action_traces.append(
            _ActionTrace(
                key=(state_depth, _parent_id(row), action_unit_id),
                action_type=action_type,
                support_values=np.asarray(support_values).copy(),
                support_log_probabilities=np.asarray(
                    support_log_probabilities
                ).copy(),
                support_order=np.asarray(support_order, dtype=np.int64).copy(),
                support_retained=np.asarray(
                    support_retained, dtype=np.bool_
                ).copy(),
                selection_uniforms=np.asarray(selection_uniforms).copy(),
                within_bin_uniforms=np.asarray(within_bin_uniforms).copy(),
                selected_quantiles=np.asarray(selected_quantiles).copy(),
                selected_values=np.asarray(selected_values).copy(),
                selected_probabilities=np.asarray(selected_probabilities).copy(),
                selected_log_probabilities=np.asarray(
                    selected_log_probabilities
                ).copy(),
            )
        )

    def sample_budgeted_candidates(
        self,
        row: np.ndarray,
        columns: Sequence[int],
        *,
        state_depth: int,
    ) -> DecodedNumericalProposals:
        """Decode exactly B raw q proposals for every state/action pair."""
        search = self._config.search
        budget = int(
            search.guidance_pool_size
            if search.guidance_policy != "none"
            else search.numerical_proposal_budget or 0
        )
        values = np.empty((len(columns), budget), dtype=np.float64)
        quantiles = np.empty_like(values)
        for pair_index, raw_column in enumerate(columns):
            column = int(raw_column)
            decoder = search.numerical_decoder
            selection_uniforms = np.full(budget, np.nan)
            within_bin_uniforms = np.full(budget, np.nan)
            if decoder == "mode":
                proposed = self._session.propose_numerical(
                    np.asarray(row).reshape(1, -1),
                    [column],
                    quantiles=None,
                    confidence=None,
                    temperature=0.0,
                )
                self.accounting.tabicl_calls += 1
                self.accounting.tabicl_rows += 1
                decoded = np.asarray(proposed, dtype=np.float64).reshape(-1)
                q = np.full(budget, np.nan)
                support_values = decoded
                support_log_probabilities = np.full(budget, np.nan)
                support_order = np.arange(budget)
                support_retained = np.ones(budget, dtype=np.bool_)
            elif decoder == "grid":
                q = np.asarray(search.candidate_quantiles, dtype=np.float64)
                proposed = self._session.propose_numerical(
                    np.asarray(row).reshape(1, -1),
                    [column],
                    quantiles=q,
                    confidence=None,
                    temperature=0.0,
                )
                self.accounting.tabicl_calls += 1
                self.accounting.tabicl_rows += budget
                decoded = np.asarray(proposed, dtype=np.float64).reshape(-1)
                support_values = decoded
                support_log_probabilities = np.full(budget, np.nan)
                support_order = np.arange(budget)
                support_retained = np.ones(budget, dtype=np.bool_)
            elif decoder == "iid":
                selection_uniforms = self._uniforms(
                    row=row,
                    column=column,
                    state_depth=state_depth,
                    count=budget,
                    stream="q-bin-selection",
                )
                q = selection_uniforms
                decoded, selected_log_probabilities = self._distribution_values(
                    row, column, q
                )
                support_values = decoded
                support_log_probabilities = selected_log_probabilities
                support_order = np.arange(budget)
                support_retained = np.ones(budget, dtype=np.bool_)
            else:
                midpoints = (np.arange(NUMERICAL_BIN_COUNT) + 0.5) / NUMERICAL_BIN_COUNT
                midpoint_values, log_probabilities = self._distribution_values(
                    row, column, midpoints
                )
                truncation = equal_mass_bin_truncation(
                    log_probabilities,
                    policy=decoder,
                    k=(search.truncation_k if decoder == "top-k" else None),
                    p=(search.truncation_p if decoder == "top-p" else None),
                )
                selection_uniforms = self._uniforms(
                    row=row,
                    column=column,
                    state_depth=state_depth,
                    count=budget,
                    stream="q-bin-selection",
                )
                within_bin_uniforms = self._uniforms(
                    row=row,
                    column=column,
                    state_depth=state_depth,
                    count=budget,
                    stream="q-within-bin",
                )
                q = sample_equal_mass_bins(
                    truncation,
                    selection_uniforms=selection_uniforms,
                    within_bin_uniforms=within_bin_uniforms,
                ).quantiles
                decoded, selected_log_probabilities = self._distribution_values(
                    row, column, q
                )
                support_values = midpoint_values
                support_log_probabilities = log_probabilities
                support_order = truncation.density_order
                support_retained = truncation.retained
            values[pair_index] = decoded
            quantiles[pair_index] = q
            self._record_action_trace(
                row=row,
                action_unit_id=str(column),
                state_depth=state_depth,
                action_type=0,
                support_values=support_values,
                support_log_probabilities=support_log_probabilities,
                support_order=support_order,
                support_retained=support_retained,
                selection_uniforms=selection_uniforms,
                within_bin_uniforms=within_bin_uniforms,
                selected_quantiles=q,
                selected_values=decoded,
                selected_probabilities=(
                    np.exp(selected_log_probabilities)
                    if decoder in {"iid", "top-k", "top-p"}
                    else np.full(budget, np.nan)
                ),
                selected_log_probabilities=(
                    selected_log_probabilities
                    if decoder in {"iid", "top-k", "top-p"}
                    else np.full(budget, np.nan)
                ),
            )
        return DecodedNumericalProposals(values=values, quantiles=quantiles)

    def record_projected_rows(
        self,
        factual: np.ndarray,
        projected_rows: np.ndarray,
        *,
        action_unit_id: str,
        state_depth: int,
    ):
        accounting = account_projected_rows(factual, projected_rows)
        parent = _parent_id(factual)
        self.accounting.raw_records.extend(
            (
                state_depth,
                parent,
                action_unit_id,
                draw_index,
                (
                    None
                    if int(disposition) == DISPOSITION_NO_OP
                    else np.ascontiguousarray(projected_rows[draw_index]).tobytes()
                ),
            )
            for draw_index, disposition in enumerate(
                accounting.terminal_dispositions
            )
        )
        self.accounting.proposal_row_keys.update(
            np.ascontiguousarray(projected_rows[index], dtype=np.float64).tobytes()
            for index, disposition in enumerate(accounting.terminal_dispositions)
            if int(disposition) != DISPOSITION_NO_OP
            and np.ascontiguousarray(
                projected_rows[index], dtype=np.float64
            ).tobytes()
            not in self.accounting.baseline_rows_by_key
        )
        return accounting

    def _record_scored_rows(
        self,
        destination: dict[bytes, tuple[np.ndarray, float, int]],
        rows: np.ndarray,
        target_probabilities: np.ndarray,
        predictions: np.ndarray | None = None,
    ) -> None:
        matrix = np.atleast_2d(np.asarray(rows, dtype=np.float64))
        probabilities = np.asarray(target_probabilities, dtype=np.float64).reshape(-1)
        if probabilities.shape != (len(matrix),) or not np.all(
            np.isfinite(probabilities)
        ):
            raise ValueError("classifier scores must align with scored rows")
        predicted = (
            np.where(probabilities >= 0.5, self._target, 1 - self._target)
            if predictions is None
            else np.asarray(predictions).reshape(-1)
        )
        if predicted.shape != (len(matrix),):
            raise ValueError("classifier predictions must align with scored rows")
        known = (
            self.accounting.classifier_rows_by_key
            | self.accounting.baseline_rows_by_key
        )
        for row, probability, prediction in zip(
            matrix, probabilities, predicted, strict=True
        ):
            canonical = np.ascontiguousarray(row, dtype=np.float64)
            key = canonical.tobytes()
            previous = known.get(key)
            if previous is not None and not np.isclose(
                previous[1], probability, rtol=0.0, atol=1e-12
            ):
                raise ValueError("classifier returned inconsistent scores for one row")
            if previous is not None and previous[2] != int(prediction):
                raise ValueError(
                    "classifier returned inconsistent predictions for one row"
                )
            if previous is None:
                destination[key] = (
                    canonical.copy(),
                    float(probability),
                    int(prediction),
                )
                known[key] = destination[key]

    def classifier_outputs(
        self, rows: np.ndarray, *, baseline: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._discriminator is None:
            raise RuntimeError("classifier scoring is unavailable for this sampler")
        matrix = np.atleast_2d(np.asarray(rows, dtype=np.float64))
        known = (
            self.accounting.classifier_rows_by_key
            | self.accounting.baseline_rows_by_key
        )
        missing: dict[bytes, np.ndarray] = {}
        for row in matrix:
            canonical = np.ascontiguousarray(row, dtype=np.float64)
            if canonical.tobytes() not in known:
                missing.setdefault(canonical.tobytes(), canonical)
        if missing:
            missing_rows = np.stack(list(missing.values()))
            probability_matrix = np.asarray(
                self._discriminator.predict_proba(missing_rows), dtype=np.float64
            )
            if (
                probability_matrix.ndim != 2
                or probability_matrix.shape[0] != len(missing_rows)
                or probability_matrix.shape[1] < 2
                or not np.all(np.isfinite(probability_matrix))
                or np.any(
                    (probability_matrix < 0.0) | (probability_matrix > 1.0)
                )
            ):
                raise ValueError("classifier returned invalid probability rows")
            classes = _classifier_classes(self._discriminator)
            positions = np.flatnonzero(classes == self._target)
            if len(positions) != 1:
                raise ValueError(
                    f"target class {self._target} is absent from classifier classes"
                )
            target_probabilities = probability_matrix[:, int(positions[0])]
            predictions = classes[np.argmax(probability_matrix, axis=1)]
            self.accounting.classifier_calls += 1
            if baseline:
                self.record_baseline_rows(
                    missing_rows, target_probabilities, predictions
                )
            else:
                self.record_classifier_rows(
                    missing_rows, target_probabilities, predictions
                )
            known = (
                self.accounting.classifier_rows_by_key
                | self.accounting.baseline_rows_by_key
            )
        scores = np.asarray(
            [known[np.ascontiguousarray(row).tobytes()][1] for row in matrix],
            dtype=np.float64,
        )
        labels = np.asarray(
            [known[np.ascontiguousarray(row).tobytes()][2] for row in matrix]
        )
        return scores, labels

    def mark_first_validity(self) -> None:
        if self.accounting.first_validity_classifier_calls is None:
            self.accounting.first_validity_classifier_calls = (
                self.accounting.classifier_calls
            )
            self.accounting.first_validity_classifier_rows = (
                self.accounting.classifier_rows
            )

    def guide_projected_rows(
        self,
        factual: np.ndarray,
        projected_pool: np.ndarray,
        *,
        action_unit_id: str,
        state_depth: int,
        proposal_budget: int,
    ) -> np.ndarray:
        """Select B pool rows using p(y_t|x)^beta without another q factor."""
        search = self._config.search
        pool = np.atleast_2d(np.asarray(projected_pool, dtype=np.float64))
        if search.guidance_policy == "none":
            return np.arange(len(pool), dtype=np.int64)
        if len(pool) != search.guidance_pool_size:
            raise ValueError("guided q pool does not match declared M")

        unique: dict[bytes, np.ndarray] = {}
        for row in pool:
            canonical = np.ascontiguousarray(row, dtype=np.float64)
            unique.setdefault(canonical.tobytes(), canonical)
        unique_rows = np.stack([unique[key] for key in sorted(unique)])
        self.classifier_outputs(unique_rows)
        known = (
            self.accounting.classifier_rows_by_key
            | self.accounting.baseline_rows_by_key
        )
        raw_scores = np.asarray(
            [known[np.ascontiguousarray(row).tobytes()][1] for row in pool],
            dtype=np.float64,
        )
        parent = _parent_id(factual)
        action_key = f"{action_unit_id}:target-{self._target}"
        pool_hasher = hashlib.sha256()
        pool_hasher.update(
            SamplingKey(
                run_seed=self._seed,
                factual_source_index=self._factual_source_index,
                state_depth=state_depth,
                parent_id=parent,
                action_unit_id=action_key,
                draw_index=0,
                stream="q-pool-id",
            ).canonical_bytes()
        )
        pool_hasher.update(np.ascontiguousarray(pool).tobytes())
        pool_id = pool_hasher.hexdigest()

        if search.guidance_policy == "classifier-top-b":
            first_positions: dict[bytes, int] = {}
            for index, row in enumerate(pool):
                first_positions.setdefault(np.ascontiguousarray(row).tobytes(), index)
            ranked = sorted(
                first_positions.values(),
                key=lambda index: (-raw_scores[index], index),
            )
            selected = np.asarray(ranked[:proposal_budget], dtype=np.int64)
            selection_uniforms = np.empty(0, dtype=np.float64)
            weights = np.full(len(pool), np.nan)
            log_normalizer = np.nan
            ess = np.nan
        else:
            normalized = normalize_log_weights(
                np.full(len(pool), -np.log(len(pool)), dtype=np.float64),
                raw_scores,
                beta=search.guidance_beta,
                epsilon=search.guidance_epsilon,
            )
            selection_uniforms = self._uniforms(
                row=factual,
                column=action_unit_id,
                state_depth=state_depth,
                count=proposal_budget,
                stream="pi-beta-resampling",
            )
            selected = weighted_resample(normalized.weights, selection_uniforms)
            weights = normalized.weights
            log_normalizer = normalized.log_normalizer
            ess = normalized.effective_sample_size
        self.accounting.guidance_traces.append(
            _GuidanceTrace(
                key=(state_depth, parent, action_unit_id),
                pool_id=pool_id,
                policy=search.guidance_policy,
                beta=search.guidance_beta,
                raw_pool_size=len(pool),
                unique_pool_size=len(unique_rows),
                log_normalizer=log_normalizer,
                effective_sample_size=ess,
                selection_uniforms=selection_uniforms.copy(),
                raw_target_probabilities=raw_scores.copy(),
                raw_weights=weights.copy(),
                selected_pool_indices=selected.copy(),
            )
        )
        return selected

    def record_classifier_rows(
        self,
        rows: np.ndarray,
        target_probabilities: np.ndarray,
        predictions: np.ndarray | None = None,
    ) -> None:
        self._record_scored_rows(
            self.accounting.classifier_rows_by_key,
            rows,
            target_probabilities,
            predictions,
        )

    def record_baseline_rows(
        self,
        rows: np.ndarray,
        target_probabilities: np.ndarray,
        predictions: np.ndarray | None = None,
    ) -> None:
        self._record_scored_rows(
            self.accounting.baseline_rows_by_key,
            rows,
            target_probabilities,
            predictions,
        )


class _CategoryDecoder:
    def __init__(self, sampler: _SessionSampler) -> None:
        self._sampler = sampler

    def __call__(self, row, group, confidence):
        proposals = self._sampler._session.categorical_distribution(
            row, group, confidence=confidence
        )
        self._sampler.accounting.tabicl_calls += 1
        self._sampler.accounting.tabicl_rows += 1
        return proposals.categories, proposals.probabilities

    def record_projected_rows(
        self, factual, projected_rows, *, action_unit_id, state_depth
    ):
        return self._sampler.record_projected_rows(
            factual,
            projected_rows,
            action_unit_id=action_unit_id,
            state_depth=state_depth,
        )

    def budgeted(self, row, group, confidence, *, state_depth: int):
        categories, probabilities = self(row, group, confidence)
        support_order = np.argsort(np.asarray(categories), kind="stable")
        support = np.asarray(categories, dtype=np.int64)[support_order]
        q = np.asarray(probabilities, dtype=np.float64)[support_order]
        if np.any(support >= len(group.columns)):
            raise ValueError("categorical support is outside the one-hot group")
        search = self._sampler._config.search
        budget = int(
            search.guidance_pool_size
            if search.guidance_policy != "none"
            else search.categorical_proposal_budget or 0
        )
        decoder = search.categorical_decoder
        truncation = categorical_truncation(
            q,
            policy=decoder,
            k=(min(search.truncation_k, len(q)) if decoder == "top-k" else None),
            p=(search.truncation_p if decoder == "top-p" else None),
        )
        if decoder == "greedy":
            selected = truncation.order[:1]
            selection_uniforms = np.full(1, np.nan)
        else:
            selection_uniforms = self._sampler._uniforms(
                row=np.asarray(row),
                column=group.name,
                state_depth=state_depth,
                count=budget,
                stream="q-category-selection",
            )
            selected = weighted_resample(
                truncation.normalized_probabilities, selection_uniforms
            )
        self._sampler._record_action_trace(
            row=np.asarray(row),
            action_unit_id=group.name,
            state_depth=state_depth,
            action_type=1,
            support_values=support,
            support_log_probabilities=np.log(
                q, where=q > 0, out=np.full_like(q, -np.inf)
            ),
            support_order=truncation.order,
            support_retained=truncation.retained,
            selection_uniforms=selection_uniforms,
            within_bin_uniforms=np.full(len(selected), np.nan),
            selected_quantiles=np.full(len(selected), np.nan),
            selected_values=support[selected],
            selected_probabilities=q[selected],
            selected_log_probabilities=np.log(
                q[selected],
                where=q[selected] > 0,
                out=np.full_like(q[selected], -np.inf),
            ),
        )
        return DecodedCategoricalProposals(
            categories=support[selected],
            probabilities=q[selected],
            support_size=len(support),
        )

    def guide_projected_rows(
        self,
        factual,
        projected_pool,
        *,
        action_unit_id,
        state_depth,
    ):
        return self._sampler.guide_projected_rows(
            factual,
            projected_pool,
            action_unit_id=action_unit_id,
            state_depth=state_depth,
            proposal_budget=int(
                self._sampler._config.search.categorical_proposal_budget or 0
            ),
        )


@dataclass
class _SessionJointScorer:
    session: ProposalSession
    batch_count: int = 0
    row_count: int = 0

    def score_rows(self, rows: np.ndarray, target_class: int):
        scores = np.asarray(
            self.session.score_joint(rows, int(target_class)), dtype=np.float64
        )
        matrix = np.atleast_2d(np.asarray(rows))
        if scores.shape != (len(matrix),) or not np.all(np.isfinite(scores)):
            raise ValueError("proposal backend returned invalid joint scores")
        self.batch_count += 1
        self.row_count += len(matrix)
        # Retained search reads this stable value object by attribute.
        return _JointScoreBatch(scores)


@dataclass(frozen=True)
class _JointScoreBatch:
    joint_log_density: np.ndarray


def _point_backend(
    session: ProposalSession,
    *,
    discriminator: DiscriminatorProtocol,
    use_categorical_distribution: bool,
    use_joint_scoring: bool,
    config: CounterContExConfig,
    seed: int,
    factual_source_index: int,
    target: int,
) -> TabICLGeneratorPointBackend:
    sampler = _SessionSampler(
        session,
        config=config,
        seed=seed,
        factual_source_index=factual_source_index,
        target=target,
        discriminator=discriminator,
    )
    category_distribution = None
    if use_categorical_distribution:
        category_distribution = _CategoryDecoder(sampler)

    joint_scorer = _SessionJointScorer(session) if use_joint_scoring else None
    return TabICLGeneratorPointBackend(
        sampler=sampler,
        candidate_confidences=session.confidence_anchors,
        category_distribution=category_distribution,
        joint_scorer=joint_scorer,
        metadata=dict(session.diagnostics),
    )


def _proposal_trace_arrays(
    samplers: Sequence[_SessionSampler],
) -> dict[str, np.ndarray]:
    action_records: list[tuple[int, int, _ActionTrace]] = []
    for factual_position, sampler in enumerate(samplers):
        action_records.extend(
            (factual_position, sampler._factual_source_index, trace)
            for trace in sorted(
                sampler.accounting.action_traces, key=lambda item: item.key
            )
        )

    support_offsets = [0]
    selected_offsets = [0]
    for _, _, trace in action_records:
        support_offsets.append(support_offsets[-1] + len(trace.support_values))
        selected_offsets.append(selected_offsets[-1] + len(trace.selected_values))

    def concatenate(arrays: list[np.ndarray], dtype) -> np.ndarray:
        arrays = [np.asarray(array, dtype=dtype) for array in arrays]
        return np.concatenate(arrays) if arrays else np.empty(0, dtype=dtype)

    return {
        "action_factual_position": np.asarray(
            [position for position, _, _ in action_records], dtype=np.int64
        ),
        "action_factual_source_index": np.asarray(
            [source for _, source, _ in action_records], dtype=np.int64
        ),
        "action_depth": np.asarray(
            [trace.key[0] for _, _, trace in action_records], dtype=np.int64
        ),
        "action_parent_id": np.asarray(
            [trace.key[1] for _, _, trace in action_records], dtype="<U64"
        ),
        "action_unit_id": np.asarray(
            [trace.key[2] for _, _, trace in action_records], dtype="<U128"
        ),
        "action_type": np.asarray(
            [trace.action_type for _, _, trace in action_records], dtype=np.int8
        ),
        "support_offsets": np.asarray(support_offsets, dtype=np.int64),
        "selected_offsets": np.asarray(selected_offsets, dtype=np.int64),
        "support_values": concatenate(
            [trace.support_values for _, _, trace in action_records], np.float64
        ),
        "support_log_probabilities": concatenate(
            [
                trace.support_log_probabilities
                for _, _, trace in action_records
            ],
            np.float64,
        ),
        "support_order": concatenate(
            [trace.support_order for _, _, trace in action_records], np.int64
        ),
        "support_retained": concatenate(
            [trace.support_retained for _, _, trace in action_records], np.bool_
        ),
        "selection_uniforms": concatenate(
            [trace.selection_uniforms for _, _, trace in action_records],
            np.float64,
        ),
        "within_bin_uniforms": concatenate(
            [trace.within_bin_uniforms for _, _, trace in action_records],
            np.float64,
        ),
        "selected_quantiles": concatenate(
            [trace.selected_quantiles for _, _, trace in action_records],
            np.float64,
        ),
        "selected_values": concatenate(
            [trace.selected_values for _, _, trace in action_records], np.float64
        ),
        "selected_probabilities": concatenate(
            [trace.selected_probabilities for _, _, trace in action_records],
            np.float64,
        ),
        "selected_log_probabilities": concatenate(
            [
                trace.selected_log_probabilities
                for _, _, trace in action_records
            ],
            np.float64,
        ),
    }


def _guidance_trace_arrays(
    samplers: Sequence[_SessionSampler],
) -> dict[str, np.ndarray]:
    records: list[tuple[int, int, _GuidanceTrace]] = []
    for factual_position, sampler in enumerate(samplers):
        records.extend(
            (factual_position, sampler._factual_source_index, trace)
            for trace in sorted(sampler.accounting.guidance_traces, key=lambda x: x.key)
        )
    pool_offsets = [0]
    selected_offsets = [0]
    for _, _, trace in records:
        pool_offsets.append(pool_offsets[-1] + trace.raw_pool_size)
        selected_offsets.append(
            selected_offsets[-1] + len(trace.selected_pool_indices)
        )

    def concatenate(arrays: list[np.ndarray], dtype) -> np.ndarray:
        values = [np.asarray(array, dtype=dtype) for array in arrays]
        return np.concatenate(values) if values else np.empty(0, dtype=dtype)

    return {
        "action_factual_position": np.asarray(
            [position for position, _, _ in records], dtype=np.int64
        ),
        "action_factual_source_index": np.asarray(
            [source for _, source, _ in records], dtype=np.int64
        ),
        "action_depth": np.asarray(
            [trace.key[0] for _, _, trace in records], dtype=np.int64
        ),
        "action_parent_id": np.asarray(
            [trace.key[1] for _, _, trace in records], dtype="<U64"
        ),
        "action_unit_id": np.asarray(
            [trace.key[2] for _, _, trace in records], dtype="<U128"
        ),
        "pool_id": np.asarray(
            [trace.pool_id for _, _, trace in records], dtype="<U64"
        ),
        "policy": np.asarray(
            [trace.policy for _, _, trace in records], dtype="<U32"
        ),
        "beta": np.asarray(
            [trace.beta for _, _, trace in records], dtype=np.float64
        ),
        "pool_offsets": np.asarray(pool_offsets, dtype=np.int64),
        "selected_offsets": np.asarray(selected_offsets, dtype=np.int64),
        "raw_pool_size": np.asarray(
            [trace.raw_pool_size for _, _, trace in records], dtype=np.int64
        ),
        "unique_pool_size": np.asarray(
            [trace.unique_pool_size for _, _, trace in records], dtype=np.int64
        ),
        "log_normalizer": np.asarray(
            [trace.log_normalizer for _, _, trace in records], dtype=np.float64
        ),
        "effective_sample_size": np.asarray(
            [trace.effective_sample_size for _, _, trace in records],
            dtype=np.float64,
        ),
        "selected_unique_fraction": np.asarray(
            [
                len(np.unique(trace.selected_pool_indices))
                / len(trace.selected_pool_indices)
                for _, _, trace in records
            ],
            dtype=np.float64,
        ),
        "selection_uniforms": concatenate(
            [trace.selection_uniforms for _, _, trace in records], np.float64
        ),
        "raw_target_probabilities": concatenate(
            [trace.raw_target_probabilities for _, _, trace in records],
            np.float64,
        ),
        "raw_weights": concatenate(
            [trace.raw_weights for _, _, trace in records], np.float64
        ),
        "selected_pool_indices": concatenate(
            [trace.selected_pool_indices for _, _, trace in records], np.int64
        ),
    }
def generate_with_backend(
    inputs: TabICLGeneratorInputs,
    *,
    discriminator: DiscriminatorProtocol,
    config: CounterContExConfig,
    backend: PreparedBackend,
    seed: int,
    n_counterfactuals: int,
) -> TabICLGeneratorResult:
    """Run retained search using only the portable proposal-backend contract."""
    validate_backend_capabilities(
        backend.capabilities,
        needs_confidence=config.foundation.confidence_quantiles is not None,
        needs_categorical=bool(inputs.categorical_groups),
        needs_joint=config.search.cf_mode == "data_plausible",
        needs_numerical_distribution=config.search.numerical_decoder
        in {"iid", "top-k", "top-p"},
    )
    if (
        config.search.strict_proposal_budget
        and inputs.factual_source_indices is None
    ):
        raise ValueError(
            "strict proposal mode requires stable factual_source_indices"
        )
    source_indices = (
        tuple(range(len(inputs.factuals)))
        if inputs.factual_source_indices is None
        else tuple(int(value) for value in inputs.factual_source_indices)
    )

    samplers: list[_SessionSampler] = []

    def point_backend_factory(
        factual: np.ndarray,
        target: int,
    ) -> TabICLGeneratorPointBackend:
        session = backend.for_factual(factual, target, seed=seed)
        point_backend = _point_backend(
            session,
            discriminator=discriminator,
            use_categorical_distribution=bool(inputs.categorical_groups),
            use_joint_scoring=config.search.cf_mode == "data_plausible",
            config=config,
            seed=seed,
            factual_source_index=source_indices[len(samplers)],
            target=target,
        )
        samplers.append(point_backend.sampler)
        return point_backend

    result = generate_counterfactual_batch(
        inputs,
        discriminator=discriminator,
        config=config.generator_config(n_counterfactuals, seed=seed),
        point_backend_factory=point_backend_factory,
    )
    if not config.search.strict_proposal_budget:
        return result
    raw_count = np.asarray(
        [sampler.accounting.raw_count for sampler in samplers], dtype=np.int64
    )
    projected_count = np.asarray(
        [sampler.accounting.projected_count for sampler in samplers], dtype=np.int64
    )
    unique_count = np.asarray(
        [sampler.accounting.unique_count for sampler in samplers], dtype=np.int64
    )
    no_op_count = np.asarray(
        [sampler.accounting.no_op_count for sampler in samplers], dtype=np.int64
    )
    duplicate_count = np.asarray(
        [sampler.accounting.duplicate_count for sampler in samplers], dtype=np.int64
    )
    classifier_rows = np.asarray(
        [sampler.accounting.classifier_rows for sampler in samplers], dtype=np.int64
    )
    classifier_calls = np.asarray(
        [sampler.accounting.classifier_calls for sampler in samplers], dtype=np.int64
    )
    classifier_calls_to_first = np.asarray(
        [
            -1
            if sampler.accounting.first_validity_classifier_calls is None
            else sampler.accounting.first_validity_classifier_calls
            for sampler in samplers
        ],
        dtype=np.int64,
    )
    classifier_rows_to_first = np.asarray(
        [
            -1
            if sampler.accounting.first_validity_classifier_rows is None
            else sampler.accounting.first_validity_classifier_rows
            for sampler in samplers
        ],
        dtype=np.int64,
    )
    tabicl_calls = np.asarray(
        [sampler.accounting.tabicl_calls for sampler in samplers], dtype=np.int64
    )
    tabicl_rows = np.asarray(
        [sampler.accounting.tabicl_rows for sampler in samplers], dtype=np.int64
    )
    terminal_arrays = [sampler.accounting.terminal_arrays() for sampler in samplers]
    raw_offsets = np.concatenate(
        (np.array([0], dtype=np.int64), np.cumsum(raw_count, dtype=np.int64))
    )
    classifier_offsets = np.concatenate(
        (
            np.array([0], dtype=np.int64),
            np.cumsum(classifier_rows, dtype=np.int64),
        )
    )
    dispositions = (
        np.concatenate([item[0] for item in terminal_arrays])
        if terminal_arrays
        else np.empty(0, dtype=np.int8)
    )
    reverse_parts: list[np.ndarray] = []
    for point_index, item in enumerate(terminal_arrays):
        local = item[1].copy()
        local[local >= 0] += classifier_offsets[point_index]
        reverse_parts.append(local)
    reverse = (
        np.concatenate(reverse_parts) if reverse_parts else np.empty(0, dtype=np.int64)
    )
    scored_rows = (
        np.concatenate([item[2] for item in terminal_arrays], axis=0)
        if terminal_arrays
        else np.empty((0, inputs.factuals.shape[1]), dtype=np.float64)
    )
    target_probabilities = (
        np.concatenate([item[3] for item in terminal_arrays])
        if terminal_arrays
        else np.empty(0, dtype=np.float64)
    )
    diagnostics = replace(
        result.diagnostics,
        proposal_policy=(
            f"{config.search.numerical_decoder}/"
            f"{config.search.categorical_decoder}"
        ),
        proposal_raw_count_per_point=raw_count,
        proposal_projected_count_per_point=projected_count,
        proposal_unique_count_per_point=unique_count,
        proposal_no_op_count_per_point=no_op_count,
        proposal_duplicate_count_per_point=duplicate_count,
        proposal_classifier_rows_per_point=classifier_rows,
        proposal_classifier_calls_per_point=classifier_calls,
        proposal_classifier_calls_to_first_validity_per_point=(
            classifier_calls_to_first
        ),
        proposal_classifier_rows_to_first_validity_per_point=(
            classifier_rows_to_first
        ),
        proposal_tabicl_calls_per_point=tabicl_calls,
        proposal_tabicl_rows_per_point=tabicl_rows,
        proposal_terminal_dispositions=dispositions,
        proposal_unique_index_by_raw=reverse,
        proposal_raw_offsets=raw_offsets,
        proposal_classifier_row_offsets=classifier_offsets,
        proposal_scored_rows=scored_rows,
        proposal_scored_target_probabilities=target_probabilities,
        proposal_trace_arrays=_proposal_trace_arrays(samplers),
        proposal_guidance_trace_arrays=_guidance_trace_arrays(samplers),
    )
    return replace(result, diagnostics=diagnostics)
