"""Iterative greedy counterfactual generation (Experiment 4).

Change actionable features **one at a time**, conditioned on all the rest and on
the target class (Y-as-appended-column trick), and **stop at the class flip**.
This directly minimizes L0 sparsity — features are added only until the origin
discriminator's prediction flips — and keeps generation in the dense-conditioning
regime (exactly one masked column per step).

Two candidate-selection strategies share the same loop, the same single-column
near-MAP commit, and the same flip stop condition; they differ only in
``select_candidate``:

  - ``prob_ascent``      — Strategy 1: steepest-ascent on the target-class
                           probability of the discriminator being explained
                           (wrapper / score-driven; SEDC / NICE).
  - ``class_divergence`` — Strategy 2: pick the feature whose class-conditional
                           predictive distribution shifts most between
                           ``Y=y_target`` and ``Y=current`` (TabPFN-intrinsic,
                           classifier-free).

The caller is responsible for fitting the sampler's context (``set_context``)
before calling ``greedy_counterfactual`` — target-class context for Strategy 1,
all-classes context for Strategy 2 (the divergence needs a non-constant Y).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from experiments.zeroshot_cf.sampler import class_conditional_shift


def infer_feature_domains(
    X_train: np.ndarray,
    *,
    max_discrete_values: int = 20,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, np.ndarray]]:
    """Infer training bounds and small empirical supports for projection."""
    X = np.asarray(X_train, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X_train must be 2D, got shape {X.shape}")
    lower = np.nanmin(X, axis=0)
    upper = np.nanmax(X, axis=0)
    supports: Dict[int, np.ndarray] = {}
    for j in range(X.shape[1]):
        values = np.unique(X[:, j][~np.isnan(X[:, j])])
        if 0 < len(values) <= max_discrete_values:
            supports[j] = values
    return lower, upper, supports


def project_candidate_values(
    candidates: List[int],
    values: np.ndarray,
    feature_domains: Tuple[np.ndarray, np.ndarray, Dict[int, np.ndarray]] | None,
) -> np.ndarray:
    """Project candidate values to training bounds and empirical supports."""
    projected = np.asarray(values, dtype=np.float64).copy()
    if feature_domains is None:
        return projected

    lower, upper, supports = feature_domains
    cols = np.asarray(candidates, dtype=int)
    projected = np.clip(projected, lower[cols], upper[cols])
    for position, col in enumerate(cols):
        support = supports.get(int(col))
        if support is not None:
            nearest = int(np.abs(support - projected[position]).argmin())
            projected[position] = support[nearest]
    return projected


def _select_prob_ascent(
    sampler,
    disc,
    x_cf: np.ndarray,
    y_target: int,
    candidates: List[int],
    temperature: float,
    feature_domains=None,
) -> Tuple[int, float, Optional[float]]:
    """Strategy 1: pick the candidate whose near-MAP value most increases
    ``disc.predict_proba[y_target]``. Returns ``(j*, score, value)`` where
    ``value`` is the near-MAP draw already computed for ``j*`` (so the loop need
    not recompute it)."""
    X = x_cf.reshape(1, -1)
    best_j: Optional[int] = None
    best_score = -np.inf
    best_val: Optional[float] = None
    for j in candidates:
        v = float(
            sampler.sample_feature(
                X,
                target_col=j,
                sample_temperature=temperature,
                fixed_target=y_target,
            )[0]
        )
        v = float(project_candidate_values([j], np.array([v]), feature_domains)[0])
        trial = x_cf.copy()
        trial[j] = v
        p = float(disc.predict_proba(trial.reshape(1, -1))[0, y_target])
        if p > best_score:
            best_score = p
            best_j = j
            best_val = v
    return int(best_j), float(best_score), best_val


def _select_prob_ascent_batched(
    sampler,
    disc,
    x_cf: np.ndarray,
    y_target: int,
    candidates: List[int],
    temperature: float,
    feature_domains=None,
) -> Tuple[int, float, Optional[float]]:
    """Strategy 1 with all candidate masks evaluated in one sampler call.

    ``sample_candidates`` returns one imputed value for each candidate feature.
    The corresponding trial counterfactuals are then scored in one discriminator
    call. At deterministic temperature this is semantically equivalent to
    :func:`_select_prob_ascent`, including first-candidate tie breaking.
    """
    values = np.asarray(
        sampler.sample_candidates(
            x_cf.reshape(1, -1),
            candidates,
            sample_temperature=temperature,
            fixed_target=y_target,
        ),
        dtype=np.float64,
    )
    if values.shape != (len(candidates),):
        raise ValueError(
            "sample_candidates must return one value per candidate; "
            f"expected {(len(candidates),)}, got {values.shape}"
        )
    values = project_candidate_values(candidates, values, feature_domains)

    trials = np.repeat(x_cf.reshape(1, -1), len(candidates), axis=0)
    trials[np.arange(len(candidates)), np.asarray(candidates, dtype=int)] = values
    probabilities = np.asarray(disc.predict_proba(trials))[:, y_target]
    best = int(np.argmax(probabilities))
    return (
        int(candidates[best]),
        float(probabilities[best]),
        float(values[best]),
    )


def _select_prob_ascent_quantile_grid(
    sampler,
    disc,
    x_cf: np.ndarray,
    y_target: int,
    candidates: List[int],
    quantiles: Tuple[float, ...],
    feature_domains=None,
    *,
    confidences: Tuple[float, ...] | None = None,
    tau: float = 0.5,
    plausibility_model=None,
    validity_first: bool = False,
    probability_slack: float = 0.02,
) -> Tuple[int, float, Optional[float], Dict]:
    """Score every feature/quantile/(optional) confidence candidate.

    The baseline path preserves maximum target probability. In the
    plausibility-aware path, classifier validity is a hard gate: if any trial
    flips, the lowest-LOF valid trial is selected. Before a direct flip exists,
    LOF selects among trials whose target probability is within
    ``probability_slack`` of the best available progress.
    """
    values = np.asarray(
        sampler.sample_candidate_grid(
            x_cf.reshape(1, -1),
            candidates,
            quantiles=quantiles,
            fixed_target=y_target,
            confidences=confidences,
        ),
        dtype=np.float64,
    )
    n_confidences = 1 if confidences is None else len(confidences)
    expected = (
        (len(candidates), len(quantiles))
        if confidences is None
        else (len(candidates), n_confidences, len(quantiles))
    )
    if values.shape != expected:
        raise ValueError(
            "sample_candidate_grid returned an unexpected shape; "
            f"expected {expected}, got {values.shape}"
        )

    expanded_candidates = np.repeat(
        np.asarray(candidates, dtype=int),
        n_confidences * len(quantiles),
    )
    expanded_quantiles = np.tile(
        np.asarray(quantiles, dtype=np.float64),
        len(candidates) * n_confidences,
    )
    expanded_confidences = None
    if confidences is not None:
        expanded_confidences = np.tile(
            np.repeat(np.asarray(confidences, dtype=np.float64), len(quantiles)),
            len(candidates),
        )
    flat_values = project_candidate_values(
        expanded_candidates.tolist(),
        values.reshape(-1),
        feature_domains,
    )
    trials = np.repeat(x_cf.reshape(1, -1), len(flat_values), axis=0)
    trials[np.arange(len(flat_values)), expanded_candidates] = flat_values
    probabilities = np.asarray(disc.predict_proba(trials))[:, y_target]
    predictions = np.asarray(disc.predict(trials))
    valid = (predictions == y_target) & (probabilities >= tau)

    lof_scores = None
    if plausibility_model is not None:
        lof_scores = -np.asarray(plausibility_model.score_samples(trials))

    if validity_first and valid.any():
        eligible = np.flatnonzero(valid)
        if lof_scores is None:
            best = int(eligible[np.argmax(probabilities[eligible])])
        else:
            best = int(eligible[np.argmin(lof_scores[eligible])])
    elif validity_first and lof_scores is not None:
        best_probability = float(np.max(probabilities))
        eligible = np.flatnonzero(
            probabilities >= best_probability - probability_slack
        )
        best = int(eligible[np.argmin(lof_scores[eligible])])
    else:
        best = int(np.argmax(probabilities))

    metadata = {
        "quantile": float(expanded_quantiles[best]),
        "confidence": (
            None
            if expanded_confidences is None
            else float(expanded_confidences[best])
        ),
        "lof": None if lof_scores is None else float(lof_scores[best]),
        "immediate_valid": bool(valid[best]),
        "n_valid_candidates": int(valid.sum()),
        "n_candidates": int(len(trials)),
    }
    return (
        int(expanded_candidates[best]),
        float(probabilities[best]),
        float(flat_values[best]),
        metadata,
    )


def _select_class_divergence(
    sampler,
    x_cf: np.ndarray,
    y_target: int,
    y_current: int,
    candidates: List[int],
) -> Tuple[int, float, Optional[float]]:
    """Strategy 2: pick the candidate whose class-conditional predictive
    distribution shifts most between ``Y=y_target`` and ``Y=y_current``.
    Classifier-free. Returns ``(j*, divergence, None)`` — the loop draws the
    committed value.

    The per-candidate divergence is delegated to ``class_conditional_shift``,
    which handles BOTH of TabPFN's per-column routings uniformly and in
    comparable [0,1] units: the absolute mean-shift for regressor columns (all
    MOONS features, most HELOC features) and the total-variation distance for
    classifier columns (HELOC's low-cardinality integer features, which
    ``infer_categorical_features`` routes to the classifier head — reaching into
    ``dist["logits"]`` directly used to KeyError on those). Features are
    MinMax-[0,1] so no extra normalization is needed.
    """
    X = x_cf.reshape(1, -1)
    best_j: Optional[int] = None
    best_div = -np.inf
    for j in candidates:
        dist_tgt = sampler.predictive_distribution(
            X, target_col=j, fixed_target=y_target
        )
        dist_cur = sampler.predictive_distribution(
            X, target_col=j, fixed_target=y_current
        )
        div = float(class_conditional_shift(dist_tgt, dist_cur)[0])
        if div > best_div:
            best_div = div
            best_j = j
    return int(best_j), float(best_div), None


def greedy_counterfactual(
    sampler,
    disc,
    x: np.ndarray,
    y_target: int,
    actionable_idx: List[int],
    selector: str,
    *,
    tau: float = 0.5,
    budget: Optional[int] = None,
    temperature: float = 1e-9,
    max_rounds: int = 1,
    batch_candidates: bool = False,
    feature_domains=None,
    retain_best: bool = False,
    candidate_quantiles: Tuple[float, ...] | None = None,
    candidate_confidences: Tuple[float, ...] | None = None,
    plausibility_model=None,
    validity_first: bool = False,
    probability_slack: float = 0.02,
) -> Tuple[np.ndarray, List[int], Dict]:
    """Greedily build a counterfactual for one factual point.

    Parameters
    ----------
    sampler : ConditionalDensitySampler
        Must already have ``set_context`` called (caller's responsibility) with
        ``append_target=True``. Target-class context for ``prob_ascent``,
        all-classes context for ``class_divergence``.
    disc : DiscriminatorModel
        Validity oracle whose flip is the stop condition.
    x : ndarray of shape (d,)
        The factual point (MinMax-[0,1] feature space).
    y_target : int
        Target class (= ``1 - disc.predict(x)`` for the binary tasks here).
    actionable_idx : list of int
        Columns that may be changed. Immutable columns are never candidates, so
        they stay byte-identical to ``x`` by construction.
    selector : {"prob_ascent", "class_divergence"}
        Candidate-selection strategy.
    tau : float
        Probability threshold for the flip: stop when ``predict == y_target``
        AND ``predict_proba[y_target] >= tau``. Default 0.5 ≡ hard flip.
    budget : int or None
        Max number of features to change **per round**. Defaults to
        ``len(actionable_idx)``.
    temperature : float
        Sampling temperature for the committed value. ``1e-9`` = near-MAP
        (deterministic single-column commit).
    max_rounds : int
        Number of greedy passes over the actionable columns. Within a round
        each column may be edited at most once (the original constraint);
        ``max_rounds=1`` (default) is byte-identical to the single-pass
        behaviour. In rounds >= 2 a column edited in an earlier round becomes
        eligible again — its re-draw is conditioned on the *current* ``x_cf``
        (the other columns have moved since it was set), so repeated rounds
        are coordinate ascent toward the target-class conditional mode.
        Two guards apply in rounds >= 2 only:
        (a) an edit is committed only if it **strictly increases**
        ``predict_proba[y_target]`` — for ``prob_ascent`` no candidate can
        beat the argmax, so a non-improving argmax ends the round; and
        (b) if a round commits nothing, the loop stops early — no
        single-column near-MAP edit improves ``p_target``, i.e. a fixed
        point, and at near-zero temperature further rounds are no-ops.
    batch_candidates : bool
        If True, evaluate all remaining ``prob_ascent`` candidates through one
        ``sampler.sample_candidates`` call per greedy step. This preserves the
        search rule while avoiding one model call per candidate. It has no
        effect on ``class_divergence``.
    candidate_quantiles : tuple of float or None
        When provided with batched ``prob_ascent``, obtain several deterministic
        conditional quantiles per feature and score every feature/value pair.
        ``None`` preserves the single point-estimate path.

    Returns
    -------
    (x_cf, changed, info)
        ``x_cf`` — the counterfactual (ndarray, shape (d,)).
        ``changed`` — **distinct** changed column indices in first-touch order
        (L0 = ``len(changed)``; a column re-edited in a later round is not
        repeated).
        ``info`` — dict with ``flipped`` (bool), ``steps`` (int = total edits
        committed, >= ``len(changed)`` when ``max_rounds > 1``), ``rounds``
        (int = rounds entered), and ``history`` (per-edit list of
        ``(feature_idx, value, p_target_after, selection_score)``).
    """
    x = np.asarray(x, dtype=np.float64).copy()
    n_features = x.shape[0]
    actionable = list(actionable_idx)
    if budget is None:
        budget = len(actionable)
    if max_rounds < 1:
        raise ValueError(f"max_rounds must be >= 1, got {max_rounds}")
    if candidate_quantiles is not None:
        candidate_quantiles = tuple(float(q) for q in candidate_quantiles)
        if selector != "prob_ascent" or not batch_candidates:
            raise ValueError(
                "candidate_quantiles require batched prob_ascent selection"
            )
    if candidate_confidences is not None:
        candidate_confidences = tuple(float(c) for c in candidate_confidences)
        if candidate_quantiles is None:
            raise ValueError(
                "candidate_confidences require candidate_quantiles"
            )
    if probability_slack < 0:
        raise ValueError("probability_slack must be non-negative")
    y_current = 1 - int(y_target)  # binary task

    x_cf = x.copy()
    changed: List[int] = []  # distinct columns, first-touch order (= L0)
    history: List[tuple] = []
    selection_history: List[Dict] = []
    total_edits = 0
    rounds_used = 0

    def _flip_state(row: np.ndarray) -> Tuple[bool, float]:
        rr = row.reshape(1, -1)
        pred = int(disc.predict(rr)[0])
        p_t = float(disc.predict_proba(rr)[0, y_target])
        return (pred == y_target and p_t >= tau), p_t

    flipped, p_t = _flip_state(x_cf)
    best_x_cf = x_cf.copy()
    best_changed = changed.copy()
    best_p_t = p_t
    best_steps = total_edits
    best_history_length = 0
    for rnd in range(max_rounds):
        if flipped:
            break
        rounds_used = rnd + 1
        edited_this_round: List[int] = []

        while not flipped and len(edited_this_round) < budget:
            candidates = [j for j in actionable if j not in edited_this_round]
            if not candidates:
                break

            selection_metadata: Dict = {}
            if selector == "prob_ascent":
                if candidate_quantiles is not None:
                    j_star, score, val, selection_metadata = (
                        _select_prob_ascent_quantile_grid(
                            sampler,
                            disc,
                            x_cf,
                            y_target,
                            candidates,
                            candidate_quantiles,
                            feature_domains,
                            confidences=candidate_confidences,
                            tau=tau,
                            plausibility_model=plausibility_model,
                            validity_first=validity_first,
                            probability_slack=probability_slack,
                        )
                    )
                else:
                    select = (
                        _select_prob_ascent_batched
                        if batch_candidates
                        else _select_prob_ascent
                    )
                    j_star, score, val = select(
                        sampler,
                        disc,
                        x_cf,
                        y_target,
                        candidates,
                        temperature,
                        feature_domains,
                    )
            elif selector == "class_divergence":
                j_star, score, val = _select_class_divergence(
                    sampler, x_cf, y_target, y_current, candidates
                )
            else:
                raise ValueError(
                    f"Unknown selector {selector!r}; expected 'prob_ascent' or "
                    "'class_divergence'."
                )

            if val is None:
                val = float(
                    sampler.sample_feature(
                        x_cf.reshape(1, -1),
                        target_col=j_star,
                        sample_temperature=temperature,
                        fixed_target=y_target,
                    )[0]
                )
                val = float(
                    project_candidate_values(
                        [j_star], np.array([val]), feature_domains
                    )[0]
                )

            if rnd > 0:
                # Strict-improvement acceptance in re-visit rounds. For
                # prob_ascent ``score`` already is p_target after the trial
                # edit; for class_divergence it is a divergence, so p must be
                # probed with one extra disc call.
                if selector == "prob_ascent":
                    p_trial = score
                else:
                    trial = x_cf.copy()
                    trial[j_star] = val
                    p_trial = float(
                        disc.predict_proba(trial.reshape(1, -1))[0, y_target]
                    )
                if p_trial <= p_t:
                    break  # end this round; round-level guard decides the rest

            x_cf[j_star] = val
            edited_this_round.append(j_star)
            if j_star not in changed:
                changed.append(j_star)
            total_edits += 1
            flipped, p_t = _flip_state(x_cf)
            history.append((j_star, val, p_t, score))
            selection_history.append(selection_metadata)
            if p_t > best_p_t:
                best_x_cf = x_cf.copy()
                best_changed = changed.copy()
                best_p_t = p_t
                best_steps = total_edits
                best_history_length = len(history)

        if rnd > 0 and not edited_this_round:
            break  # fixed point: no single-column edit improves p_target

    attempt_history = history.copy()
    attempt_selection_history = selection_history.copy()
    if retain_best and not flipped:
        x_cf = best_x_cf
        changed = best_changed
        p_t = best_p_t
        total_edits = best_steps
        history = history[:best_history_length]
        selection_history = selection_history[:best_history_length]

    # Immutability assert (extends the predecessor Stage-7 check): every
    # non-actionable column must be byte-identical to the factual.
    non_actionable = [c for c in range(n_features) if c not in actionable]
    if non_actionable:
        cols = np.asarray(non_actionable)
        assert np.array_equal(x_cf[cols], x[cols]), (
            "Non-actionable columns drifted during greedy generation — "
            "immutables must be preserved exactly by construction."
        )

    info = {
        "flipped": bool(flipped),
        "steps": total_edits,
        "attempt_steps": len(attempt_history),
        "rounds": rounds_used,
        "history": history,
        "attempt_history": attempt_history,
        "best_target_probability": float(best_p_t),
        "batch_candidates": bool(batch_candidates),
        "candidate_quantiles": candidate_quantiles,
        "candidate_confidences": candidate_confidences,
        "selection_history": selection_history,
        "attempt_selection_history": attempt_selection_history,
        "validity_first": bool(validity_first),
        "probability_slack": float(probability_slack),
        "retain_best": bool(retain_best),
    }
    return x_cf, changed, info
