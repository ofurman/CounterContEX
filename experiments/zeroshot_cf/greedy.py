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


def _select_prob_ascent(
    sampler,
    disc,
    x_cf: np.ndarray,
    y_target: int,
    candidates: List[int],
    temperature: float,
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
        trial = x_cf.copy()
        trial[j] = v
        p = float(disc.predict_proba(trial.reshape(1, -1))[0, y_target])
        if p > best_score:
            best_score = p
            best_j = j
            best_val = v
    return int(best_j), float(best_score), best_val


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
        dist_tgt = sampler.predictive_distribution(X, target_col=j, fixed_target=y_target)
        dist_cur = sampler.predictive_distribution(X, target_col=j, fixed_target=y_current)
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
    y_current = 1 - int(y_target)  # binary task

    x_cf = x.copy()
    changed: List[int] = []  # distinct columns, first-touch order (= L0)
    history: List[tuple] = []
    total_edits = 0
    rounds_used = 0

    def _flip_state(row: np.ndarray) -> Tuple[bool, float]:
        rr = row.reshape(1, -1)
        pred = int(disc.predict(rr)[0])
        p_t = float(disc.predict_proba(rr)[0, y_target])
        return (pred == y_target and p_t >= tau), p_t

    flipped, p_t = _flip_state(x_cf)
    for rnd in range(max_rounds):
        if flipped:
            break
        rounds_used = rnd + 1
        edited_this_round: List[int] = []

        while not flipped and len(edited_this_round) < budget:
            candidates = [j for j in actionable if j not in edited_this_round]
            if not candidates:
                break

            if selector == "prob_ascent":
                j_star, score, val = _select_prob_ascent(
                    sampler, disc, x_cf, y_target, candidates, temperature
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

        if rnd > 0 and not edited_this_round:
            break  # fixed point: no single-column edit improves p_target

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
        "rounds": rounds_used,
        "history": history,
    }
    return x_cf, changed, info
