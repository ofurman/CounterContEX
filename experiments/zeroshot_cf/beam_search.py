"""Task-guided beam search for *from-scratch* counterfactual generation (Exp 4).

Unlike Exp 2/3 (imputation: freeze immutables, mask only actionables), this module
generates **every** feature of a counterfactual autoregressively, conditioning only
on Y=target. The factual instance enters solely through a per-feature *proximity*
penalty — it is never observed during generation.

Mechanism
---------
The autoregressive chain is reimplemented directly on top of the local
``TabPFNRegressor`` conditional-density API (``predict(output_type="full")`` →
``criterion`` + ``logits``), rather than ``TabPFNUnsupervisedModel.impute``,
because beam search needs explicit control over the per-step branch point.

For a fixed feature ordering ``f_1 … f_D`` (Y appended as the last context column):

  for k in 1..D:
      observed = [y_idx] + ordering[:k-1]                # SAME for every beam & query
      fit reg once:  context[:, observed]  →  context[:, f_k]
      predict on all current partial beams (batched)     → criterion, logits
      for each beam:
          candidates = {icdf(logits, q) : q in probs} ∪ {mode}   # K spread values
          drop candidates ∉ [0, 1]                       # hard OOB rejection
          step_score(c) = log p(c)  −  λ_k · |c − factual_k|
      keep the top-`beam_width` extensions per query point
  rerank the B complete beams per query: prefer validity (disc==target),
      tie-break by cumulative beam score (log-density − proximity).

The key efficiency property: with a *fixed* ordering, at step k every beam (and
every query point) shares the same observed-column set, so the regressor is fit
once per step and all (query × beam) partial rows are predicted in one batch.
Total cost ≈ D fits + D batched predicts.

λ is per-feature: large on immutable columns (soft-freeze — they are still
generated, but strongly pulled to the factual value) and tunable on actionables.

All public functions accept/return numpy float64 arrays (cel metrics contract).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Generation ordering
# ---------------------------------------------------------------------------


def build_generation_ordering(
    n_features: int,
    immutable_idx: Sequence[int],
    actionable_order: Optional[Sequence[int]] = None,
) -> List[int]:
    """Build a full-feature generation order: immutables first, then actionables.

    Immutables are generated early so the (high-λ, near-factual) values they take
    become rich conditioning context for the actionable features that follow.

    Args:
        n_features: Number of original features d (every one is generated).
        immutable_idx: Immutable column indices.
        actionable_order: Optional explicit order for the actionable columns
            (e.g. by descending |discriminator coef|). When None, actionables are
            taken in ascending index order.

    Returns:
        A permutation of range(n_features): immutables (ascending) followed by
        actionables in ``actionable_order``.
    """
    immut = list(immutable_idx)
    immut_set = set(immut)
    if actionable_order is None:
        actionable = [c for c in range(n_features) if c not in immut_set]
    else:
        actionable = [c for c in actionable_order if c not in immut_set]
        # Defensively append any actionable not covered by the explicit order.
        covered = set(actionable) | immut_set
        actionable += [c for c in range(n_features) if c not in covered]
    ordering = sorted(immut) + actionable
    assert sorted(ordering) == list(range(n_features)), (
        f"ordering must be a permutation of 0..{n_features - 1}, got {ordering}"
    )
    return ordering


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class BeamConfig:
    """Hyper-parameters for task-guided from-scratch beam search."""

    beam_width: int = 8
    n_candidates: int = 6  # candidate values branched per beam per step
    lambda_actionable: float = 1.0
    lambda_immutable: float = 100.0
    max_context: int = 256
    # Quantile probabilities used to spread the K-1 icdf candidates; the K-th
    # candidate is always the distribution mode. When None, an evenly-spaced
    # interior grid of (n_candidates - 1) probabilities is used.
    candidate_probs: Optional[Sequence[float]] = None
    random_state: int = 42

    def probs(self) -> List[float]:
        if self.candidate_probs is not None:
            return list(self.candidate_probs)
        k = max(1, self.n_candidates - 1)  # reserve one slot for the mode
        # interior grid avoiding the extreme tails (which are most often OOB)
        return [(i + 1) / (k + 1) for i in range(k)]


# ---------------------------------------------------------------------------
# Per-step candidate generation + scoring
# ---------------------------------------------------------------------------


def _candidates_and_logpdf(
    criterion,
    logits: torch.Tensor,
    probs: Sequence[float],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (candidates, logpdf) tensors of shape (n_rows, K).

    Candidates are the ``icdf`` values at ``probs`` plus the distribution mode.
    ``logpdf`` is the conditional log-density of each candidate under ``logits``.
    """
    dev = criterion.borders.device
    logits = logits.to(dev)

    cols = [criterion.icdf(logits, float(p)) for p in probs]  # each (n_rows,)
    cols.append(criterion.mode(logits))  # mode candidate
    cand = torch.stack(cols, dim=1)  # (n_rows, K)

    # log p(c) = -NLL(c); forward expects y broadcastable to logits.shape[:-1].
    logpdf_cols = []
    for j in range(cand.shape[1]):
        nll = criterion.forward(logits, cand[:, j].contiguous())
        logpdf_cols.append(-nll)
    logpdf = torch.stack(logpdf_cols, dim=1)  # (n_rows, K)
    return cand, logpdf


# ---------------------------------------------------------------------------
# Beam search
# ---------------------------------------------------------------------------


@dataclass
class _Beams:
    """Vectorized beam state across all query points.

    Each row is one partial counterfactual. ``group`` maps it to its query index.
    Columns of ``rows`` are the augmented matrix [d original features | Y].
    """

    rows: np.ndarray  # (R, d+1) float64; unfilled original cols = NaN, Y col set
    group: np.ndarray  # (R,) int — query index each beam belongs to
    score: np.ndarray  # (R,) float — cumulative (log-density − proximity)
    logdens: np.ndarray  # (R,) float — cumulative log-density only (plausibility)


def generate_cf_beam(
    reg,
    X_context: np.ndarray,
    y_context: np.ndarray,
    X_factual: np.ndarray,
    target_class: int,
    ordering: Sequence[int],
    immutable_idx: Sequence[int],
    config: Optional[BeamConfig] = None,
    disc_model=None,
    freeze_immutable: bool = False,
) -> Tuple[np.ndarray, Dict]:
    """Generate counterfactuals via task-guided beam search.

    Two regimes (see ``freeze_immutable``):

    - **from scratch** (default): *every* feature is generated; immutables are still
      generated but soft-pulled to the factual via ``lambda_immutable``.
    - **frozen-immutable** (``freeze_immutable=True``): immutables are *observed*
      (held at the factual value, never generated) and the beam generates only the
      actionable features — directly comparable to the Exp 2/3 imputation baseline,
      with ``true_actionability == 1.0`` by construction.

    Args:
        reg: A ``TabPFNRegressor`` (from ``checkpoints.get_models``). Re-fit per step.
        X_context: (n_ctx, d) conditioning rows. Should be *all-classes* training
            rows so the appended Y column is informative (a constant Y in context
            triggers TabPFN's constant-feature validator).
        y_context: (n_ctx,) labels for the context rows.
        X_factual: (m, d) factual rows to explain. Used for the proximity penalty and
            final metrics; in ``freeze_immutable`` mode its immutable columns are also
            copied into the (observed) immutable columns of every beam.
        target_class: Desired class for all m counterfactuals (call once per
            target class, mirroring exp2's per-class batching).
        ordering: Generation order. In from-scratch mode a full permutation of all d
            features; in frozen mode any immutable entries are skipped (immutables are
            observed, not generated).
        immutable_idx: Immutable columns. Soft-pulled (from scratch) or held observed
            (frozen).
        config: Beam hyper-parameters.
        disc_model: Optional validity oracle (``.predict``) used for the terminal
            rerank. When None, beams are ranked by cumulative score only.
        freeze_immutable: If True, observe immutables and generate only actionables.

    Returns:
        (X_cf, aux) where X_cf is (m, d) float64 and aux carries per-row diagnostics.
    """
    cfg = config or BeamConfig()
    rng = np.random.default_rng(cfg.random_state)
    torch.manual_seed(cfg.random_state)

    Xc = np.asarray(X_context, dtype=np.float64)
    yc = np.asarray(y_context, dtype=np.float64)
    Xf = np.asarray(X_factual, dtype=np.float64)
    m, d = Xf.shape
    y_idx = d  # appended Y column index in the augmented matrix
    immut_list = sorted(int(i) for i in immutable_idx)
    immut_set = set(immut_list)
    probs = cfg.probs()

    if freeze_immutable:
        # Immutables are observed parents at every step; only actionables generated.
        gen_order = [int(f) for f in ordering if int(f) not in immut_set]
        base_observed = [y_idx] + immut_list
    else:
        gen_order = [int(f) for f in ordering]
        base_observed = [y_idx]

    # Subsample context deterministically (all classes; Y must vary).
    if len(Xc) > cfg.max_context:
        idx = rng.choice(len(Xc), size=cfg.max_context, replace=False)
        idx.sort()
        Xc, yc = Xc[idx], yc[idx]
    ctx_aug = np.concatenate([Xc, yc.reshape(-1, 1)], axis=1)  # (n_ctx, d+1)

    # Initial beams: one per query, generated features NaN, Y = target.
    # In frozen mode, immutable columns are pre-filled with the factual values.
    init_rows = np.full((m, d + 1), np.nan, dtype=np.float64)
    init_rows[:, y_idx] = float(target_class)
    if freeze_immutable and immut_list:
        init_rows[:, immut_list] = Xf[:, immut_list]
    beams = _Beams(
        rows=init_rows,
        group=np.arange(m, dtype=np.int64),
        score=np.zeros(m, dtype=np.float64),
        logdens=np.zeros(m, dtype=np.float64),
    )

    n_oob_fallback = 0

    for step, f in enumerate(gen_order):
        observed = base_observed + gen_order[:step]  # augmented indices
        lam = cfg.lambda_immutable if f in immut_set else cfg.lambda_actionable

        # Fit the conditional density p(f | observed) once on the shared context.
        reg.fit(ctx_aug[:, observed], ctx_aug[:, f])
        out = reg.predict(beams.rows[:, observed], output_type="full")
        criterion = out["criterion"]
        logits = torch.as_tensor(out["logits"])

        cand, logpdf = _candidates_and_logpdf(criterion, logits, probs)
        cand_np = cand.detach().cpu().numpy().astype(np.float64)  # (R, K)
        logpdf_np = logpdf.detach().cpu().numpy().astype(np.float64)  # (R, K)
        R, K = cand_np.shape

        factual_f = Xf[beams.group, f]  # (R,)
        prox = np.abs(cand_np - factual_f[:, None])  # (R, K)
        step_score = logpdf_np - lam * prox  # (R, K)

        # Hard out-of-[0,1] rejection. If every candidate for a beam is OOB,
        # keep its least-bad (clipped) candidate so the beam survives.
        in_bounds = (cand_np >= 0.0) & (cand_np <= 1.0)  # (R, K)
        all_oob = ~in_bounds.any(axis=1)  # (R,)
        if all_oob.any():
            n_oob_fallback += int(all_oob.sum())
            cand_np[all_oob] = np.clip(cand_np[all_oob], 0.0, 1.0)
            in_bounds[all_oob] = True
        step_score = np.where(in_bounds, step_score, -np.inf)

        # Expand: every parent beam → K children (masked where OOB-rejected).
        parent = np.repeat(np.arange(R), K)
        child_val = cand_np.reshape(-1)
        child_score = beams.score[parent] + step_score.reshape(-1)
        child_logdens = beams.logdens[parent] + logpdf_np.reshape(-1)
        child_group = beams.group[parent]
        keep = np.isfinite(child_score)
        parent, child_val = parent[keep], child_val[keep]
        child_score, child_logdens = child_score[keep], child_logdens[keep]
        child_group = child_group[keep]

        # Materialize child rows (copy parent, set feature f).
        child_rows = beams.rows[parent].copy()
        child_rows[:, f] = child_val

        # Prune to top-`beam_width` per query group.
        beams = _prune(
            child_rows, child_group, child_score, child_logdens, cfg.beam_width
        )

    # ---- Terminal rerank: prefer validity, tie-break by cumulative score ----
    X_cf, aux = _select_best(beams, Xf, m, d, target_class, immutable_idx, disc_model)
    aux["n_oob_fallback"] = n_oob_fallback
    return X_cf, aux


def _prune(
    rows: np.ndarray,
    group: np.ndarray,
    score: np.ndarray,
    logdens: np.ndarray,
    beam_width: int,
) -> _Beams:
    """Keep the top-`beam_width` rows per group by score (stable, deterministic).

    Vectorized: a single lexsort by (group asc, score desc, original index asc)
    replaces the per-group ``np.where`` scan, which was O(n_groups x n_rows) and
    dominated runtime on full-split evaluations. Tie-breaking is unchanged —
    equal scores keep ascending original order — as is the ascending row order
    of the returned beams.
    """
    n_rows = len(group)
    if n_rows == 0:
        return _Beams(rows=rows, group=group, score=score, logdens=logdens)

    row_idx = np.arange(n_rows)
    # Primary key is the last: group asc, then -score asc (= score desc), then idx asc.
    order = np.lexsort((row_idx, -score, group))
    g_sorted = group[order]

    # Rank within each group: 0, 1, 2, ... restarting at every group boundary.
    starts = np.r_[0, np.flatnonzero(g_sorted[1:] != g_sorted[:-1]) + 1]
    sizes = np.diff(np.r_[starts, n_rows])
    within_rank = np.arange(n_rows) - np.repeat(starts, sizes)

    keep_idx = np.sort(order[within_rank < beam_width])
    return _Beams(
        rows=rows[keep_idx],
        group=group[keep_idx],
        score=score[keep_idx],
        logdens=logdens[keep_idx],
    )


def _select_best(
    beams: _Beams,
    X_factual: np.ndarray,
    m: int,
    d: int,
    target_class: int,
    immutable_idx: Sequence[int],
    disc_model,
) -> Tuple[np.ndarray, Dict]:
    """Pick one final CF per query: validity first, then cumulative score."""
    cf_rows = beams.rows[:, :d]  # strip Y column
    if disc_model is not None:
        pred = np.asarray(disc_model.predict(cf_rows))
        valid = (pred == target_class).astype(np.float64)
    else:
        valid = np.zeros(len(cf_rows), dtype=np.float64)

    # Validity dominates; cumulative score (density − proximity) breaks ties.
    rank = valid * 1e12 + beams.score

    X_cf = np.empty((m, d), dtype=np.float64)
    chosen_valid = np.zeros(m, dtype=bool)
    chosen_score = np.full(m, np.nan)
    chosen_logdens = np.full(m, np.nan)

    # Vectorized argmax-per-group (see _prune): sort by group asc, rank desc,
    # index asc, then take the first row of each group. This matches the previous
    # ``np.argmax`` semantics, which returned the lowest index among ties.
    n_rows = len(cf_rows)
    if n_rows:
        row_idx = np.arange(n_rows)
        order = np.lexsort((row_idx, -rank, beams.group))
        g_sorted = beams.group[order]
        first_pos = np.r_[0, np.flatnonzero(g_sorted[1:] != g_sorted[:-1]) + 1]
        best_rows = order[first_pos]
        groups_present = g_sorted[first_pos]

        X_cf[groups_present] = cf_rows[best_rows]
        chosen_valid[groups_present] = valid[best_rows] > 0
        chosen_score[groups_present] = beams.score[best_rows]
        chosen_logdens[groups_present] = beams.logdens[best_rows]

    immut = list(immutable_idx)
    if immut:
        drift = np.abs(X_cf[:, immut] - X_factual[:, immut]).mean(axis=1)
    else:
        drift = np.zeros(m)

    aux = {
        "chosen_valid": chosen_valid,
        "chosen_score": chosen_score,
        "chosen_logdens": chosen_logdens,
        "immutable_drift": drift,
    }
    return X_cf, aux
