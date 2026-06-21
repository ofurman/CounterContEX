"""Offline unit tests for task-guided from-scratch beam search (Stage 10 / Exp 4).

These tests use a real ``FullSupportBarDistribution`` (from TabPFN core) wired to a
tiny fake regressor, so they run with **no checkpoint and no network** — only the
beam-search control flow and scoring are exercised, not TabPFN inference quality.

Covered:
  1. build_generation_ordering — permutation, immutables-first, explicit order.
  2. _candidates_and_logpdf — shape (R, K), finite log-density, mode included.
  3. generate_cf_beam — output shape, all features filled, in-bounds when the
     distribution is in [0,1], non-immutable validity rerank.
  4. Immutable soft-freeze — higher lambda_immutable ⇒ smaller immutable drift.
  5. OOB rejection — a distribution wholly outside [0,1] triggers clip-fallback
     and is counted in n_oob_fallback.
"""

from __future__ import annotations

import numpy as np
import torch

from tabpfn.architectures.base.bar_distribution import FullSupportBarDistribution

from experiments.zeroshot_cf.beam_search import (
    BeamConfig,
    build_generation_ordering,
    generate_cf_beam,
    _candidates_and_logpdf,
)


# ---------------------------------------------------------------------------
# Fake regressor returning a fixed conditional density
# ---------------------------------------------------------------------------


class _FakeReg:
    """Mimics TabPFNRegressor.predict(output_type='full') with a fixed distribution.

    The same peaked ``FullSupportBarDistribution`` is returned at every step,
    which is all the beam-search mechanics need (candidate generation, scoring,
    pruning, rerank). The borders control whether candidates fall in [0, 1].
    """

    def __init__(
        self, lo: float = 0.0, hi: float = 1.0, n_bars: int = 20, peak: int = 10
    ):
        borders = torch.linspace(lo, hi, n_bars + 1)
        self.criterion = FullSupportBarDistribution(borders)
        base = torch.zeros(n_bars)
        base[peak] = 4.0  # concentrate mass near one bucket
        self._base = base

    def fit(self, X, y):  # noqa: D401 - context is ignored on purpose
        return self

    def predict(self, X, output_type="full"):
        n = len(X)
        logits = self._base.unsqueeze(0).repeat(n, 1)
        return {"criterion": self.criterion, "logits": logits}


class _FakeDisc:
    """Validity oracle: predicts class 1 iff feature 0 >= threshold, else 0."""

    def __init__(self, col: int = 0, threshold: float = 0.5):
        self.col, self.threshold = col, threshold

    def predict(self, X):
        return (np.asarray(X)[:, self.col] >= self.threshold).astype(np.int64)


# ---------------------------------------------------------------------------
# 1. Ordering
# ---------------------------------------------------------------------------


def test_ordering_is_permutation_immutables_first():
    order = build_generation_ordering(5, immutable_idx=[3, 1])
    assert sorted(order) == [0, 1, 2, 3, 4]
    assert order[:2] == [1, 3], "immutables must come first, ascending"
    assert set(order[2:]) == {0, 2, 4}


def test_ordering_respects_explicit_actionable_order():
    order = build_generation_ordering(
        5, immutable_idx=[0], actionable_order=[4, 2, 3, 1]
    )
    assert order[0] == 0  # immutable first
    assert order[1:] == [4, 2, 3, 1]


def test_ordering_no_immutables():
    order = build_generation_ordering(3, immutable_idx=[])
    assert order == [0, 1, 2]


# ---------------------------------------------------------------------------
# 2. Candidate generation + scoring
# ---------------------------------------------------------------------------


def test_candidates_shape_and_finite():
    reg = _FakeReg()
    out = reg.predict(np.zeros((4, 2)))
    probs = BeamConfig(n_candidates=6).probs()
    cand, logpdf = _candidates_and_logpdf(out["criterion"], out["logits"], probs)
    assert cand.shape == (4, 6)
    assert logpdf.shape == (4, 6)
    assert torch.all(torch.isfinite(cand))
    assert torch.all(torch.isfinite(logpdf))


# ---------------------------------------------------------------------------
# 3. End-to-end beam generation
# ---------------------------------------------------------------------------


def _toy_context(n=64, d=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, size=(n, d))
    y = (X[:, 0] > 0.5).astype(np.int64)
    return X, y


def test_generate_shape_filled_and_in_bounds():
    reg = _FakeReg(lo=0.0, hi=1.0)
    Xc, yc = _toy_context()
    Xf = np.random.default_rng(1).uniform(0, 1, size=(5, 3))
    ordering = build_generation_ordering(3, immutable_idx=[0])
    cfg = BeamConfig(beam_width=4, n_candidates=5, max_context=64)

    X_cf, aux = generate_cf_beam(
        reg,
        Xc,
        yc,
        Xf,
        target_class=1,
        ordering=ordering,
        immutable_idx=[0],
        config=cfg,
    )
    assert X_cf.shape == (5, 3)
    assert not np.any(np.isnan(X_cf)), "every feature must be generated (no NaN)"
    assert X_cf.min() >= 0.0 and X_cf.max() <= 1.0, "in-[0,1] dist ⇒ in-bounds CFs"
    assert aux["n_oob_fallback"] == 0
    assert aux["immutable_drift"].shape == (5,)


def test_validity_rerank_prefers_target_class():
    # Distribution spans [0,1]; disc says class 1 iff feature0 >= 0.5.
    reg = _FakeReg(lo=0.0, hi=1.0)
    Xc, yc = _toy_context()
    Xf = np.full((4, 3), 0.2)  # factuals near 0 → class 0
    ordering = build_generation_ordering(3, immutable_idx=[])
    disc = _FakeDisc(col=0, threshold=0.5)
    # Low proximity weight so validity (not proximity) drives feature 0.
    cfg = BeamConfig(beam_width=6, n_candidates=6, lambda_actionable=0.01)

    X_cf, aux = generate_cf_beam(
        reg,
        Xc,
        yc,
        Xf,
        target_class=1,
        ordering=ordering,
        immutable_idx=[],
        config=cfg,
        disc_model=disc,
    )
    # At least one beam per row reaches feature0 >= 0.5 (mode≈0.5..0.55 in range),
    # so the rerank should mark them valid.
    assert aux["chosen_valid"].all(), "valid target-class beams should be selected"
    assert (disc.predict(X_cf) == 1).all()


# ---------------------------------------------------------------------------
# 4. Immutable soft-freeze
# ---------------------------------------------------------------------------


def test_higher_lambda_immutable_reduces_drift():
    reg = _FakeReg(lo=0.0, hi=1.0)
    Xc, yc = _toy_context()
    rng = np.random.default_rng(7)
    Xf = rng.uniform(0, 1, size=(8, 3))
    ordering = build_generation_ordering(3, immutable_idx=[0, 1])

    common = dict(beam_width=6, n_candidates=6, max_context=64)
    _, aux_low = generate_cf_beam(
        reg,
        Xc,
        yc,
        Xf,
        1,
        ordering,
        [0, 1],
        BeamConfig(lambda_immutable=0.0, lambda_actionable=0.0, **common),
    )
    _, aux_high = generate_cf_beam(
        reg,
        Xc,
        yc,
        Xf,
        1,
        ordering,
        [0, 1],
        BeamConfig(lambda_immutable=100.0, lambda_actionable=0.0, **common),
    )
    assert aux_high["immutable_drift"].mean() < aux_low["immutable_drift"].mean(), (
        "high lambda_immutable must pull immutable columns closer to factual"
    )


# ---------------------------------------------------------------------------
# 4b. Frozen-immutable mode (Set 1): immutables observed, held exactly at factual
# ---------------------------------------------------------------------------


def test_freeze_immutable_holds_immutables_exactly():
    reg = _FakeReg(lo=0.0, hi=1.0)
    Xc, yc = _toy_context(d=4)
    rng = np.random.default_rng(11)
    Xf = rng.uniform(0, 1, size=(6, 4))
    immutable_idx = [0, 2]
    ordering = build_generation_ordering(4, immutable_idx=immutable_idx)
    cfg = BeamConfig(beam_width=4, n_candidates=5, max_context=64)

    X_cf, aux = generate_cf_beam(
        reg, Xc, yc, Xf, 1, ordering, immutable_idx, cfg, freeze_immutable=True
    )
    # Immutable columns must be byte-identical to the factual (true_actionability=1.0).
    np.testing.assert_array_equal(X_cf[:, immutable_idx], Xf[:, immutable_idx])
    assert aux["immutable_drift"].max() == 0.0
    # Actionable columns are still generated (in [0,1]).
    actionable = [1, 3]
    assert X_cf[:, actionable].min() >= 0.0 and X_cf[:, actionable].max() <= 1.0


# ---------------------------------------------------------------------------
# 5. Out-of-bounds rejection / fallback
# ---------------------------------------------------------------------------


def test_oob_distribution_triggers_clip_fallback():
    # Distribution entirely above 1.0 → every candidate is OOB → clip fallback.
    reg = _FakeReg(lo=1.5, hi=2.5)
    Xc, yc = _toy_context()
    Xf = np.full((3, 3), 0.5)
    ordering = build_generation_ordering(3, immutable_idx=[])
    cfg = BeamConfig(beam_width=4, n_candidates=5, max_context=64)

    X_cf, aux = generate_cf_beam(
        reg,
        Xc,
        yc,
        Xf,
        1,
        ordering,
        [],
        cfg,
    )
    assert aux["n_oob_fallback"] > 0, "all-OOB candidates must hit the fallback path"
    assert X_cf.max() <= 1.0 and X_cf.min() >= 0.0, "fallback must clip into [0,1]"
    assert np.allclose(X_cf, 1.0), "clip of >1 candidates lands on the upper bound"
