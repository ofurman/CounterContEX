"""Unit tests for the iterative greedy CF core (Stage 1).

Covers:
  (a) loop terminates and flips the discriminator within budget;
  (b) changed ⊆ actionable_idx and non-actionable columns byte-identical;
  (c) per-CF L0 = |changed| and ≤ |A|;
  (d) prob_ascent picks the candidate maximizing predict_proba[y_target];
  (e) class_divergence picks the higher mean-shift feature;
  (f) budget exhaustion returns flipped=False (counted invalid);
  (g) predictive_distribution correctness (mean ≈ empirical mean of draws);
  (h) sample_feature class-conditional pass-through (append_target=True) and the
      existing append_target=False path both work.

The selection-logic tests (d, e) use lightweight stubs so they are deterministic
and fast; the rest use the real local v2 checkpoints via the shared ``models``
fixture (tests/conftest.py).
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from experiments.zeroshot_cf.discriminator import DiscriminatorModel
from experiments.zeroshot_cf.greedy import (
    greedy_counterfactual,
    _select_prob_ascent,
    _select_class_divergence,
)
from experiments.zeroshot_cf.sampler import ConditionalDensitySampler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_separable(n: int = 60, seed: int = 0):
    """Clearly separable 2-class, 2-feature dataset in [0,1].

    class 0: both features ~U(0, 0.35); class 1: both features ~U(0.65, 1.0).
    A linear discriminator flips when both features move from low to high.
    """
    rng = np.random.default_rng(seed)
    x_c0 = rng.uniform(0.0, 0.35, (n, 2))
    x_c1 = rng.uniform(0.65, 1.0, (n, 2))
    X = np.vstack([x_c0, x_c1]).astype(np.float64)
    y = np.array([0] * n + [1] * n, dtype=np.int64)
    return X, y


def _make_synthetic(n: int = 80, seed: int = 0):
    """3-feature dataset: x2 = x0 + x1 + noise, binary label (mirror test_sampler)."""
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(0, 1, n)
    x1 = rng.uniform(0, 1, n)
    x2 = x0 + x1 + 0.05 * rng.standard_normal(n)
    X = np.stack([x0, x1, x2], axis=1).astype(np.float64)
    y = (x2 > 1.0).astype(np.int64)
    return X, y


def _make_disc(X, y):
    clf = LogisticRegression(max_iter=1000, random_state=42).fit(X, y)
    return DiscriminatorModel(clf)


def _make_lowcard(n: int = 120, seed: int = 0):
    """3-feature dataset where col 1 is a low-cardinality categorical-looking
    column that ``infer_categorical_features`` routes to TabPFN's classifier
    head (few unique values, many samples per value). Col 0 and col 2 stay
    continuous (regressor-routed). Binary label depends on col 0 and col 1 so
    both are informative. All features are MinMax-[0,1]."""
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(0, 1, n)
    # col 1: only 4 distinct levels in [0,1] → categorical, classifier-routed.
    levels = np.array([0.0, 1.0 / 3, 2.0 / 3, 1.0])
    x1 = levels[rng.integers(0, 4, n)]
    x2 = rng.uniform(0, 1, n)
    X = np.stack([x0, x1, x2], axis=1).astype(np.float64)
    y = ((x0 + x1) > 1.0).astype(np.int64)
    return X, y


# ---------------------------------------------------------------------------
# (a)(b)(c) real-model greedy loop on a separable case
# ---------------------------------------------------------------------------

def test_greedy_flips_within_budget(models):
    clf, reg = models
    X, y = _make_separable(n=60, seed=0)
    disc = _make_disc(X, y)

    sampler = ConditionalDensitySampler(
        clf, reg, append_target=True, n_permutations=3,
        temperature=1e-9, random_state=42,
    )
    sampler.set_context(X, y_context=y, target_class=1)

    x = np.array([0.1, 0.1], dtype=np.float64)  # clearly class 0
    actionable_idx = [0, 1]
    y_target = 1

    x_cf, changed, info = greedy_counterfactual(
        sampler, disc, x, y_target, actionable_idx, "prob_ascent",
        tau=0.5, budget=len(actionable_idx), temperature=1e-9,
    )

    # (a) terminates with a flip
    assert info["flipped"] is True
    assert int(disc.predict(x_cf.reshape(1, -1))[0]) == y_target
    # (b) changed ⊆ actionable
    assert set(changed).issubset(set(actionable_idx))
    # (c) L0 = |changed| and ≤ |A|
    assert info["steps"] == len(changed)
    assert len(changed) <= len(actionable_idx)


def test_greedy_immutable_preserved(models):
    """Non-actionable columns must be byte-identical after generation."""
    clf, reg = models
    X, y = _make_synthetic(n=80, seed=1)
    disc = _make_disc(X, y)

    sampler = ConditionalDensitySampler(
        clf, reg, append_target=True, n_permutations=3,
        temperature=1e-9, random_state=7,
    )
    sampler.set_context(X, y_context=y, target_class=1)

    x = X[0].copy()
    y_target = 1 - int(disc.predict(x.reshape(1, -1))[0])
    actionable_idx = [0, 2]  # col 1 immutable
    immutable = [1]

    x_cf, changed, info = greedy_counterfactual(
        sampler, disc, x, y_target, actionable_idx, "prob_ascent",
        budget=len(actionable_idx),
    )

    assert set(changed).issubset(set(actionable_idx))
    np.testing.assert_array_equal(x_cf[immutable], x[immutable])
    assert len(changed) <= len(actionable_idx)


# ---------------------------------------------------------------------------
# (d) prob_ascent selection logic (stubbed, deterministic)
# ---------------------------------------------------------------------------

class _StubSamplerPA:
    def __init__(self, vals):
        self.vals = vals  # {col: value}

    def sample_feature(self, X, target_col, sample_temperature=None, fixed_target=None):
        return np.array([self.vals[target_col]])


class _StubDiscPA:
    """p(target) increases more when column 1 is set (weight 5 vs 1)."""

    def __init__(self, w):
        self.w = np.asarray(w, dtype=float)

    def predict_proba(self, X):
        s = float(np.clip((X[0] * self.w).sum() / 10.0, 0.0, 1.0))
        return np.array([[1.0 - s, s]])


def test_prob_ascent_picks_max_score():
    sampler = _StubSamplerPA(vals={0: 1.0, 1: 1.0})
    disc = _StubDiscPA(w=[1.0, 5.0])
    x_cf = np.array([0.0, 0.0], dtype=np.float64)

    j_star, score, val = _select_prob_ascent(
        sampler, disc, x_cf, y_target=1, candidates=[0, 1], temperature=1e-9
    )
    assert j_star == 1  # column 1 raises p(target) most
    assert val == 1.0
    assert score > 0.0


# ---------------------------------------------------------------------------
# (e) class_divergence selection logic (stubbed, deterministic)
# ---------------------------------------------------------------------------

class _FakeCrit:
    def mean(self, logits):
        return logits  # logits already hold the mean


class _StubSamplerCD:
    def __init__(self, means):
        self.means = means  # {(col, fixed_target): mean}

    def predictive_distribution(self, X, target_col, fixed_target=None):
        m = self.means[(target_col, fixed_target)]
        return {"logits": torch.tensor([m], dtype=torch.float32),
                "criterion": _FakeCrit()}


def test_class_divergence_picks_max_shift():
    # col 0: |0.1 - 0.0| = 0.1 ; col 1: |0.9 - 0.0| = 0.9 → col 1 wins
    means = {
        (0, 1): 0.1, (0, 0): 0.0,
        (1, 1): 0.9, (1, 0): 0.0,
    }
    sampler = _StubSamplerCD(means)
    x_cf = np.array([0.0, 0.0], dtype=np.float64)

    j_star, div, val = _select_class_divergence(
        sampler, x_cf, y_target=1, y_current=0, candidates=[0, 1]
    )
    assert j_star == 1
    assert abs(div - 0.9) < 1e-6
    assert val is None  # the loop draws the committed value


# ---------------------------------------------------------------------------
# (e2) class_divergence on a classifier-routed column (regression for the
#      KeyError: 'logits' crash on HELOC's low-cardinality integer features).
# ---------------------------------------------------------------------------

def test_class_divergence_handles_classifier_column(models):
    """A low-cardinality column routes to TabPFN's classifier head, whose
    predictive_distribution returns {"proba","classes"} (not {"logits"}). The
    class_divergence selector must run without raising KeyError and return a
    valid candidate index. Fails before the fix (KeyError: 'logits')."""
    clf, reg = models
    X, y = _make_lowcard(n=120, seed=0)

    sampler = ConditionalDensitySampler(
        clf, reg, append_target=True, n_permutations=3,
        temperature=1e-9, random_state=42,
    )
    sampler.set_context(X, y_context=y, target_class=None)  # both-classes pool

    # Sanity: col 1 really is routed to the classifier head in this context.
    assert sampler.model.use_classifier_(1, sampler.model.X_[:, 1]), (
        "test precondition broken: col 1 is not classifier-routed"
    )

    x_cf = X[0].copy()
    candidates = [0, 1, 2]  # includes the classifier-routed col 1

    j_star, div, val = _select_class_divergence(
        sampler, x_cf, y_target=1, y_current=0, candidates=candidates
    )
    assert j_star in candidates
    assert np.isfinite(div) and div >= 0.0
    assert val is None

    # And the full loop with selector="class_divergence" must also not raise.
    disc = _make_disc(X, y)
    y_target = 1 - int(disc.predict(x_cf.reshape(1, -1))[0])
    x_out, changed, info = greedy_counterfactual(
        sampler, disc, x_cf, y_target, candidates, "class_divergence",
        budget=len(candidates),
    )
    assert set(changed).issubset(set(candidates))
    assert "flipped" in info


# ---------------------------------------------------------------------------
# (f) budget exhaustion → flipped=False
# ---------------------------------------------------------------------------

def test_budget_exhaustion_returns_not_flipped():
    X, y = _make_separable(n=60, seed=2)
    disc = _make_disc(X, y)
    x = np.array([0.1, 0.1], dtype=np.float64)  # class 0
    # budget=0: no feature may change, factual is not the target → not flipped.
    x_cf, changed, info = greedy_counterfactual(
        sampler=None, disc=disc, x=x, y_target=1,
        actionable_idx=[0, 1], selector="prob_ascent", budget=0,
    )
    assert info["flipped"] is False
    assert changed == []
    np.testing.assert_array_equal(x_cf, x)


# ---------------------------------------------------------------------------
# (g) predictive_distribution correctness — mean ≈ empirical mean of draws
# ---------------------------------------------------------------------------

def test_predictive_distribution_mean_matches_samples(models):
    from experiments.zeroshot_cf.sampler import mean_of_prediction

    clf, reg = models
    X, y = _make_synthetic(n=80, seed=3)
    disc_target = 1

    sampler = ConditionalDensitySampler(
        clf, reg, append_target=True, n_permutations=3,
        temperature=1.0, random_state=0,
    )
    sampler.set_context(X, y_context=y, target_class=disc_target)

    X_query = X[:3].copy()
    j = 2  # reconstruct x2

    dist = sampler.predictive_distribution(X_query, target_col=j, fixed_target=disc_target)
    dist_mean = mean_of_prediction(dist["logits"], dist["criterion"])

    draws = sampler.sample_feature(
        X_query, target_col=j, sample_temperature=1.0, n_samples=200,
        fixed_target=disc_target,
    )  # shape (200, 3)
    emp_mean = draws.mean(axis=0)

    print(f"\ndist_mean={dist_mean}  emp_mean={emp_mean}")
    np.testing.assert_allclose(dist_mean, emp_mean, atol=0.15)


# ---------------------------------------------------------------------------
# (h) sample_feature class-conditional pass-through
# ---------------------------------------------------------------------------

def test_sample_feature_class_conditional_passthrough(models):
    clf, reg = models
    X, y = _make_synthetic(n=80, seed=4)

    # append_target=True path: must accept fixed_target without raising.
    sampler_ct = ConditionalDensitySampler(
        clf, reg, append_target=True, n_permutations=2,
        temperature=1e-9, random_state=0,
    )
    sampler_ct.set_context(X, y_context=y, target_class=1)
    vals = sampler_ct.sample_feature(X[:5], target_col=2, fixed_target=1)
    assert vals.shape == (5,)
    assert np.all(np.isfinite(vals))

    # append_target=False path: existing behaviour, no fixed_target.
    sampler_plain = ConditionalDensitySampler(
        clf, reg, append_target=False, n_permutations=2,
        temperature=1e-9, random_state=0,
    )
    sampler_plain.set_context(X)
    vals_plain = sampler_plain.sample_feature(X[:5], target_col=2)
    assert vals_plain.shape == (5,)
    assert np.all(np.isfinite(vals_plain))
