"""Offline unit tests for the Exp-7 sweep plumbing added to ``exp4_beam_search``.

The sweep runs the same cell many times under different beam settings, so every
artifact has to be config-tagged. These tests pin the two properties that make that
safe, and that a silent regression would otherwise destroy months of results with:

  1. **The default path is byte-identical to the original Exp-4 layout.** Omitting
     ``--run-id`` must reproduce ``exp4_<dataset>_<set>_cfs.npz`` and friends exactly,
     so the four historical cells keep their meaning.
  2. **Sweep artifacts are invisible to the existing scorers.** ``exp4_metrics_table``
     and ``reference_metrics`` glob ``arrays/exp4_*_cfs.npz`` non-recursively and parse
     the stem with ``rsplit("_", 1)``. A config-tagged file that matched that glob
     would be mis-parsed into a bogus (dataset, set) pair and silently pollute the
     Exp-4 table.

No checkpoint, no network, no GPU — this is pure path/argument logic.
"""

from __future__ import annotations

import fnmatch

import pytest

from experiments.zeroshot_cf.beam_search import BeamConfig
from experiments.zeroshot_cf.exp4_beam_search import (
    ARRAYS_DIR,
    CANDIDATE_PROB_PRESETS,
    RESULTS_DIR,
    cell_paths,
    parse_candidate_probs,
    parse_run_id,
)
from experiments.zeroshot_cf.exp7_sweep_table import (
    COLLIDING_METRICS,
    merge_scorer_rows,
    parse_cell_name,
)

# The glob the pre-existing scorers use to find Exp-4 cells.
LEGACY_GLOB = "exp4_*_cfs.npz"


# ---------------------------------------------------------------------------
# 1. The default run-id reproduces the original filenames
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("dataset", "tag"),
    [
        ("heloc", "frozen"),
        ("heloc", "fromscratch"),
        ("law", "frozen"),
        ("law", "fromscratch"),
        ("moons", "frozen"),
    ],
)
def test_default_run_id_reproduces_legacy_filenames(dataset, tag):
    paths = cell_paths(dataset, tag, None)
    assert paths["npz"] == ARRAYS_DIR / f"exp4_{dataset}_{tag}_cfs.npz"
    assert paths["metrics_csv"] == RESULTS_DIR / f"exp4_{dataset}_{tag}_metrics.csv"
    assert paths["summary"] == RESULTS_DIR / "exp4_summary.md"


def test_empty_run_id_is_the_default_run_id():
    """``--run-id ''`` from a shell variable that expanded to nothing must not
    create a cell tagged with the empty string."""
    assert parse_run_id("") is None
    assert parse_run_id(None) is None
    assert cell_paths("law", "frozen", parse_run_id("")) == cell_paths(
        "law", "frozen", None
    )


# ---------------------------------------------------------------------------
# 2. Sweep artifacts live in their own namespace
# ---------------------------------------------------------------------------


def test_sweep_paths_are_config_tagged_and_separated():
    paths = cell_paths("heloc", "frozen", "bw16")
    assert paths["npz"].name == "exp4_heloc_frozen__bw16_cfs.npz"
    assert paths["npz"].parent == ARRAYS_DIR / "sweep"
    assert paths["metrics_csv"].name == "exp4_heloc_frozen__bw16_metrics.csv"
    # A per-run-id summary: a shared one would be overwritten by whichever config
    # happened to finish last and would report one arbitrary cell as "the" result.
    assert paths["summary"].name == "exp4_summary__bw16.md"


def test_sweep_npz_never_matches_the_legacy_scorer_glob():
    """The decisive isolation property. The sweep file lives in a SUBdirectory, so a
    non-recursive ``arrays/exp4_*_cfs.npz`` cannot reach it — but assert on the
    relative path too, so moving it back up would fail loudly here."""
    for run_id in ["bw16", "lam0", "probs-tail", "base", "ctx1024"]:
        npz = cell_paths("heloc", "frozen", run_id)["npz"]
        rel = npz.relative_to(ARRAYS_DIR)
        assert rel.parent.name == "sweep", "sweep arrays must not sit in arrays/"
        assert not fnmatch.fnmatch(str(rel), LEGACY_GLOB)


def test_distinct_run_ids_never_collide():
    ids = ["base", "bw4", "bw16", "bw32", "nc4", "nc16", "lam0", "probs-tail"]
    names = {cell_paths("heloc", "frozen", r)["npz"].name for r in ids}
    assert len(names) == len(ids)


# ---------------------------------------------------------------------------
# 3. Run-id validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("run_id", ["bw16", "lam0.1", "probs-tail", "base", "n8"])
def test_valid_run_ids_accepted(run_id):
    assert parse_run_id(run_id) == run_id


@pytest.mark.parametrize("run_id", ["bad_id", "-lead", "has space", "a/b", "λ0"])
def test_invalid_run_ids_rejected(run_id):
    """Underscore especially: it is the field separator in the filename, so a run-id
    containing one would make ``exp4_heloc_frozen__a_b_cfs.npz`` ambiguous."""
    with pytest.raises(ValueError, match="invalid --run-id"):
        parse_run_id(run_id)


# ---------------------------------------------------------------------------
# 4. candidate_probs CLI parsing and its effect on BeamConfig
# ---------------------------------------------------------------------------


def test_candidate_probs_default_is_the_interior_grid():
    assert parse_candidate_probs(None) is None
    assert parse_candidate_probs("") is None
    assert parse_candidate_probs("interior") is None
    # None → BeamConfig derives the interior grid from n_candidates, unchanged.
    assert BeamConfig(n_candidates=6, candidate_probs=None).probs() == [
        1 / 6,
        2 / 6,
        3 / 6,
        4 / 6,
        5 / 6,
    ]


def test_tail_preset_reaches_the_tails():
    probs = parse_candidate_probs("tail")
    assert probs == CANDIDATE_PROB_PRESETS["tail"]
    interior = BeamConfig(n_candidates=6).probs()
    # The point of the preset: it proposes values further out than the default grid,
    # which is what lets a step take a large move.
    assert min(probs) < min(interior) and max(probs) > max(interior)


def test_explicit_list_parsed_and_used_verbatim():
    probs = parse_candidate_probs("0.05,0.5,0.95")
    assert probs == [0.05, 0.5, 0.95]
    assert BeamConfig(n_candidates=6, candidate_probs=probs).probs() == probs


def test_explicit_probs_override_n_candidates():
    """Documented precedence: with an explicit list the branching factor becomes
    len(probs) + 1 (the mode), regardless of --n-candidates."""
    probs = parse_candidate_probs("tail")
    cfg = BeamConfig(n_candidates=99, candidate_probs=probs)
    assert len(cfg.probs()) + 1 == len(probs) + 1 == 6


@pytest.mark.parametrize("bad", ["0.0,0.5", "0.5,1.0", "-0.1", "abc", "0.5,foo"])
def test_invalid_candidate_probs_rejected(bad):
    with pytest.raises(ValueError):
        parse_candidate_probs(bad)


# ---------------------------------------------------------------------------
# 5. The aggregator recovers the cell identity the generator wrote
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("dataset", "tag", "run_id"),
    [
        ("heloc", "frozen", None),
        ("heloc", "fromscratch", None),
        ("law", "frozen", "base"),
        ("law", "fromscratch", "probs-tail"),
        ("heloc", "frozen", "lam0.1"),
    ],
)
def test_filename_round_trips_through_the_aggregator(dataset, tag, run_id):
    """``exp7_sweep_table`` reads the sweep back purely from filenames, so the
    generator's naming and the aggregator's parse must be exact inverses. A drift
    here silently relabels which config produced which numbers."""
    npz = cell_paths(dataset, tag, run_id)["npz"]
    got_dataset, got_tag, got_run_id = parse_cell_name(npz)
    assert (got_dataset, got_tag) == (dataset, tag)
    assert got_run_id == (run_id or "default")


# ---------------------------------------------------------------------------
# 6. Merging the two scorers must not lose or mislabel a metric
# ---------------------------------------------------------------------------


def test_colliding_metric_keeps_both_conventions():
    """``eps_sparsity`` is emitted by both scorers under different formulas — the
    registry averages over all rows, the reference over valid rows only. On
    heloc/frozen these are 0.4456 and 0.4633. Keeping only one would print a
    valid-only number in a column labelled 'registry'."""
    assert "eps_sparsity" in COLLIDING_METRICS
    merged = merge_scorer_rows(
        reference={"eps_sparsity": 0.4633, "validity_target": 0.3762},
        registry={"eps_sparsity": 0.4456, "coverage": 1.0},
    )
    assert merged["eps_sparsity"] == 0.4633, "bare name is the reference convention"
    assert merged["registry__eps_sparsity"] == 0.4456
    assert merged["coverage"] == 1.0


def test_unreliable_metrics_are_prefixed_out_of_reach():
    merged = merge_scorer_rows(
        reference={"validity_vs_true": 0.72, "lof_score_median_log": 0.03},
        registry={"lof_scores_cf": 6.5e6, "sparsity": 1.0},
    )
    assert "validity_vs_true" not in merged
    assert "lof_scores_cf" not in merged and "sparsity" not in merged
    assert merged["UNRELIABLE__lof_scores_cf"] == 6.5e6
    # The reportable plausibility number keeps its plain name.
    assert merged["lof_score_median_log"] == 0.03


def test_undeclared_collision_raises_rather_than_overwriting():
    """The guard that stops a future formula divergence from silently picking one
    side. Without it the merge would keep whichever scorer ran last."""
    with pytest.raises(AssertionError, match="disagrees between scorers"):
        merge_scorer_rows(
            reference={"proximity_l1_continuous": 0.0625},
            registry={"proximity_l1_continuous": 0.9999},
        )


def test_identical_values_in_both_scorers_are_not_a_collision():
    merged = merge_scorer_rows(
        reference={"disc_accuracy": 0.7232},
        registry={"disc_accuracy": 0.7232},
    )
    assert merged["disc_accuracy"] == 0.7232
