"""Aggregate the Exp-7 beam-search hyperparameter sweep into one long-format table.

Globs every config-tagged array produced by ``exp4_beam_search.py --run-id <slug>``
and scores each one under **both** metric conventions, emitting a single tidy CSV:

    dataset, set, run_id, <config columns>, <registry metrics>, <reference metrics>

The two conventions, and why both:

* **reference (dicoflex)** — ``reference_metrics.py``, valid-CFs-only with
  median-log LOF. This is the convention the paper table uses and the one to read
  first. ``lof_score_median_log`` is the plausibility number that means anything on
  HELOC (see below).
* **registry (cel)** — ``exp4_metrics_table.py``, the vendored ``cel`` registry's own
  metric classes, all-rows means. Carried for continuity with the Exp-4 table.

Columns that must NOT be reported (carried forward from Exp 4, guarded here by
prefixing them ``UNRELIABLE__``):

* ``validity_vs_true`` — the registry's ``mean(y_cf_pred != y_test)``. This project
  relabels (``y_target = 1 - disc.predict(X_test)``), under which that expression
  reduces algebraically to the discriminator's accuracy and says nothing about the
  generator. The validity to read is ``validity_target`` (``== y_target``).
* ``lof_scores_cf`` — the registry's all-rows LOF *mean*. HELOC's 115 all-zeros rows
  (the MinMax image of the ``-9`` "no record" code, 473-fold duplicated in X_train)
  sit on a zero-radius neighbourhood and blow the mean up to ~6.5e6. Use
  ``lof_score_median_log``.
* ``sparsity`` — exact-equality sparsity, saturates at 1.0 for continuously generated
  CFs. Use ``eps_sparsity``.
* ``sparsity_categorical`` / ``pairwise_diversity_mixed`` — not computable for this
  method (continuous relaxation of one-hots; one CF per factual).

Scoring is CPU-only — no GPU, no TabPFN. Generation happens on PLGrid; this runs
locally against the arrays pulled back.

Usage:
  uv run python experiments/zeroshot_cf/exp7_sweep_table.py
  uv run python experiments/zeroshot_cf/exp7_sweep_table.py --include-legacy
  uv run python experiments/zeroshot_cf/exp7_sweep_table.py --csv /tmp/sweep.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

RESULTS_DIR = Path(__file__).parent / "results"
ARRAYS_DIR = RESULTS_DIR / "arrays"
SWEEP_ARRAYS_DIR = ARRAYS_DIR / "sweep"

# Config fields lifted out of the npz's embedded config_json into table columns.
CONFIG_COLUMNS = [
    "beam_width",
    "n_candidates",
    "n_candidates_effective",
    "candidate_probs",
    "lambda_actionable",
    "lambda_immutable",
    "max_context",
    "n_estimators",
    "chunk_size",
    "max_test",
    "freeze_immutable",
    "git_commit",
    "slurm_job_id",
    "device",
    "elapsed_s",
]

# Metrics that are computed but must never be read as results. Renamed on the way
# into the table so a reader cannot pick one up by accident.
UNRELIABLE = [
    "validity_vs_true",
    "lof_scores_cf",
    "sparsity",
    "sparsity_categorical",
    "pairwise_diversity_mixed",
]

# Metric names emitted by BOTH scorers under DIFFERENT formulas. The reference
# version keeps the bare name (it is the convention to read); the registry version
# is namespaced ``registry__<name>``. Any other name collision is treated as a bug
# and raises, so a future divergence cannot slip through as a silent overwrite.
#
#   eps_sparsity — registry: mean over ALL rows (exp4_metrics_table.eps_sparsity)
#                  reference: mean over VALID rows only (reference_metrics.eps_sparsity)
COLLIDING_METRICS = {"eps_sparsity"}


def _both_nan(a: Any, b: Any) -> bool:
    try:
        return bool(np.isnan(a)) and bool(np.isnan(b))
    except (TypeError, ValueError):
        return False


# The headline columns, in reading order.
DISPLAY_COLUMNS = [
    "dataset",
    "set",
    "run_id",
    "n",
    "validity_target",
    "proximity_l1_continuous",
    "eps_sparsity",
    "lof_score_median_log",
    "proximity_l1_jaccard",
    "proximity_l2_jaccard",
    "isolation_forest_scores_cf",
]


def parse_cell_name(npz_path: Path) -> Tuple[str, str, str]:
    """Recover (dataset, set, run_id) from a filename.

    Two layouts:
      ``exp4_<dataset>_<set>_cfs.npz``               → run_id "default" (legacy Exp-4)
      ``exp4_<dataset>_<set>__<run-id>_cfs.npz``     → config-tagged Exp-7 sweep cell

    Run ids are validated to contain no underscore (``parse_run_id``), so splitting
    on ``__`` is unambiguous.
    """
    stem = npz_path.stem
    if stem.startswith("exp4_"):
        stem = stem[len("exp4_") :]
    if stem.endswith("_cfs"):
        stem = stem[: -len("_cfs")]

    if "__" in stem:
        cell, run_id = stem.split("__", 1)
    else:
        cell, run_id = stem, "default"
    dataset_name, tag = cell.rsplit("_", 1)
    return dataset_name, tag, run_id


def read_config(npz_path: Path) -> Dict[str, Any]:
    """Read the config dict embedded in the npz by ``run_dataset``.

    Arrays generated before ``--run-id`` existed carry no ``config_json``; those get
    an empty dict, and the config columns come back as NaN rather than as a guess.
    """
    with np.load(npz_path, allow_pickle=False) as z:
        if "config_json" not in z.files:
            return {}
        return json.loads(str(z["config_json"].item()))


def merge_scorer_rows(
    reference: Dict[str, Any], registry: Dict[str, Any], context: str = ""
) -> Dict[str, Any]:
    """Merge the two scorers' outputs into one flat row.

    Three rules, in order:

    1. The reference (dicoflex) convention keeps the bare column name — it is the
       one to read.
    2. Names in ``UNRELIABLE`` are prefixed so they cannot be picked up by accident.
    3. Names in ``COLLIDING_METRICS`` — same name, genuinely different formula — get
       the registry version namespaced ``registry__<name>`` so both survive.

    Any *other* name that appears in both with different values raises: it means a
    formula diverged without anyone noticing, and silently keeping one of them is how
    a mislabelled number reaches a paper table.
    """
    out: Dict[str, Any] = {}
    for key, value in reference.items():
        if key in ("dataset", "set"):
            continue
        out[f"UNRELIABLE__{key}" if key in UNRELIABLE else key] = value

    for key, value in registry.items():
        if key in ("dataset", "set", "n", "validity"):
            continue  # validity == validity_target, already carried
        name = f"UNRELIABLE__{key}" if key in UNRELIABLE else key
        if name in COLLIDING_METRICS:
            name = f"registry__{name}"
        elif name in out and out[name] != value and not _both_nan(out[name], value):
            where = f"{context}: " if context else ""
            raise AssertionError(
                f"{where}column {name!r} disagrees between scorers "
                f"({out[name]!r} vs {value!r}) — add it to COLLIDING_METRICS if the "
                "two formulas genuinely differ."
            )
        out[name] = value
    return out


def all_zero_row_breakdown(npz_path: Path, dataset_name: str) -> Dict[str, Any]:
    """Validity with and without the all-zeros factual rows.

    HELOC's test split contains a block of byte-identical all-zeros rows — the
    MinMax image of the ``-9`` "no record" sentinel, not observations. They are 5.5%
    of the split and are known to break all-rows LOF. Open-work item 3 is whether
    they belong in the evaluation set at all, so every sweep row carries the number
    computed both ways and the decision can be made on evidence.

    Because the rows are byte-identical and generation is deterministic, they all
    receive the *same* counterfactual: they are one query point counted n times, not
    n independent attempts.
    """
    from experiments.zeroshot_cf.data import load_dataset  # noqa: PLC0415
    from experiments.zeroshot_cf.discriminator import train_discriminator  # noqa: PLC0415

    with np.load(npz_path, allow_pickle=False) as z:
        X_test = z["X_test"]
        X_cf = np.clip(z["X_cf"], 0.0, 1.0)
        y_test = z["y_test"].astype(np.int64).squeeze()
        y_target = z["y_target"].astype(np.int64).squeeze()

    zero_mask = (X_test == 0.0).all(axis=1)
    n_zero = int(zero_mask.sum())
    if n_zero == 0:
        return {
            "n_allzero_rows": 0,
            "validity_excl_allzero": float("nan"),
            "validity_among_allzero": float("nan"),
            "allzero_cfs_identical": True,
        }

    bundle = load_dataset(dataset_name)
    disc = train_discriminator(
        bundle.X_train, bundle.y_train, X_test, y_test, dataset_name
    )
    pred = np.asarray(disc.predict(X_cf)).squeeze()
    keep = ~zero_mask
    cf_zero = X_cf[zero_mask]
    return {
        "n_allzero_rows": n_zero,
        "validity_excl_allzero": float((pred[keep] == y_target[keep]).mean()),
        "validity_among_allzero": float(
            (pred[zero_mask] == y_target[zero_mask]).mean()
        ),
        "allzero_cfs_identical": bool(
            np.array_equal(cf_zero, np.tile(cf_zero[0], (n_zero, 1)))
        ),
    }


def score_run(npz_path: Path) -> Dict[str, Any]:
    """Score one saved array under both metric conventions."""
    from experiments.zeroshot_cf import exp4_metrics_table, reference_metrics  # noqa: PLC0415

    dataset_name, tag, run_id = parse_cell_name(npz_path)
    config = read_config(npz_path)

    registry = exp4_metrics_table.score_cell(npz_path, dataset_name, tag)
    reference = reference_metrics.score_cell(npz_path, dataset_name, tag)

    row: Dict[str, Any] = {
        "dataset": dataset_name,
        "set": tag,
        "run_id": run_id,
        "n": registry["n"],
        "npz": npz_path.name,
    }
    for key in CONFIG_COLUMNS:
        value = config.get(key)
        # candidate_probs is a list (or None); flatten it to a stable string so the
        # column stays scalar and groupby/sort behave.
        if key == "candidate_probs":
            value = "interior" if value is None else ",".join(str(p) for p in value)
        row[key] = value

    row.update(merge_scorer_rows(reference, registry, context=npz_path.name))
    row.update(all_zero_row_breakdown(npz_path, dataset_name))

    # Cross-check: the two scorers must agree on validity. They compute it from the
    # same arrays and the same cached discriminator, so a mismatch means one of them
    # parsed the cell wrong — fail loudly rather than emit a plausible table.
    if not np.isclose(registry["validity"], reference["validity_target"]):
        raise AssertionError(
            f"{npz_path.name}: validity disagrees between scorers "
            f"({registry['validity']} vs {reference['validity_target']})"
        )
    return row


def collect(include_legacy: bool, min_n: int) -> List[Path]:
    paths = sorted(SWEEP_ARRAYS_DIR.glob("exp4_*_cfs.npz"))
    if include_legacy:
        # The four historical cells, generated locally on MPS before --run-id existed.
        # Off by default: they are a different backend and must not be mixed into a
        # sensitivity table (PROJECT_STATE, "confounded with hardware").
        paths += sorted(ARRAYS_DIR.glob("exp4_*_cfs.npz"))
    kept = []
    for p in paths:
        with np.load(p, allow_pickle=False) as z:
            n = int(z["X_cf"].shape[0])
        if n < min_n:
            print(f"skip {p.name} (n={n} < {min_n})")
            continue
        kept.append(p)
    return kept


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--min-n",
        type=int,
        default=100,
        help="Skip cells with fewer rows (default 100: drops smoke runs).",
    )
    parser.add_argument(
        "--include-legacy",
        action="store_true",
        help="Also score the untagged Exp-4 arrays in results/arrays/. These were "
        "generated locally on MPS; mixing them into a sensitivity table "
        "reintroduces the hardware confound. Off by default.",
    )
    parser.add_argument(
        "--csv", type=str, default=str(RESULTS_DIR / "exp7_sweep_metrics.csv")
    )
    args = parser.parse_args()

    paths = collect(args.include_legacy, args.min_n)
    if not paths:
        print(f"No arrays found under {SWEEP_ARRAYS_DIR}. Nothing to score.")
        return

    rows: List[Dict[str, Any]] = []
    for i, p in enumerate(paths, 1):
        print(f"\n[{i}/{len(paths)}] === {p.name} ===")
        rows.append(score_run(p))

    df = pd.DataFrame(rows).sort_values(["dataset", "set", "run_id"])
    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.csv, index=False)

    cols = [c for c in DISPLAY_COLUMNS if c in df.columns]
    print("\n" + "=" * 78)
    print("Reference (dicoflex) convention — valid-CFs-only, median-log LOF")
    print("=" * 78)
    print(df[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nWrote {args.csv}  ({len(df)} runs)")


if __name__ == "__main__":
    main()
