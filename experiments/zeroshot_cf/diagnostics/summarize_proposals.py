"""Recompute trace findings from completed, hash-checked diagnostic payloads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from experiments.zeroshot_cf.diagnostics.proposal_backends import digest, write_json


def load(path):
    return json.loads(path.read_text())


def rows_set(rows):
    return {tuple(row) for row in rows}


def numerical_menu(probe, state_index, n_columns):
    state = np.asarray(probe["states"][state_index])
    projected = np.asarray(probe["projected"])
    rows = []
    for pair in range(state_index * n_columns, (state_index + 1) * n_columns):
        column = probe["columns"][pair]
        for value in projected[pair].reshape(-1):
            row = state.copy()
            row[column] = value
            if not np.array_equal(row, state):
                rows.append(row.tolist())
    return rows_set(rows)


def summarize(directory):
    complete = load(directory / "COMPLETE.json")
    for name, expected in complete["files"].items():
        if digest(directory / name) != expected:
            raise ValueError(f"payload hash mismatch: {directory / name}")
    meta = load(directory / "metadata.json")
    for backend in ("tabicl", "empirical"):
        resolved = meta["source_identities"][backend]["resolved"]
        if (
            meta["backend_implementations"][backend]
            != resolved["backend_implementation"]
        ):
            raise ValueError("trace backend identity differs from historical run")
        checkpoints = meta["checkpoint_content_ids"] if backend == "tabicl" else {}
        if checkpoints != resolved["checkpoint_content_ids"]:
            raise ValueError("trace checkpoint identity differs from historical run")
    summary = load(directory / "summary.json")
    if len(summary) != complete["points"] or len(summary) != meta["count"]:
        raise ValueError("trace is incomplete")
    points = []
    for item in summary:
        i = item["point"]
        probes = {
            b: load(directory / f"{i:02d}_{b}_common.json")
            for b in ("tabicl", "empirical")
        }
        traces = {
            b: load(directory / f"{i:02d}_{b}_trace.json")
            for b in ("tabicl", "empirical")
        }
        ncol = len(meta["numerical_columns"])
        a, b = probes["tabicl"], probes["empirical"]
        np.testing.assert_array_equal(a["states"], b["states"])
        np.testing.assert_array_equal(a["columns"], b["columns"])
        # Original factual only: do not pool the perturbed-state control into it.
        ra, rb = [np.asarray(p["raw"])[:ncol] for p in (a, b)]
        pa, pb = [np.asarray(p["projected"])[:ncol] for p in (a, b)]
        raw_diff = np.abs(ra - rb) > 1e-8
        projected_diff = np.abs(pa - pb) > 1e-8
        menus = {
            backend: numerical_menu(probes[backend], 0, ncol) for backend in probes
        }
        search = {}
        cat_menus = {}
        final_pools = {}
        for backend, trace in traces.items():
            events = trace["search"]
            trials = [e for e in events if e["stage"].endswith("_trials")]
            classifiers = [e for e in events if e["stage"] == "classifier"][1:]
            pools = [e for e in events if e["stage"] == "pool"]
            qualities = [e for e in events if e["stage"] == "quality"]
            categorical = [e for e in events if e["stage"] == "categorical_trials"]
            cat_menus[backend] = (
                rows_set(categorical[0]["rows"]) if categorical else set()
            )
            trial_count = sum(len(e["rows"]) for e in trials)
            scored = sum(len(e["rows"]) for e in classifiers)
            valid_rows = set()
            for event in classifiers:
                for row, probability, prediction in zip(
                    event["rows"],
                    event["probabilities"],
                    event["predictions"],
                    strict=True,
                ):
                    if probability >= 0.5 and prediction == item["target"]:
                        valid_rows.add(tuple(row))
            numerical_rows = set().union(
                *(
                    rows_set(e["rows"])
                    for e in trials
                    if e["stage"] == "numerical_trials"
                )
            )
            categorical_rows = set().union(
                *(
                    rows_set(e["rows"])
                    for e in trials
                    if e["stage"] == "categorical_trials"
                )
            )
            final_pools[backend] = rows_set(pools[-1]["rows"]) if pools else set()
            search[backend] = {
                "valid_numerical_trial_rows": len(valid_rows & numerical_rows),
                "valid_categorical_trial_rows": len(valid_rows & categorical_rows),
                "proposed_trial_rows": trial_count,
                "scored_trial_rows": scored,
                "visited_or_duplicate_rows": trial_count - scored,
                "valid_scored_rows": sum(
                    sum(
                        p >= 0.5 and y == item["target"]
                        for p, y in zip(
                            e["probabilities"], e["predictions"], strict=True
                        )
                    )
                    for e in classifiers
                ),
                "final_quality_input": len(qualities[-1]["input"]) if qualities else 0,
                "final_quality_survivors": len(qualities[-1]["rows"])
                if qualities
                else 0,
                "final_pool_size": len(pools[-1]["rows"]) if pools else 0,
                "returned": item["stats"][backend]["available"],
            }
        factual = np.asarray(a["states"][0])
        ca, cb = [
            np.asarray(traces[k]["candidates"][0], dtype=float)
            for k in ("tabicl", "empirical")
        ]
        shared_rows = rows_set(ca[np.all(np.isfinite(ca), axis=1)]) & rows_set(
            cb[np.all(np.isfinite(cb), axis=1)]
        )
        shared_changes = []
        for row in sorted(shared_rows):
            changes = []
            for column in np.flatnonzero(np.asarray(row) != factual):
                entry = {
                    "feature": meta["feature_names"][column],
                    "column": int(column),
                    "from": float(factual[column]),
                    "to": row[column],
                }
                if column in meta["numerical_columns"]:
                    pos = meta["numerical_columns"].index(column)
                    entry["original_state_raw_tabicl"] = ra[pos].reshape(-1).tolist()
                    entry["original_state_raw_empirical"] = rb[pos].reshape(-1).tolist()
                    entry["original_state_projected_tabicl"] = (
                        pa[pos].reshape(-1).tolist()
                    )
                    entry["original_state_projected_empirical"] = (
                        pb[pos].reshape(-1).tolist()
                    )
                changes.append(entry)
            shared_changes.append(changes)
        points.append(
            {
                "point": i,
                "source_index": item["source_index"],
                "initial_raw_quantiles": int(raw_diff.size),
                "initial_raw_differing": int(raw_diff.sum()),
                "initial_projected_differing": int(projected_diff.sum()),
                "initial_differences_erased": int((raw_diff & ~projected_diff).sum()),
                "initial_numerical_unique_tabicl": len(menus["tabicl"]),
                "initial_numerical_unique_empirical": len(menus["empirical"]),
                "initial_numerical_shared": len(menus["tabicl"] & menus["empirical"]),
                "initial_categorical_tabicl": len(cat_menus["tabicl"]),
                "initial_categorical_empirical": len(cat_menus["empirical"]),
                "initial_categorical_equal": cat_menus["tabicl"]
                == cat_menus["empirical"],
                "search": search,
                "final_pool_exact": final_pools["tabicl"] == final_pools["empirical"],
                "sensitivity": item["sensitivity"],
                "final_comparison": item["final_comparison"],
                "shared_candidate_changes": shared_changes,
            }
        )
    totals = {
        key: sum(p[key] for p in points)
        for key in (
            "initial_raw_quantiles",
            "initial_raw_differing",
            "initial_projected_differing",
            "initial_differences_erased",
            "initial_numerical_unique_tabicl",
            "initial_numerical_unique_empirical",
            "initial_numerical_shared",
        )
    }
    return {
        "dataset": meta["dataset"],
        "complete_sha256": digest(directory / "COMPLETE.json"),
        "points": points,
        "totals": totals,
        "total_s": complete["total_s"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = [summarize(directory) for directory in args.directories]
    write_json(args.output, result)
    for dataset in result:
        print(dataset["dataset"], dataset["totals"])


if __name__ == "__main__":
    main()
