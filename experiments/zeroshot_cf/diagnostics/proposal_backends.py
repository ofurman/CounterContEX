"""Audit paired artifacts and trace existing CounterContEx without changing policy.

Run with ``python -m experiments.zeroshot_cf.diagnostics.proposal_backends``.
Trace output is diagnostic evidence, never a canonical benchmark run.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
import time
from collections import defaultdict
from collections.abc import Mapping
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from experiments.zeroshot_cf import diverse_search as beam
from experiments.zeroshot_cf.analysis.core import load_published_cells
from experiments.zeroshot_cf.candidate_domains import project_candidate_values
from experiments.zeroshot_cf.core.contracts import GenerationRequest
from experiments.zeroshot_cf.datasets.benchmark import method_context
from experiments.zeroshot_cf.methods.registry import DEFAULT_METHOD_REGISTRY
from experiments.zeroshot_cf.orchestration.artifacts import ArtifactStore
from experiments.zeroshot_cf.orchestration.matrix import load_matrix_config
from experiments.zeroshot_cf.orchestration.runner import _default_case_loader


def serial(value):
    if isinstance(value, np.ndarray):
        return serial(value.tolist())
    if isinstance(value, np.generic):
        return serial(value.item())
    if isinstance(value, Mapping):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serial(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(serial(value), indent=2, allow_nan=False) + "\n")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def row_id(row):
    return hashlib.sha256(np.asarray(row, dtype=np.float64).tobytes()).hexdigest()


def compare_candidates(a, am, b, bm, *, atol=1e-8):
    """Pair ranks and unordered sets; unavailable slots are never matches."""
    if a.shape != b.shape or am.shape != bm.shape or a.shape[:2] != am.shape:
        raise ValueError("paired candidate shapes differ")
    result = []
    for i in range(len(a)):
        aa, bb = a[i][am[i]], b[i][bm[i]]
        joint = am[i] & bm[i]
        distances = np.abs(a[i][joint] - b[i][joint])
        matches = np.all(aa[:, None] == bb[None, :], axis=-1)
        close = np.all(np.abs(aa[:, None] - bb[None, :]) <= atol, axis=-1)
        both = len(aa) > 0 and len(bb) > 0

        # k=3: enumerate bijections rather than incorrectly accepting many-to-one.
        def equal_sets(matrix):
            if not both:
                return None
            if len(aa) != len(bb):
                return False
            return any(
                all(matrix[j, p[j]] for j in range(len(aa)))
                for p in itertools.permutations(range(len(bb)))
            )

        result.append(
            {
                "point": i,
                "a_count": len(aa),
                "b_count": len(bb),
                "joint_available_count": int(joint.sum()),
                "rank_exact_count": int(np.all(distances == 0, axis=-1).sum()),
                "matched_exact_count": int(matches.any(axis=1).sum()),
                "matched_close_count": int(close.any(axis=1).sum()),
                "set_exact": equal_sets(matches),
                "set_close": equal_sets(close),
                "rank_max_abs_difference": float(distances.max())
                if distances.size
                else None,
            }
        )
    return result


def paired_runs(root, matrix):
    cells = load_published_cells(root, matrix)
    store = ArtifactStore(root)
    pairs = defaultdict(dict)
    for row in cells:
        run = store.read(row["run_id"])
        spec = run.manifest["scientific_spec"]
        backend = spec["method"]["params"]["foundation"]["backend"]
        key = (spec["dataset"]["name"], spec["seed"])
        if backend in pairs[key]:
            raise ValueError("duplicate backend cell")
        pairs[key][backend] = run
    for arms in pairs.values():
        if set(arms) != {"tabicl", "empirical"}:
            raise ValueError("missing or unexpected backend arm")
        identities = []
        for backend in ("tabicl", "empirical"):
            identity = deepcopy(arms[backend].manifest["identity"])
            identity["scientific_spec"]["method"]["params"]["foundation"].pop("backend")
            identity["resolved"].pop("backend_implementation")
            identity["resolved"].pop("checkpoint_content_ids")
            identities.append(identity)
        if identities[0] != identities[1]:
            raise ValueError("pair differs outside backend identity bundle")
    return pairs


def audit(root, matrix, output):
    output.mkdir(parents=True, exist_ok=False)
    pairs = paired_runs(root, matrix)
    records, summaries, provenance = [], [], []
    for (dataset, seed), arms in sorted(pairs.items()):
        a, b = [arms[x].report.arrays.values for x in ("tabicl", "empirical")]
        comparisons = compare_candidates(
            a["common.candidates"],
            a["common.available"],
            b["common.candidates"],
            b["common.available"],
        )
        for comparison in comparisons:
            action_types = {}
            for backend, run in arms.items():
                histories = run.manifest["method_point_diagnostics"][
                    comparison["point"]
                ]["diverse_histories"]
                action_types[backend] = sorted(
                    {step["action_type"] for history in histories for step in history}
                )
            records.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    **comparison,
                    "selected_history_action_types": action_types,
                }
            )
        for backend, run in arms.items():
            summaries.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "backend": backend,
                    **dict(run.report.summary.values),
                    "timings": run.manifest["timings"],
                }
            )
            provenance.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "backend": backend,
                    "run_id": run.manifest["run_id"],
                    "identity": run.manifest["identity"],
                    "sha256": {
                        p.name: digest(p) for p in run.path.iterdir() if p.is_file()
                    },
                }
            )
    aggregates = []
    for dataset in sorted({r["dataset"] for r in records}):
        rr = [r for r in records if r["dataset"] == dataset]
        aggregates.append(
            {
                "dataset": dataset,
                "factual_seed_pairs": len(rr),
                "both_returned": sum(r["set_exact"] is not None for r in rr),
                "identical_sets": sum(r["set_exact"] is True for r in rr),
                "close_sets": sum(r["set_close"] is True for r in rr),
                "tabicl_returned": sum(r["a_count"] for r in rr),
                "empirical_returned": sum(r["b_count"] for r in rr),
                "exact_shared_candidates": sum(r["matched_exact_count"] for r in rr),
                "close_shared_candidates": sum(r["matched_close_count"] for r in rr),
                "same_rank_candidates": sum(r["rank_exact_count"] for r in rr),
            }
        )
    write_json(output / "paired_points.json", records)
    write_json(output / "summaries.json", summaries)
    write_json(output / "provenance.json", provenance)
    write_json(output / "audit.json", aggregates)
    print(json.dumps(aggregates, indent=2), flush=True)


class SearchTrace:
    """Observe owning beam functions with exactly one call to each original."""

    def __init__(self):
        self.events = []
        self.stack = ExitStack()

    def __enter__(self):
        names = {
            "_numerical_trials_for_beam": "numerical_trials",
            "_categorical_trials_for_beam": "categorical_trials",
            "_classifier_outputs": "classifier",
            "_prune_beam": "beam",
            "_quality_eligible_candidates": "quality",
            "_curate_candidate_pool": "pool",
            "select_candidate_subset": "selection",
        }
        for name, stage in names.items():
            original = getattr(beam, name)

            def observe(*args, _original=original, _stage=stage, **kwargs):
                result = _original(*args, **kwargs)
                event = {"stage": _stage}
                if _stage.endswith("_trials"):
                    rows, parents, metadata = result
                    event.update(
                        rows=[np.asarray(r).copy() for r in rows],
                        parents=[p.row.copy() for p in parents],
                        metadata=deepcopy(metadata),
                    )
                elif _stage == "classifier":
                    event.update(
                        rows=np.atleast_2d(args[1]).copy(),
                        probabilities=result[0].copy(),
                        predictions=result[1].copy(),
                    )
                elif _stage == "selection":
                    event.update(
                        rows=np.asarray(args[0]).copy(),
                        indices=np.asarray(result[0]).copy(),
                    )
                else:
                    event.update(
                        input=[s.row.copy() for s in args[0]],
                        rows=[s.row.copy() for s in result],
                    )
                self.events.append(event)
                return result

            self.stack.enter_context(patch.object(beam, name, observe))
        return self

    def __exit__(self, *exc):
        return self.stack.__exit__(*exc)


class SessionTrace:
    def __init__(self, session, records):
        self.session, self.records = session, records

    def __getattr__(self, name):
        return getattr(self.session, name)

    def propose_numerical_batch(self, rows, columns, **kwargs):
        result = self.session.propose_numerical_batch(rows, columns, **kwargs)
        self.records.append(
            {
                "kind": "numerical",
                "rows": np.asarray(rows).copy(),
                "columns": list(columns),
                "values": result.copy(),
                "quantiles": list(kwargs["quantiles"]),
            }
        )
        return result

    def categorical_distribution(self, row, group, **kwargs):
        result = self.session.categorical_distribution(row, group, **kwargs)
        self.records.append(
            {
                "kind": "categorical",
                "row": row.copy(),
                "group": group.name,
                "categories": result.categories.copy(),
                "probabilities": result.probabilities.copy(),
            }
        )
        return result


class BackendTrace:
    def __init__(self, backend, records):
        self.backend, self.records = backend, records

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def for_factual(self, *args, **kwargs):
        return SessionTrace(self.backend.for_factual(*args, **kwargs), self.records)


def common_probe(session, states, schema, quantiles):
    columns = list(schema.actionable_scalars)
    rows = np.repeat(np.stack(states), len(columns), axis=0)
    repeated = columns * len(states)
    if columns:
        raw = session.propose_numerical_batch(
            rows, repeated, quantiles=quantiles, confidences=None, temperature=0.0
        )
        projected = project_candidate_values(
            np.repeat(repeated, len(quantiles)),
            raw.reshape(-1),
            (schema.domains.lower, schema.domains.upper, dict(schema.domains.discrete)),
        ).reshape(raw.shape)
    else:
        raw = projected = np.empty((0, 1, len(quantiles)))
    categories = []
    for row in states:
        for group in schema.actionable_groups:
            dist = session.categorical_distribution(row, group, confidence=None)
            categories.append(
                {
                    "state": row_id(row),
                    "group": group.name,
                    "categories": dist.categories,
                    "probabilities": dist.probabilities,
                }
            )
    return {
        "raw": raw,
        "projected": projected,
        "columns": repeated,
        "states": np.stack(states),
        "categories": categories,
    }


def trace_dataset(root, matrix, output, dataset, count, device):
    """Trace first count E3 factuals in pinned source-index order (seed 42)."""
    output.mkdir(parents=True, exist_ok=False)
    pairs = paired_runs(root, matrix)
    arms = pairs[(dataset, 42)]
    matrix_config = load_matrix_config(matrix)
    specs = {
        r.method.params["foundation"]["backend"]: r
        for r in matrix_config.runs
        if r.dataset.name == dataset and r.seed == 42
    }
    start = time.perf_counter()
    case = _default_case_loader(specs["empirical"]).case
    if (
        case.case_id
        != arms["empirical"].manifest["identity"]["resolved"]["case_fingerprint"]
    ):
        raise ValueError("reconstructed case differs from historical E3 case")
    context = method_context(case)
    prepared, runtimes = {}, {}
    for backend, spec in specs.items():
        runtime = DEFAULT_METHOD_REGISTRY.resolve_runtime(
            spec.method.name,
            spec.method.params,
            cache_paths=matrix_config.execution.cache_paths,
            device=device,
        )
        runtimes[backend] = runtime
        with runtime.activate():
            prepared[backend] = DEFAULT_METHOD_REGISTRY.create(
                spec.method.name, runtime.params
            ).prepare(context)
    metadata = {
        "diagnostic_source_sha256": digest(__file__),
        "case_id": case.case_id,
        "dataset": dataset,
        "seed": 42,
        "selection": "first N source-ordered factuals of authenticated E3 case",
        "factual_indices": case.factuals.indices[:count],
        "device": device,
        "host": platform.node(),
        "count": count,
        "k": 3,
        "checkpoint_content_ids": dict(runtimes["tabicl"].checkpoint_content_ids),
        "backend_implementations": {
            k: v.backend_implementation for k, v in runtimes.items()
        },
        "reference_shape": context.X_reference.shape,
        "feature_names": context.feature_schema.names,
        "numerical_columns": context.feature_schema.actionable_scalars,
        "discrete_columns": list(context.feature_schema.domains.discrete),
        "source_runs": {k: v.manifest["run_id"] for k, v in arms.items()},
        "source_identities": {k: v.manifest["identity"] for k, v in arms.items()},
        "prepare_s": time.perf_counter() - start,
    }
    write_json(output / "metadata.json", metadata)
    summaries = []
    for i in range(min(count, len(case.targets))):
        factual = case.factuals.values[i]
        target = int(case.targets[i])
        request = GenerationRequest(factual[None], np.array([target]), 3, 42)
        state_map = {row_id(factual): factual.copy()}
        generated, stats = {}, {}
        for backend in ("empirical", "tabicl"):
            records = []
            method = replace(
                prepared[backend],
                backend=BackendTrace(prepared[backend].backend, records),
            )
            started = time.perf_counter()
            with runtimes[backend].activate(), SearchTrace() as trace:
                result = method.generate(request)
            duration = time.perf_counter() - started
            generated[backend] = result
            # One earliest distinct intermediate query from each trajectory.
            for record in records:
                if record["kind"] != "numerical":
                    continue
                alternatives = [r for r in record["rows"] if row_id(r) not in state_map]
                if alternatives:
                    state_map[row_id(alternatives[0])] = alternatives[0].copy()
                    break
            purity = None
            if i == 0:
                with runtimes[backend].activate():
                    plain = prepared[backend].generate(request)
                purity = bool(
                    np.array_equal(result.candidates, plain.candidates, equal_nan=True)
                    and np.array_equal(result.available, plain.available)
                )
                if not purity:
                    raise ValueError("trace changes generation output")
            historical = arms[backend].report.arrays.values
            historical_match = bool(
                np.array_equal(
                    result.candidates[0],
                    historical["common.candidates"][i],
                    equal_nan=True,
                )
            )
            stats[backend] = {
                "generate_with_trace_s": duration,
                "trace_purity_exact": purity,
                "historical_candidates_exact": historical_match,
                "available": int(result.available.sum()),
                "point_diagnostics": dict(result.point_diagnostics[0]),
            }
            write_json(
                output / f"{i:02d}_{backend}_trace.json",
                {
                    "proposals": records,
                    "search": trace.events,
                    "candidates": result.candidates,
                    "available": result.available,
                },
            )
            print(
                f"{dataset} point={i} {backend} returned={result.available.sum()} "
                f"trace_s={duration:.2f} historical_exact={historical_match}",
                flush=True,
            )

        states = list(state_map.values())
        # A legal perturbation while keeping the factual-specific context fixed.
        probe = factual.copy()
        changed_column = None
        schema = context.feature_schema
        for column in schema.actionable_scalars:
            bounds = [schema.domains.lower[column], schema.domains.upper[column]]
            value = max(bounds, key=lambda v: abs(v - factual[column]))
            if value != factual[column]:
                probe[column] = value
                changed_column = column
                break
        if changed_column is None:
            raise ValueError("conditioning probe requires a mutable numerical feature")
        states.append(probe)
        probes = {}
        for backend in ("empirical", "tabicl"):
            with runtimes[backend].activate():
                session = prepared[backend].backend.for_factual(
                    factual, target, seed=42
                )
                probes[backend] = common_probe(
                    session,
                    states,
                    schema,
                    prepared[backend].config.search.candidate_quantiles,
                )
            write_json(output / f"{i:02d}_{backend}_common.json", probes[backend])
        a, b = probes["tabicl"], probes["empirical"]
        raw_diff = np.abs(a["raw"] - b["raw"])
        projected_diff = np.abs(a["projected"] - b["projected"])
        cols = len(schema.actionable_scalars)
        sensitivity = {}
        for backend, values in probes.items():
            raw = values["raw"].reshape(len(states), cols, -1)
            eligible = np.asarray(schema.actionable_scalars) != changed_column
            delta = np.abs(raw[0, eligible] - raw[-1, eligible])
            sensitivity[backend] = {
                "changed_quantiles": int((delta > 1e-8).sum()),
                "total_quantiles": int(delta.size),
                "max_abs_change": float(delta.max()),
            }
        comparison = compare_candidates(
            generated["tabicl"].candidates,
            generated["tabicl"].available,
            generated["empirical"].candidates,
            generated["empirical"].available,
        )[0]
        summary = {
            "point": i,
            "source_index": int(case.factuals.indices[i]),
            "target": target,
            "states": len(states),
            "stats": stats,
            "raw_quantiles": int(raw_diff.size),
            "raw_differing": int((raw_diff > 1e-8).sum()),
            "projected_differing": int((projected_diff > 1e-8).sum()),
            "differences_erased_by_projection": int(
                ((raw_diff > 1e-8) & (projected_diff <= 1e-8)).sum()
            ),
            "raw_max_abs_difference": float(raw_diff.max()),
            "perturbed_column": changed_column,
            "sensitivity": sensitivity,
            "final_comparison": comparison,
        }
        summaries.append(summary)
        write_json(output / "summary.json", summaries)
        print(
            f"{dataset} point={i} raw_differing={summary['raw_differing']}/"
            f"{summary['raw_quantiles']} erased="
            f"{summary['differences_erased_by_projection']}",
            flush=True,
        )
    write_json(
        output / "COMPLETE.json",
        {
            "points": len(summaries),
            "total_s": time.perf_counter() - start,
            "files": {p.name: digest(p) for p in output.iterdir() if p.is_file()},
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["audit", "trace"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--matrix",
        type=Path,
        default=Path(
            "experiments/zeroshot_cf/configs/matrices/campaign_e3_backend.yaml"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        choices=["heloc", "adult_census", "give_me_some_credit"],
        default="heloc",
    )
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.mode == "audit":
        audit(args.root, args.matrix, args.output)
    else:
        trace_dataset(
            args.root, args.matrix, args.output, args.dataset, args.count, args.device
        )


if __name__ == "__main__":
    main()
