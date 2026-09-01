"""Generic offline-safe counterfactual benchmark command line interface."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path


def _config(path: str):
    from experiments.zeroshot_cf.orchestration.matrix import load_matrix_config

    return load_matrix_config(Path(path))


def _run_matrix(path: str, *, resume: bool, dry_run: bool, single: bool) -> int:
    config = _config(path)
    if single and len(config.runs) != 1:
        raise ValueError("single requires a config that expands to exactly one cell")
    if dry_run:
        for spec in config.runs:
            print(json.dumps({"cell_id": spec.cell_id, **spec.scientific_payload()}))
        return 0
    from experiments.zeroshot_cf.orchestration.runner import GenericRunner

    runner = GenericRunner(config.execution)
    outcomes = runner.run_all(config.runs, resume=resume)
    for outcome in outcomes:
        status = "skipped" if outcome.skipped else "completed"
        print(f"{status} {outcome.run_id} {outcome.spec.method.name}")
    return 0


def _aggregate(path: str) -> int:
    config = _config(path)
    from experiments.zeroshot_cf.orchestration.artifacts import ArtifactStore

    output = config.execution.output_root / "aggregate_summary.csv"
    rows = ArtifactStore(config.execution.output_root).aggregate_expected(
        config.expected_cells,
        output=output,
    )
    print(f"aggregated {len(rows)} cells into {output}")
    return 0


def _analyze(path: str, output: str | None) -> int:
    config = _config(path)
    from experiments.zeroshot_cf.analysis.builders import build_all

    destination = Path(output) if output else config.execution.output_root / "analysis"
    products = build_all(config.execution.output_root, Path(path), destination)
    print(f"wrote {len(products)} analysis products into {destination}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    single = commands.add_parser("single", help="run one configured cell")
    single.add_argument("--config", required=True)
    matrix = commands.add_parser("matrix", help="run a configured matrix")
    matrix.add_argument("--config", required=True)
    matrix.add_argument("--resume", action="store_true")
    matrix.add_argument("--dry-run", action="store_true")
    aggregate = commands.add_parser("aggregate", help="aggregate a configured matrix")
    aggregate.add_argument("--config", required=True)
    analyze = commands.add_parser("analyze", help="build paper tables and figures")
    analyze.add_argument("--config", required=True)
    analyze.add_argument("--output")
    commands.add_parser("list-methods", help="list registered method names")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "single":
        return _run_matrix(args.config, resume=False, dry_run=False, single=True)
    if args.command == "matrix":
        return _run_matrix(
            args.config,
            resume=args.resume,
            dry_run=args.dry_run,
            single=False,
        )
    if args.command == "aggregate":
        return _aggregate(args.config)
    if args.command == "analyze":
        return _analyze(args.config, args.output)
    if args.command == "list-methods":
        from experiments.zeroshot_cf.methods.registry import DEFAULT_METHOD_REGISTRY

        print("\n".join(DEFAULT_METHOD_REGISTRY.names()))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
