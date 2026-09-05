"""Slice a matrix config down to exactly one (dataset, method, seed) cell.

Picks the Nth cell using the same Cartesian product order the matrix loader
uses (dataset x method x seed; see orchestration/matrix.py), so ``--index N``
always names the same cell as the Nth row of
``cli.py matrix --config <config> --dry-run``. This lets one Slurm array task
run exactly one benchmark cell from a shared matrix config without needing a
separately maintained per-task case file.
"""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path
from typing import Any

import yaml


def _name(entry: Any) -> str:
    return entry if isinstance(entry, str) else str(entry["name"])


def _cells(payload: dict[str, Any]) -> list[tuple[Any, Any, Any]]:
    return list(product(payload["datasets"], payload["methods"], payload["seeds"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--count",
        action="store_true",
        help="print the total number of cells in the config and exit",
    )
    parser.add_argument("--index", type=int)
    parser.add_argument("--output")
    parser.add_argument(
        "--tabicl-cache-dir",
        default=None,
        help="override cache_paths.tabicl in the sliced config",
    )
    parser.add_argument(
        "--device", default=None, help="override device in the sliced config"
    )
    args = parser.parse_args()

    payload = yaml.safe_load(Path(args.config).read_text())
    cells = _cells(payload)

    if args.count:
        print(len(cells))
        return 0

    if args.index is None or args.output is None:
        parser.error("--index and --output are required unless --count is given")

    if not 0 <= args.index < len(cells):
        raise SystemExit(
            f"--index {args.index} out of range for {len(cells)} cells "
            f"({len(payload['datasets'])} datasets x {len(payload['methods'])} "
            f"methods x {len(payload['seeds'])} seeds)"
        )
    dataset, method, seed = cells[args.index]

    sliced = dict(payload)
    sliced["datasets"] = [dataset]
    sliced["methods"] = [method]
    sliced["seeds"] = [seed]
    sliced["suite"] = f"{payload['suite']}_{_name(dataset)}_{_name(method)}_seed{seed}"
    if args.tabicl_cache_dir:
        cache_paths = dict(sliced.get("cache_paths", {}))
        cache_paths["tabicl"] = args.tabicl_cache_dir
        sliced["cache_paths"] = cache_paths
    if args.device:
        sliced["device"] = args.device

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(sliced, sort_keys=False))
    print(
        f"cell index={args.index} dataset={_name(dataset)} "
        f"method={_name(method)} seed={seed} -> {output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
