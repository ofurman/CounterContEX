"""Offline entry-point, artifact, and dependency baselines for architecture v1."""

from __future__ import annotations

import ast
import csv
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from experiments.zeroshot_cf.benchmark_protocol import DATASETS, dataset_result_paths

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURES = Path(__file__).parent / "fixtures" / "architecture_v1"


def _fixture(name: str) -> dict[str, object]:
    return json.loads((FIXTURES / name).read_text())


def test_public_generator_import_path_remains_callable() -> None:
    module_name, attribute = _fixture("compatibility.json")["generator_import"].rsplit(
        ".", 1
    )
    assert callable(getattr(importlib.import_module(module_name), attribute))


def test_legacy_method_inventory_has_stable_artifact_names() -> None:
    fixture = _fixture("compatibility.json")
    methods = fixture["methods"]
    assert set(methods) == {
        "dicoflex",
        "nice",
        "wachter",
        "growing_spheres",
        "dice",
        "face",
    }
    common_point_columns = set(fixture["common_point_columns"])
    for method_name, contract in methods.items():
        assert contract["legacy_method_ids"]
        source_path = (
            REPO_ROOT
            / "experiments"
            / "zeroshot_cf"
            / f"{contract['source_module']}.py"
        )
        source = source_path.read_text()
        tree = ast.parse(source)
        string_literals = {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        dict_keys = {
            key.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Dict)
            for key in node.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        arrays_calls = [
            keyword.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            for keyword in node.keywords
            if keyword.arg == "arrays" and isinstance(keyword.value, ast.Dict)
        ]
        assert len(arrays_calls) == 1
        actual_npz_keys = {
            key.value
            for key in arrays_calls[0].keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }

        if method_name == "dicoflex":
            from experiments.zeroshot_cf.generator import CF_MODES

            actual_method_ids = {
                "tabicl_v2_diverse_dpp",
                *(f"tabicl_v2_{mode}" for mode in CF_MODES),
            }
        elif contract["source_module"] == "exp12_optimization_baselines":
            from experiments.zeroshot_cf.exp12_optimization_baselines import METHODS

            actual_method_ids = {method_name} if method_name in METHODS else set()
        else:
            actual_method_ids = set(contract["legacy_method_ids"]) & string_literals

        assert set(contract["legacy_method_ids"]) == actual_method_ids
        if method_name in {"wachter", "growing_spheres"}:
            assert contract["filename_stem"] == f"exp12_{method_name}"
            assert 'f"exp12_{method}"' in source
        else:
            assert contract["filename_stem"] in string_literals
        assert set(contract["required_npz_keys"]) == actual_npz_keys
        assert common_point_columns <= dict_keys

        paths = dataset_result_paths(
            Path("results"), contract["filename_stem"], "heloc"
        )
        assert (
            paths.metrics_csv.name == f"{contract['filename_stem']}_heloc_metrics.csv"
        )
        assert paths.points_csv.name == f"{contract['filename_stem']}_heloc_points.csv"
        assert paths.arrays_npz.name == f"{contract['filename_stem']}_heloc_arrays.npz"
        assert len(contract["required_npz_keys"]) == len(
            set(contract["required_npz_keys"])
        )


def _checkpoint_spy_environment(tmp_path: Path) -> tuple[dict[str, str], Path]:
    spy_dir = tmp_path / "checkpoint_spy"
    spy_dir.mkdir()
    marker = spy_dir / "checkpoint-loaded"
    (spy_dir / "sitecustomize.py").write_text(
        """\
import os
from pathlib import Path

import huggingface_hub
import importlib.abc
import joblib
import sys
import torch

def _reject_checkpoint_load(*args, **kwargs):
    Path(os.environ["CHECKPOINT_SPY_MARKER"]).write_text(repr(args[:1]))
    raise RuntimeError("CHECKPOINT_LOAD_SPY")

huggingface_hub.hf_hub_download = _reject_checkpoint_load
joblib.load = _reject_checkpoint_load
torch.load = _reject_checkpoint_load

class _RejectModelImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        roots = set(os.environ["MODEL_IMPORT_SPY_ROOTS"].split(","))
        if fullname.split(".", 1)[0] in roots:
            Path(os.environ["CHECKPOINT_SPY_MARKER"]).write_text(fullname)
            raise RuntimeError(f"MODEL_IMPORT_SPY: {fullname}")
        return None

sys.meta_path.insert(0, _RejectModelImports())
"""
    )
    pythonpath = os.pathsep.join(
        value for value in (str(spy_dir), os.environ.get("PYTHONPATH")) if value
    )
    environment = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "CHECKPOINT_SPY_MARKER": str(marker),
        "MODEL_IMPORT_SPY_ROOTS": ",".join(
            _fixture("compatibility.json")["optional_model_imports"]
        ),
        "PYTHONPATH": pythonpath,
    }
    return environment, marker


@pytest.mark.parametrize(
    "command",
    _fixture("compatibility.json")["cli_commands"],
    ids=lambda command: command["module"].rsplit(".", 1)[-1],
)
def test_retained_cli_help_is_offline_and_does_not_load_checkpoints(
    command, tmp_path: Path
) -> None:
    environment, marker = _checkpoint_spy_environment(tmp_path)
    completed = subprocess.run(
        [sys.executable, "-m", command["module"], "--help"],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    for token in command["help_tokens"]:
        assert token in completed.stdout
    assert not marker.exists(), marker.read_text() if marker.exists() else ""


def test_exp9_aggregate_cli_is_offline_and_does_not_load_checkpoints(
    tmp_path: Path,
) -> None:
    for dataset in DATASETS:
        path = dataset_result_paths(tmp_path, "exp9_tabicl", dataset).metrics_csv
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["dataset", "method"])
            writer.writeheader()
            writer.writerow({"dataset": dataset, "method": "tabicl_v2_sparse"})

    environment, marker = _checkpoint_spy_environment(tmp_path)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.zeroshot_cf.exp9_dicoflex_benchmark",
            "--dataset",
            "aggregate",
            "--results-dir",
            str(tmp_path),
        ],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert (tmp_path / "exp9_tabicl_all_metrics.csv").is_file()
    assert not marker.exists(), marker.read_text() if marker.exists() else ""


def _internal_edges(source: str) -> set[tuple[str, str]]:
    path = REPO_ROOT / "experiments" / "zeroshot_cf" / f"{source}.py"
    tree = ast.parse(path.read_text())
    edges: set[tuple[str, str]] = set()
    prefix = "experiments.zeroshot_cf."
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.level:
                edges.add((source, node.module.split(".", 1)[0]))
            elif node.module.startswith(prefix):
                edges.add((source, node.module[len(prefix) :].split(".", 1)[0]))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(prefix):
                    edges.add((source, alias.name[len(prefix) :].split(".", 1)[0]))
    return edges


def test_known_forbidden_dependency_edges_can_only_shrink() -> None:
    fixture = _fixture("boundary_edges.json")
    baseline = {
        (entry["source"], entry["target"]) for entry in fixture["known_forbidden_edges"]
    }
    forbidden_targets = set(fixture["forbidden_targets"])
    observed = set().union(
        *(_internal_edges(source) for source in fixture["monitored_sources"])
    )
    relevant = {
        (source, target) for source, target in observed if target in forbidden_targets
    }
    assert relevant <= baseline
