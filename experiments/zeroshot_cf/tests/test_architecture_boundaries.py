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
    from experiments.zeroshot_cf.orchestration.v1_contract import V1_CONTRACT

    fixture = _fixture("compatibility.json")
    methods = fixture["methods"]
    assert set(methods) == {
        "countercontex",
        "nice",
        "wachter",
        "growing_spheres",
        "dice",
        "face",
    }
    assert set(V1_CONTRACT) == set(methods)
    common_point_columns = set(fixture["common_point_columns"])
    for method_name, contract in methods.items():
        assert contract["legacy_method_ids"]
        frozen = V1_CONTRACT[method_name]

        if method_name == "countercontex":
            from experiments.zeroshot_cf.generator import CF_MODES

            actual_method_ids = {
                "tabicl_v2_diverse_dpp",
                *(f"tabicl_v2_{mode}" for mode in CF_MODES),
            }
        else:
            actual_method_ids = {frozen["method_id"]}

        assert set(contract["legacy_method_ids"]) == actual_method_ids
        assert contract["filename_stem"] == frozen["stem"]
        assert tuple(contract["required_npz_keys"]) == frozen["npz_keys"]
        assert tuple(contract["summary_columns"]) == frozen["summary_columns"]
        assert tuple(contract["point_columns"]) == frozen["point_columns"]
        assert common_point_columns <= set(frozen["point_columns"])

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
            "experiments.zeroshot_cf.exp9_countercontex_benchmark",
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
    if not path.exists():
        return set()
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


def test_cutover_removed_obsolete_runtime_and_reporting_implementations() -> None:
    suite = REPO_ROOT / "experiments" / "zeroshot_cf"
    removed = {"reporting.py", "runner_compat.py"}
    assert not any((suite / name).exists() for name in removed)
    for path in suite.rglob("*.py"):
        relative = path.relative_to(suite)
        if relative.parts[0] in {".venv", "tests", "vendor"}:
            continue
        source = path.read_text()
        assert not any(name.removesuffix(".py") in source for name in removed)

    runtime = suite / "tabicl_runtime.py"
    tree = ast.parse(runtime.read_text())
    assert {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    } == set()
    assert any(
        isinstance(node, ast.ImportFrom)
        and node.module == "experiments.zeroshot_cf.orchestration.tabicl_runtime_compat"
        for node in tree.body
    )


def test_generic_runner_contains_no_concrete_method_or_backend_policy() -> None:
    source = (
        (REPO_ROOT / "experiments" / "zeroshot_cf" / "orchestration" / "runner.py")
        .read_text()
        .lower()
    )
    assert "countercontex" not in source
    assert "tabicl" not in source
    assert "empirical" not in source


def test_exp8_numbered_entrypoint_only_translates_and_delegates() -> None:
    path = REPO_ROOT / "experiments" / "zeroshot_cf" / "exp8_tabicl_cf.py"
    source = path.read_text()
    tree = ast.parse(source)
    functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert functions.isdisjoint(
        {"_legacy_info", "_legacy_metrics", "_legacy_row", "_canonical_run"}
    )
    assert not any(
        isinstance(node, ast.Import)
        and any(alias.name == "csv" for alias in node.names)
        for node in tree.body
    )

    compatibility_imports = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module == "experiments.zeroshot_cf.orchestration.exp8_compat"
        for alias in node.names
    }
    assert compatibility_imports == {"export_exp8_result", "load_exp8_result"}

    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert {
        "legacy_run_spec",
        "run_legacy_dataset_with_context",
        "load_exp8_result",
        "export_exp8_result",
    } <= calls
    assert "DictWriter" not in calls
    assert not any(
        isinstance(node, ast.Attribute)
        and node.attr in {"open", "mkdir", "write_text", "write_bytes"}
        for node in ast.walk(tree)
    )
    assert "proximity_l2_jaccard" not in source
    assert "exp8_tabicl_" not in source
