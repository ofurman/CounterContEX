"""Translation and dependency gates for numbered compatibility shims."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
from experiments.zeroshot_cf import (
    exp9_countercontex_benchmark as exp9,
)
from experiments.zeroshot_cf import (
    exp11_nice_nun_baseline as exp11,
)
from experiments.zeroshot_cf import (
    exp12_optimization_baselines as exp12,
)
from experiments.zeroshot_cf import (
    exp13_dice_baseline as exp13,
)
from experiments.zeroshot_cf import (
    exp14_face_baseline as exp14,
)
from experiments.zeroshot_cf.orchestration import compat_cli


@pytest.mark.parametrize(
    "module",
    (exp9, exp11, exp12, exp13, exp14),
)
def test_numbered_entry_points_are_thin_translation_shims(module) -> None:
    source = Path(module.__file__).read_text()
    tree = ast.parse(source)
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert not any(
        name.endswith(
            (".data", ".evaluation.evaluator", ".metrics_harness", ".runner_compat")
        )
        or name.endswith(".tabicl_runtime")
        for name in imported
    )
    limit = 340 if module is exp9 else 150
    assert len(source.splitlines()) <= limit
    assert callable(module.prepare_benchmark_context)


def test_baseline_shims_translate_every_method_specific_setting() -> None:
    nice = exp11._spec(
        "heloc",
        max_test=3,
        validation_fraction=0.2,
        drop_heloc_all_minus9=False,
    )
    growing = exp12._spec(
        "bank_marketing",
        "growing_spheres",
        max_test=4,
        validation_fraction=0.3,
        drop_heloc_all_minus9=True,
        random_state=17,
        sphere_candidates=99,
    )
    dice = exp13._spec(
        "give_me_some_credit",
        max_test=5,
        max_iterations=11,
        search_restarts=2,
        stopping_threshold=0.75,
        validation_fraction=0.2,
        drop_heloc_all_minus9=True,
    )
    face = exp14._spec(
        "lending_club",
        max_test=6,
        n_neighbors=7,
        density_power=2.0,
        tau=0.85,
        validation_fraction=0.2,
        drop_heloc_all_minus9=True,
    )

    assert nice.method.name == "nice" and nice.method.params == {}
    assert growing.method.params == {"n_candidates": 99}
    assert growing.seed == 17
    assert dice.method.params == {
        "max_iterations": 11,
        "search_restarts": 2,
        "stopping_threshold": 0.75,
    }
    assert dice.evaluation.probability_threshold == 0.75
    assert face.method.params == {
        "n_neighbors": 7,
        "density_power": 2.0,
        "tau": 0.85,
    }


@pytest.mark.parametrize("max_test", (-1, None))
def test_legacy_unlimited_max_test_is_normalized(max_test: int | None) -> None:
    spec = compat_cli.legacy_run_spec(
        "heloc",
        "nice",
        max_test=max_test,
        validation_fraction=0.2,
        drop_heloc_all_minus9=True,
        probability_threshold=0.5,
    )

    assert spec.protocol.max_test is None


def test_programmatic_run_dataset_signatures_remain_compatible() -> None:
    assert tuple(inspect.signature(exp11.run_dataset).parameters) == (
        "dataset_name",
        "max_test",
        "validation_fraction",
        "drop_heloc_all_minus9",
        "results_dir",
    )
    assert tuple(inspect.signature(exp12.run_dataset).parameters) == (
        "dataset_name",
        "method",
        "max_test",
        "validation_fraction",
        "drop_heloc_all_minus9",
        "random_state",
        "sphere_candidates",
        "results_dir",
    )
    assert tuple(inspect.signature(exp13.run_dataset).parameters) == (
        "dataset_name",
        "max_test",
        "max_iterations",
        "search_restarts",
        "stopping_threshold",
        "validation_fraction",
        "drop_heloc_all_minus9",
        "results_dir",
    )
    assert tuple(inspect.signature(exp14.run_dataset).parameters) == (
        "dataset_name",
        "max_test",
        "n_neighbors",
        "density_power",
        "tau",
        "validation_fraction",
        "drop_heloc_all_minus9",
        "results_dir",
    )


def test_compat_runner_records_selected_slurm_environment(
    monkeypatch, tmp_path
) -> None:
    captured = {}

    class _Runner:
        def __init__(self, execution, *, store):
            captured["execution"] = execution
            captured["store"] = store

        def run(self, spec, *, resume):
            captured["spec"] = spec
            captured["resume"] = resume
            return SimpleNamespace(
                stored=SimpleNamespace(
                    manifest={
                        "timings": {
                            "prepare_s": 1.0,
                            "generate_s": 2.0,
                            "evaluate_s": 3.0,
                            "write_s": 4.0,
                            "total_s": 10.0,
                        }
                    }
                )
            )

    monkeypatch.setenv("COUNTERCONTEX_SLURM_WALLTIME", "10:00:00")
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(compat_cli, "GenericRunner", _Runner)
    monkeypatch.setattr(
        compat_cli,
        "read_legacy_metrics",
        lambda *args, **kwargs: {"dataset": "heloc"},
    )
    spec = compat_cli.legacy_run_spec(
        "heloc",
        "nice",
        max_test=1,
        validation_fraction=0.2,
        drop_heloc_all_minus9=True,
        probability_threshold=0.5,
    )

    row = compat_cli.run_legacy_dataset(spec, results_dir=tmp_path)

    assert row == {
        "dataset": "heloc",
        "prepare_s": 1.0,
        "generate_s": 2.0,
        "evaluate_s": 3.0,
        "write_s": 4.0,
        "total_s": 10.0,
    }
    assert captured["execution"].environment == {
        "COUNTERCONTEX_SLURM_WALLTIME": "10:00:00",
        "SLURM_JOB_ID": "123",
        "HF_HUB_OFFLINE": "1",
    }
    assert captured["execution"].legacy_export is True
    assert captured["resume"] is True
