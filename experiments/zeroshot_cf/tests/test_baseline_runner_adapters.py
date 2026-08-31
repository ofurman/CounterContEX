"""Numbered baseline shims preserve method selection and v1 artifact names."""

from __future__ import annotations

from pathlib import Path

import pytest
from experiments.zeroshot_cf.orchestration.legacy import generic_legacy_paths


@pytest.mark.parametrize(
    ("module_name", "method_name", "stem", "kwargs", "expected_params"),
    [
        ("exp11_nice_nun_baseline", "nice", "exp11_nice_nun", {}, {}),
        (
            "exp12_optimization_baselines",
            "wachter",
            "exp12_wachter",
            {"method": "wachter"},
            {},
        ),
        (
            "exp12_optimization_baselines",
            "growing_spheres",
            "exp12_growing_spheres",
            {"method": "growing_spheres", "sphere_candidates": 64},
            {"n_candidates": 64},
        ),
        (
            "exp13_dice_baseline",
            "dice",
            "exp13_dice_genetic",
            {"max_iterations": 17, "search_restarts": 2},
            {
                "max_iterations": 17,
                "search_restarts": 2,
                "stopping_threshold": 0.5,
            },
        ),
        (
            "exp14_face_baseline",
            "face",
            "exp14_face_knn",
            {"n_neighbors": 3, "density_power": 2.0},
            {"n_neighbors": 3, "density_power": 2.0, "tau": 0.5},
        ),
    ],
)
def test_numbered_baseline_run_dataset_delegates_to_generic_lifecycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    method_name: str,
    stem: str,
    kwargs: dict,
    expected_params: dict,
) -> None:
    module = __import__(
        f"experiments.zeroshot_cf.{module_name}", fromlist=["run_dataset"]
    )
    captured = {}
    expected_row = {"dataset": "heloc", "method": method_name}

    def fake_run(spec, *, results_dir, tabicl_cache_dir=None):
        captured.update(
            spec=spec,
            results_dir=results_dir,
            tabicl_cache_dir=tabicl_cache_dir,
        )
        return expected_row

    monkeypatch.setattr(module, "run_legacy_dataset", fake_run)

    row = module.run_dataset("heloc", max_test=1, results_dir=tmp_path, **kwargs)

    assert row is expected_row
    assert captured["results_dir"] == tmp_path
    assert captured["spec"].dataset.name == "heloc"
    assert captured["spec"].protocol.max_test == 1
    assert captured["spec"].method.name == method_name
    assert dict(captured["spec"].method.params) == expected_params
    paths = generic_legacy_paths(tmp_path, method_name, "heloc")
    assert paths.metrics_csv.name == f"{stem}_heloc_metrics.csv"
    assert paths.points_csv.name == f"{stem}_heloc_points.csv"
    assert paths.arrays_npz.name == f"{stem}_heloc_arrays.npz"
