"""YAML expansion and expected-cell ownership for benchmark matrices."""

from __future__ import annotations

import pytest
from experiments.zeroshot_cf.orchestration.matrix import load_matrix_config


def test_two_by_two_matrix_expands_fully_resolved_unique_specs(tmp_path) -> None:
    path = tmp_path / "matrix.yaml"
    path.write_text(
        """\
schema_version: countercontex.matrix.v1
suite: fixture
output_root: results/fixture
datasets: [one, two]
methods:
  - name: alpha
    params: {strength: 1}
  - name: beta
    variant: tuned
    params: {strength: 2}
seeds: [7]
protocol: {max_test: 2, test_selection: first}
target_model: {name: fixture, params: {version: one}}
evaluation: {probability_threshold: 0.8}
"""
    )

    config = load_matrix_config(path)

    assert len(config.runs) == 4
    assert len(set(config.expected_cells)) == 4
    assert {(run.dataset.name, run.method.name) for run in config.runs} == {
        ("one", "alpha"),
        ("one", "beta"),
        ("two", "alpha"),
        ("two", "beta"),
    }
    assert all(run.seed == 7 for run in config.runs)
    assert all(run.protocol.max_test == 2 for run in config.runs)


def test_matrix_rejects_duplicate_cells_and_unknown_fields(tmp_path) -> None:
    duplicate = tmp_path / "duplicate.yaml"
    duplicate.write_text(
        """\
schema_version: countercontex.matrix.v1
suite: duplicate
output_root: results
datasets: [one, one]
methods: [alpha]
seeds: [1]
"""
    )
    try:
        load_matrix_config(duplicate)
    except ValueError as error:
        assert "duplicate" in str(error)
    else:
        raise AssertionError("duplicate matrix cells were accepted")

    unknown = tmp_path / "unknown.yaml"
    unknown.write_text(
        duplicate.read_text().replace("datasets:", "mystery: 1\ndatasets:")
    )
    try:
        load_matrix_config(unknown)
    except ValueError as error:
        assert "unknown matrix fields" in str(error)
    else:
        raise AssertionError("unknown matrix field was accepted")


def test_toml_matrix_expands_with_the_same_typed_specs(tmp_path) -> None:
    path = tmp_path / "matrix.toml"
    path.write_text(
        """\
schema_version = "countercontex.matrix.v1"
suite = "fixture-toml"
output_root = "results/fixture-toml"
datasets = ["one", "two"]
methods = ["alpha"]
seeds = [7]

[protocol]
max_test = 2
test_selection = "first"

[target_model]
name = "fixture"

[target_model.params]
version = "one"
"""
    )

    config = load_matrix_config(path)

    assert config.suite == "fixture-toml"
    assert len(config.runs) == 2
    assert {run.dataset.name for run in config.runs} == {"one", "two"}
    assert all(run.protocol.test_selection == "first" for run in config.runs)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("seeds: [7]", "seeds: [7.5]"),
        (
            "methods: [alpha]",
            "methods: [{name: alpha, n_counterfactuals: 1.5}]",
        ),
    ],
)
def test_matrix_rejects_fractional_integer_fields(tmp_path, field, replacement) -> None:
    path = tmp_path / "matrix.yaml"
    path.write_text(
        """\
schema_version: countercontex.matrix.v1
suite: fixture
output_root: results
datasets: [one]
methods: [alpha]
seeds: [7]
""".replace(field, replacement)
    )

    with pytest.raises(ValueError, match="must be an integer"):
        load_matrix_config(path)


def test_matrix_rejects_colliding_legacy_exports(tmp_path) -> None:
    path = tmp_path / "matrix.yaml"
    path.write_text(
        """\
schema_version: countercontex.matrix.v1
suite: fixture
output_root: results
datasets: [heloc]
methods: [countercontex]
seeds: [7, 8]
legacy_export: true
"""
    )

    with pytest.raises(ValueError, match="at most one run"):
        load_matrix_config(path)
