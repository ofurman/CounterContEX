#  Copyright (c) Prior Labs GmbH 2026.

"""Freeze the pinned Exp9 dataset contract against the vendored CEL inputs."""

from __future__ import annotations

from experiments.zeroshot_cf.tests._retained_contract import (
    build_dataset_contract_fixture,
    read_dataset_contract_fixture,
)


def test_dataset_contract_matches_pinned_cel_inputs() -> None:
    """Every recorded split, feature, actionability, and hash must reproduce."""
    assert build_dataset_contract_fixture() == read_dataset_contract_fixture()
