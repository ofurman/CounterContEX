"""Focused tests for explicit TabPFN v2 checkpoint preconditions."""

from __future__ import annotations

from pathlib import Path

import pytest
from experiments.zeroshot_cf.tabpfn_checkpoints import (
    TABPFN_CLF_FILENAME,
    TABPFN_REG_FILENAME,
    checkpoint_paths,
    require_checkpoints,
)


def test_tabpfn_checkpoint_paths_use_stable_v2_filenames(tmp_path: Path) -> None:
    classifier, regressor = checkpoint_paths(tmp_path)

    assert classifier == tmp_path / TABPFN_CLF_FILENAME
    assert regressor == tmp_path / TABPFN_REG_FILENAME


def test_tabpfn_missing_checkpoints_report_both_paths(tmp_path: Path) -> None:
    classifier, regressor = checkpoint_paths(tmp_path)

    with pytest.raises(FileNotFoundError, match="TabPFN v2 checkpoint") as error:
        require_checkpoints(tmp_path)

    assert str(classifier) in str(error.value)
    assert str(regressor) in str(error.value)


def test_tabpfn_checkpoint_checksum_mismatch_fails_before_inference(
    tmp_path: Path,
) -> None:
    classifier, regressor = checkpoint_paths(tmp_path)
    classifier.parent.mkdir(parents=True, exist_ok=True)
    classifier.write_bytes(b"wrong classifier")
    regressor.write_bytes(b"wrong regressor")

    with pytest.raises(RuntimeError, match="TabPFN v2 checkpoint checksum mismatch"):
        require_checkpoints(tmp_path)
