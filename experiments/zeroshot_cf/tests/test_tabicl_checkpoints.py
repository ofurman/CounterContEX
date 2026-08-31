#  Copyright (c) Prior Labs GmbH 2026.

"""Focused tests for explicit TabICL checkpoint preconditions."""

from __future__ import annotations

from pathlib import Path

import pytest

from experiments.zeroshot_cf.tabicl_checkpoints import (
    TABICL_CLF_FILENAME,
    TABICL_REG_FILENAME,
    checkpoint_paths,
    require_checkpoints,
)


def test_checkpoint_paths_use_stable_local_filenames(tmp_path: Path) -> None:
    """The public default paths stay explicit and repo-local."""
    clf_path, reg_path = checkpoint_paths(tmp_path)

    assert clf_path == tmp_path / TABICL_CLF_FILENAME
    assert reg_path == tmp_path / TABICL_REG_FILENAME


def test_require_checkpoints_reports_exact_missing_paths(tmp_path: Path) -> None:
    """Missing staged weights are reported as a measurable precondition failure."""
    clf_path, reg_path = checkpoint_paths(tmp_path)

    with pytest.raises(FileNotFoundError, match="TabICL checkpoint\\(s\\) are not staged"):
        require_checkpoints(tmp_path)

    message = str(
        pytest.raises(FileNotFoundError, require_checkpoints, tmp_path).value
    )
    assert str(clf_path) in message
    assert str(reg_path) in message


def test_require_checkpoints_rejects_checksum_mismatch(tmp_path: Path) -> None:
    """Corrupted staged weights must fail before any inference starts."""
    clf_path, reg_path = checkpoint_paths(tmp_path)
    clf_path.parent.mkdir(parents=True, exist_ok=True)
    clf_path.write_bytes(b"wrong classifier checkpoint")
    reg_path.write_bytes(b"wrong regressor checkpoint")

    with pytest.raises(RuntimeError, match="TabICL checkpoint checksum mismatch"):
        require_checkpoints(tmp_path)
