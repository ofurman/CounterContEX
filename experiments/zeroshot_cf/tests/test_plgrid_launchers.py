"""Shell-level contracts for the Helios clean-E3 launcher."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

SUITE_ROOT = Path(__file__).resolve().parents[1]
PLGRID = SUITE_ROOT / "plgrid"


def test_submit_injects_allocation_and_forwards_test_only(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    captured = tmp_path / "sbatch-args"
    fake_sbatch = fake_bin / "sbatch"
    fake_sbatch.write_text(
        '#!/usr/bin/env bash\nprintf \'%s\\n\' "$@" > "$SBATCH_CAPTURE"\n'
    )
    fake_sbatch.chmod(0o755)
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "SBATCH_CAPTURE": str(captured),
        "PLG_ACCOUNT": "test-account",
        "PLG_PARTITION": "test-partition",
    }

    completed = subprocess.run(
        [
            "bash",
            str(PLGRID / "submit.sh"),
            "--test-only",
            str(PLGRID / "run_e3.sbatch"),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr
    arguments = captured.read_text()
    assert "--account=test-account\n" in arguments
    assert "--partition=test-partition\n" in arguments
    assert "--test-only\n" in arguments


def test_e3_job_keeps_allocation_out_of_headers_and_runs_strict_outputs():
    source = (PLGRID / "run_e3.sbatch").read_text()
    assert "#SBATCH --account" not in source
    assert "#SBATCH --partition" not in source
    assert "cli aggregate" in source
    assert "cli analyze-e3" in source
    assert "e3_clean_backend_analysis" in source
