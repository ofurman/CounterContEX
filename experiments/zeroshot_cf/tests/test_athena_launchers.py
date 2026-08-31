"""Static and shell-level contracts for the retained Athena launcher."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

SUITE_ROOT = Path(__file__).resolve().parents[1]
ATHENA = SUITE_ROOT / "athena"


def _run_submit(tmp_path: Path, *, walltime: str | None) -> str:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    captured = tmp_path / "sbatch-args"
    fake_sbatch = fake_bin / "sbatch"
    fake_sbatch.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$@\" > \"$SBATCH_CAPTURE\"\n"
    )
    fake_sbatch.chmod(0o755)
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "PROJECT_DIR": str(SUITE_ROOT.parents[1]),
        "SUITE_DIR": str(SUITE_ROOT),
        "SBATCH_CAPTURE": str(captured),
    }
    if walltime is not None:
        environment["WALLTIME"] = walltime
    else:
        environment.pop("WALLTIME", None)
    completed = subprocess.run(
        ["bash", str(ATHENA / "submit_exp9_dicoflex.sh")],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert completed.returncode == 0, completed.stderr
    return captured.read_text()


def test_submit_uses_ten_hour_default_and_exports_manifest_metadata(tmp_path):
    arguments = _run_submit(tmp_path, walltime=None)
    assert "--array=0-3\n" in arguments
    assert "--time=10:00:00\n" in arguments
    assert (
        "--export=ALL,COUNTERCONTEX_SLURM_WALLTIME=10:00:00\n" in arguments
    )


def test_submit_propagates_configurable_walltime(tmp_path):
    arguments = _run_submit(tmp_path, walltime="12:34:56")
    assert "--time=12:34:56\n" in arguments
    assert (
        "--export=ALL,COUNTERCONTEX_SLURM_WALLTIME=12:34:56\n" in arguments
    )


def test_array_script_has_direct_submission_fallback_and_exp9_shim():
    source = (ATHENA / "exp9_dicoflex_array.sbatch").read_text()
    assert "#SBATCH --time=10:00:00" in source
    fallback = (
        'COUNTERCONTEX_SLURM_WALLTIME="'
        '${COUNTERCONTEX_SLURM_WALLTIME:-10:00:00}"'
    )
    assert fallback in source
    assert "-m experiments.zeroshot_cf.exp9_dicoflex_benchmark" in source
