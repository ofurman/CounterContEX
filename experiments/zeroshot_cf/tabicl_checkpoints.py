#  Copyright (c) Prior Labs GmbH 2026.

"""Local checkpoint staging for the TabICL counterfactual backend.

TabICL normally downloads its classifier and regressor checkpoints from the
Hugging Face cache.  The counterfactual experiments instead stage both files in
one explicit, repo-local directory so offline runs have no hidden dependency on
the state of a user's global cache.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

_DEFAULT_LOCAL_CACHE = Path(__file__).parent / "models" / "tabicl"
_TABICL_HF_REPO = "jingang/TabICL"

TABICL_LOCAL_CACHE = Path(
    os.environ.get("TABICL_LOCAL_CACHE", str(_DEFAULT_LOCAL_CACHE))
)
TABICL_DEVICE = os.environ.get("TABICL_DEVICE", "auto")

TABICL_CLF_FILENAME = "tabicl-classifier-v2-20260212.ckpt"
TABICL_REG_FILENAME = "tabicl-regressor-v2-20260212.ckpt"
_CHECKPOINT_SHA256 = {
    TABICL_CLF_FILENAME: (
        "bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0"
    ),
    TABICL_REG_FILENAME: (
        "0db9cb538f114e79026bf08f45f41ad8dd7ad2de2aaca9a5ca8cd3bd9748ae7a"
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_paths(cache_dir: Path | None = None) -> tuple[Path, Path]:
    """Return the expected local classifier and regressor checkpoint paths."""
    root = Path(cache_dir or TABICL_LOCAL_CACHE)
    return root / TABICL_CLF_FILENAME, root / TABICL_REG_FILENAME


def require_checkpoints(cache_dir: Path | None = None) -> tuple[Path, Path]:
    """Return local paths, raising an actionable error if either is missing."""
    clf_path, reg_path = checkpoint_paths(cache_dir)
    missing = [path for path in (clf_path, reg_path) if not path.is_file()]
    if missing:
        missing_text = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "TabICL checkpoint(s) are not staged:\n"
            f"{missing_text}\n"
            "Run `python -m experiments.zeroshot_cf.tabicl_checkpoints` once "
            "with network access, then rerun the experiment offline."
        )
    invalid = [
        path
        for path in (clf_path, reg_path)
        if _sha256(path) != _CHECKPOINT_SHA256[path.name]
    ]
    if invalid:
        invalid_text = "\n".join(f"  - {path}" for path in invalid)
        raise RuntimeError(
            "TabICL checkpoint checksum mismatch:\n"
            f"{invalid_text}\n"
            "Restage or retransmit the affected checkpoint before inference."
        )
    return clf_path, reg_path


def stage_checkpoints(cache_dir: Path | None = None) -> tuple[Path, Path]:
    """Download both TabICLv2 checkpoints to explicit local paths.

    Downloading directly through ``huggingface_hub`` avoids instantiating either
    model merely to stage weights. Existing complete files are reused.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to stage the TabICL checkpoints."
        ) from exc

    clf_path, reg_path = checkpoint_paths(cache_dir)
    clf_path.parent.mkdir(parents=True, exist_ok=True)

    for destination in (clf_path, reg_path):
        verified = (
            destination.is_file()
            and _sha256(destination) == _CHECKPOINT_SHA256[destination.name]
        )
        if verified:
            print(f"Reusing verified {destination}")
            continue
        print(f"Staging {destination.name} at {destination}")
        downloaded = Path(
            hf_hub_download(
                repo_id=_TABICL_HF_REPO,
                filename=destination.name,
                local_dir=destination.parent,
                force_download=destination.exists(),
            )
        )
        if downloaded.resolve() != destination.resolve():
            downloaded.replace(destination)

    require_checkpoints(cache_dir)
    print("TabICL checkpoints staged successfully.")
    return clf_path, reg_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage local TabICLv2 checkpoints")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help=f"Destination directory (default: {TABICL_LOCAL_CACHE})",
    )
    args = parser.parse_args()
    stage_checkpoints(args.cache_dir)


if __name__ == "__main__":
    main()
