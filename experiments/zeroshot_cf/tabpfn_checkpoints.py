"""Verified local checkpoint staging for the TabPFN v2 proposal backend."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

_DEFAULT_LOCAL_CACHE = Path(__file__).parent / "models" / "tabpfn"

TABPFN_LOCAL_CACHE = Path(
    os.environ.get("TABPFN_LOCAL_CACHE", str(_DEFAULT_LOCAL_CACHE))
)
TABPFN_DEVICE = os.environ.get("TABPFN_DEVICE", "auto")

TABPFN_CLF_FILENAME = "tabpfn-v2-classifier-finetuned-zk73skhh.ckpt"
TABPFN_REG_FILENAME = "tabpfn-v2-regressor.ckpt"
_CHECKPOINT_REPOS = {
    TABPFN_CLF_FILENAME: "Prior-Labs/TabPFN-v2-clf",
    TABPFN_REG_FILENAME: "Prior-Labs/TabPFN-v2-reg",
}
_CHECKPOINT_SHA256 = {
    TABPFN_CLF_FILENAME: (
        "cf8c519c01eaf1613ee91239006d57b1c806ff5f23ac1aeb1315ba1015210e49"
    ),
    TABPFN_REG_FILENAME: (
        "2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736"
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_paths(cache_dir: Path | None = None) -> tuple[Path, Path]:
    root = Path(cache_dir or TABPFN_LOCAL_CACHE)
    return root / TABPFN_CLF_FILENAME, root / TABPFN_REG_FILENAME


def require_checkpoints(cache_dir: Path | None = None) -> tuple[Path, Path]:
    paths = checkpoint_paths(cache_dir)
    missing = [path for path in paths if not path.is_file()]
    if missing:
        lines = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "TabPFN v2 checkpoint(s) are not staged:\n"
            f"{lines}\n"
            "Run `python -m experiments.zeroshot_cf.tabpfn_checkpoints` once "
            "with network access, then rerun offline."
        )
    invalid = [
        path for path in paths if _sha256(path) != _CHECKPOINT_SHA256[path.name]
    ]
    if invalid:
        lines = "\n".join(f"  - {path}" for path in invalid)
        raise RuntimeError(
            "TabPFN v2 checkpoint checksum mismatch:\n"
            f"{lines}\nRestage or retransmit the affected checkpoint."
        )
    return paths


def stage_checkpoints(cache_dir: Path | None = None) -> tuple[Path, Path]:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise RuntimeError("huggingface_hub is required to stage TabPFN") from error

    paths = checkpoint_paths(cache_dir)
    paths[0].parent.mkdir(parents=True, exist_ok=True)
    for destination in paths:
        if (
            destination.is_file()
            and _sha256(destination) == _CHECKPOINT_SHA256[destination.name]
        ):
            print(f"Reusing verified {destination}")
            continue
        print(f"Staging {destination.name} at {destination}")
        downloaded = Path(
            hf_hub_download(
                repo_id=_CHECKPOINT_REPOS[destination.name],
                filename=destination.name,
                local_dir=destination.parent,
                force_download=destination.exists(),
            )
        )
        if downloaded.resolve() != destination.resolve():
            downloaded.replace(destination)
    require_checkpoints(cache_dir)
    print("TabPFN v2 checkpoints staged successfully.")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage local TabPFN v2 checkpoints")
    parser.add_argument("--cache-dir", type=Path, default=None)
    args = parser.parse_args()
    stage_checkpoints(args.cache_dir)


if __name__ == "__main__":
    main()
