"""Pin, patch, and verify the vendored CEL checkout used by the TabICL suite."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

VENDOR_DIR = Path(__file__).parent / "vendor"
CEL_REPO = VENDOR_DIR / "counterfactuals"
CEL_GIT_URL = "https://github.com/ofurman/counterfactuals.git"
PINNED_CEL_REVISION = "3587f943826f6b087a0d198c8c4aa4373712c7ee"
REQUIRED_BENCHMARK_DATASETS = (
    "heloc",
    "bank_marketing",
    "give_me_some_credit",
    "lending_club",
)
PATCH_SRC = Path(__file__).parent / "patches" / "cel_init.py"
_ALLOWED_LOCAL_CHANGES = {"cel/__init__.py"}
_MINIMAL_DEPS = ("cel-nflows", "torchdiffeq", "UMNN", "omegaconf", "hydra-core")


def run(cmd: list[str], **kwargs) -> None:
    print(f"$ {' '.join(str(c) for c in cmd)}")
    subprocess.run(cmd, check=True, **kwargs)


def _git(*args: str, capture_output: bool = False) -> str | None:
    command = ["git", "-C", str(CEL_REPO), *args]
    if capture_output:
        return subprocess.check_output(command, text=True).strip()
    run(command)
    return None


def _ensure_repo() -> None:
    if (CEL_REPO / ".git").exists():
        print(f"Vendor already exists: {CEL_REPO}")
        return
    VENDOR_DIR.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", CEL_GIT_URL, str(CEL_REPO)])


def _restore_supported_local_patch() -> None:
    changed = _git("status", "--short", capture_output=True)
    if not changed:
        return
    changed_paths = {
        line.split(maxsplit=1)[1]
        for line in changed.splitlines()
        if len(line.split(maxsplit=1)) == 2
    }
    unexpected = changed_paths - _ALLOWED_LOCAL_CHANGES
    if unexpected:
        names = ", ".join(sorted(unexpected))
        raise RuntimeError(
            "CEL vendor checkout has unexpected local changes and will not be "
            f"rewritten automatically: {names}"
        )
    if "cel/__init__.py" in changed_paths:
        _git("checkout", "--", "cel/__init__.py")


def _ensure_revision(revision: str) -> None:
    current = _git("rev-parse", "HEAD", capture_output=True)
    if current == revision:
        print(f"CEL revision already pinned: {revision}")
        return
    _restore_supported_local_patch()
    run(["git", "-C", str(CEL_REPO), "fetch", "origin", revision, "--depth", "1"])
    run(["git", "-C", str(CEL_REPO), "checkout", "--detach", revision])


def _apply_patch() -> None:
    patch_dest = CEL_REPO / "cel" / "__init__.py"
    shutil.copy2(PATCH_SRC, patch_dest)
    print(f"Patch applied: {patch_dest}")


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def required_dataset_pairs() -> list[tuple[str, Path, Path]]:
    pairs: list[tuple[str, Path, Path]] = []
    config_dir = CEL_REPO / "config" / "datasets"
    for dataset_name in REQUIRED_BENCHMARK_DATASETS:
        config_path = config_dir / f"{dataset_name}.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Missing dataset config: {config_path}")
        config = yaml.safe_load(config_path.read_text())
        raw_data_path = CEL_REPO / config["raw_data_path"]
        pairs.append((dataset_name, config_path, raw_data_path))
    return pairs


def verify_vendor_checkout(revision: str) -> None:
    if not (CEL_REPO / ".git").exists():
        raise FileNotFoundError(
            f"CEL vendor checkout not found at {CEL_REPO}. Run vendor_setup first."
        )
    current = _git("rev-parse", "HEAD", capture_output=True)
    if current != revision:
        raise RuntimeError(
            f"CEL revision mismatch: expected {revision}, found {current}"
        )

    print(f"CEL revision: {current}")
    for dataset_name, config_path, raw_data_path in required_dataset_pairs():
        if not raw_data_path.exists():
            raise FileNotFoundError(f"Missing dataset CSV: {raw_data_path}")
        print(
            f"{dataset_name}: config={_display_path(config_path)} "
            f"data={_display_path(raw_data_path)}"
        )
    print("Vendor checkout OK.")


def install_vendor_checkout() -> None:
    uv = shutil.which("uv") or "uv"
    suite_dir = Path(__file__).parent
    run(
        [uv, "pip", "install", "--python", sys.executable, "--no-deps", "-e", str(CEL_REPO)],
        cwd=suite_dir,
    )
    run(
        [uv, "pip", "install", "--python", sys.executable, *_MINIMAL_DEPS],
        cwd=suite_dir,
    )


def ensure_vendor_checkout(revision: str, *, repin: bool) -> None:
    repo_exists = (CEL_REPO / ".git").exists()
    if not repo_exists:
        _ensure_repo()
        _ensure_revision(revision)
        return

    current = _git("rev-parse", "HEAD", capture_output=True)
    if current == revision:
        print(f"CEL revision already pinned: {revision}")
        return
    if not repin:
        raise RuntimeError(
            "CEL vendor checkout is present but not pinned to the required revision.\n"
            f"expected: {revision}\n"
            f"found:    {current}\n"
            "Review the local checkout, then rerun with `--repin` to rewrite it "
            "or remove the vendor directory and run setup again."
        )
    _ensure_revision(revision)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pin, patch, and verify the vendored CEL checkout."
    )
    parser.add_argument(
        "--revision",
        default=PINNED_CEL_REVISION,
        help="Exact CEL Git revision to pin and verify.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only verify the current vendor checkout and required benchmark files.",
    )
    parser.add_argument(
        "--repin",
        action="store_true",
        help="Explicitly rewrite an existing vendor checkout to the requested revision.",
    )
    args = parser.parse_args()

    if args.check:
        verify_vendor_checkout(args.revision)
        return

    ensure_vendor_checkout(args.revision, repin=args.repin)
    _apply_patch()
    verify_vendor_checkout(args.revision)
    install_vendor_checkout()
    print("\n=== vendor_setup DONE ===")
    print("Run: python -c \"import cel; print(cel.__version__)\" to verify.")


if __name__ == "__main__":
    main()
