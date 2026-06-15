"""One-time vendor setup: clone cel (counterfactuals) and apply Python 3.13 patch.

Run from the repo root:
    python experiments/zeroshot_cf/vendor_setup.py

This creates experiments/zeroshot_cf/vendor/counterfactuals/ (gitignored) and
installs it editable with --no-deps, then installs the minimal required deps.
"""

import subprocess
import shutil
import sys
from pathlib import Path

VENDOR_DIR = Path(__file__).parent / "vendor"
CEL_REPO = VENDOR_DIR / "counterfactuals"
CEL_GIT_URL = "https://github.com/ofurman/counterfactuals.git"
PATCH_SRC = Path(__file__).parent / "patches" / "cel_init.py"
PYTHON = sys.executable


def run(cmd, **kwargs):
    print(f"$ {' '.join(str(c) for c in cmd)}")
    subprocess.run(cmd, check=True, **kwargs)


def main():
    # 1. Clone if not already there
    if not CEL_REPO.exists():
        VENDOR_DIR.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--depth", "1", CEL_GIT_URL, str(CEL_REPO)])
    else:
        print(f"Vendor already exists: {CEL_REPO}")

    # 2. Apply the Python 3.13 patch (makes heavy imports optional)
    patch_dest = CEL_REPO / "cel" / "__init__.py"
    shutil.copy2(PATCH_SRC, patch_dest)
    print(f"Patch applied: {patch_dest}")

    # 3. Install editable (no deps — avoids tensorflow conflict on Python 3.13)
    uv = shutil.which("uv") or "uv"
    run([uv, "pip", "install", "--no-deps", "-e", str(CEL_REPO)])

    # 4. Install minimal required transitive deps
    minimal_deps = ["cel-nflows", "torchdiffeq", "UMNN", "omegaconf", "hydra-core"]
    run([uv, "pip", "install", *minimal_deps])

    print("\n=== vendor_setup DONE ===")
    print("Run: python -c \"import cel; print(cel.__version__)\" to verify.")


if __name__ == "__main__":
    main()
