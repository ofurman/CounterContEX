#!/bin/bash -l
###############################################################################
# Build the Python environment for the beam-search experiment — inside a GH200
# job, because Helios login nodes are x86_64 and GH200 compute nodes are
# aarch64. A login-built venv installs wheels the GPU node cannot import.
#
# torch is NOT taken from ML-bundle. That module is a CUDA/compiler stack whose
# only torch is a local cp313 wheelhouse, which would force Python 3.13 on us.
# PyPI publishes CUDA-enabled aarch64 torch wheels for this platform and a
# sibling project on this exact partition (job 19930987) installed
# torch 2.11.0+cu130 that way on a GH200 node — that same job also proves the
# compute nodes have outbound network for PyPI, which is what makes this work.
#
#   bash plgrid/submit.sh plgrid/00_setup_env.sbatch
###############################################################################

#SBATCH --job-name=cx_setup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/cx_setup-%j.out
#SBATCH --error=logs/cx_setup-%j.err

set -euo pipefail
mkdir -p logs

SKIP_VENV=1
# shellcheck source=_common.sh
source "${SLURM_SUBMIT_DIR:-${PWD}}/plgrid/_common.sh"

VENV="${HEAVY}/envs/${VENV_NAME}"
BOOT="${HEAVY}/envs/bootstrap"

echo ""
echo "=== 0/5 architecture ==="
test "$(uname -m)" = "${COMPUTE_ARCH}" || {
    echo "ERROR: expected ${COMPUTE_ARCH}, got $(uname -m). Building here would" >&2
    echo "       produce a venv the GPU nodes cannot import." >&2
    exit 1
}
echo "arch $(uname -m) — ok"

echo ""
echo "=== 1/5 bootstrap uv ==="
# uv is not in the Helios module set, so it gets its own throwaway venv built
# from the module python. This step needs network; if it fails with a
# connection error the node is firewalled and nothing below will work.
if [ ! -x "${BOOT}/bin/uv" ]; then
    rm -rf "${BOOT}"
    python -m venv "${BOOT}"
    "${BOOT}/bin/pip" install --quiet --upgrade pip
    "${BOOT}/bin/pip" install --quiet uv
fi
UV="${BOOT}/bin/uv"
"${UV}" --version

echo ""
echo "=== 2/5 CPython ${PY_VERSION} ==="
# uv fetches its own interpreter, so the venv's Python is independent of the
# module set's 3.13. ${PY_VERSION} is the version proven to resolve cleanly on
# aarch64 here; see plgrid/README.md for the parity caveat vs the local run.
"${UV}" python install "${PY_VERSION}"

echo ""
echo "=== 3/5 locked dependencies -> ${VENV} ==="
# --frozen: resolve uv.lock exactly, never re-resolve. pyproject sets
# tool.uv.exclude-newer = "7 days", so an unfrozen sync would drift with the
# calendar and stop matching the environment the local cells were produced in.
UV_PROJECT_ENVIRONMENT="${VENV}" "${UV}" sync --frozen --python "${PY_VERSION}"

echo ""
echo "=== 4/5 vendored cel library ==="
# `cel` (ofurman/counterfactuals) provides load_dataset and the preprocessing
# pipeline. It is NOT in pyproject/uv.lock — vendor_setup.py normally git
# clones it, which a compute node should not rely on, so sync-to-plgrid.sh
# rsyncs the working copy instead.
#
# Installed editable --no-deps on purpose: its real dependency list pulls
# alibi[tensorflow], cvxpy, mlflow, dice-ml and friends, none of which the
# beam search touches and several of which will not build here. The five
# packages below are the minimal transitive set, matching vendor_setup.py.
#
# NOTE: `uv sync` above prunes anything not in uv.lock, so this must run after
# it — and must be re-run after any future sync.
CEL="${PROJECT}/experiments/zeroshot_cf/vendor/counterfactuals"
if [ ! -d "${CEL}/cel" ]; then
    echo "ERROR: vendored cel missing at ${CEL}" >&2
    echo "       re-run plgrid/sync-to-plgrid.sh (vendor/ is gitignored)." >&2
    exit 1
fi
VIRTUAL_ENV="${VENV}" "${UV}" pip install --no-deps -e "${CEL}"
VIRTUAL_ENV="${VENV}" "${UV}" pip install cel-nflows torchdiffeq UMNN omegaconf hydra-core

echo ""
echo "=== 5/5 verification ==="
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
python - <<'PY'
import os
import platform
import sys
from pathlib import Path

import numpy
import torch

print("machine       ", platform.machine())
print("python        ", platform.python_version())
print("numpy         ", numpy.__version__)
print("torch         ", torch.__version__)
print("cuda available", torch.cuda.is_available())
assert torch.cuda.is_available(), "no CUDA device visible in the setup job"
print("device        ", torch.cuda.get_device_name(0))

# A real synchronised calculation, not just a device query.
a = torch.randn(4096, 4096, device="cuda")
torch.cuda.synchronize()
print("matmul        ", float((a @ a.T).sum()))

# cel + its data files, which live inside the vendored repo and are resolved
# relative to cel/__file__ rather than cwd.
from cel.datasets.file_dataset import FileDataset  # noqa: F401,E402

print("cel import    ", "ok")

# TabPFN loaded from the staged v2 checkpoints with the hub bolted shut. If
# HF_HUB_OFFLINE were not doing its job this is where it would show.
assert os.environ.get("HF_HUB_OFFLINE") == "1", "HF_HUB_OFFLINE not set"
cache = Path(os.environ["TABPFN_LOCAL_CACHE"])
for f in sorted(cache.glob("*.ckpt")):
    print(f"ckpt           {f.name}  {f.stat().st_size / 1e6:.1f} MB")

sys.path.insert(0, ".")
from experiments.zeroshot_cf.checkpoints import get_models  # noqa: E402

clf, reg = get_models(n_estimators=4)
print("tabpfn v2     ", type(clf).__name__, type(reg).__name__)
PY

echo ""
echo "Environment ready: ${VENV}"
echo "Job ${SLURM_JOB_ID} finished at $(date)"
