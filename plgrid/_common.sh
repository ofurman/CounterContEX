#!/bin/bash
# Shared job preamble: paths, caches, modules, venv, TabPFN offline settings.
# Sourced by every .sbatch in this directory after `set -euo pipefail`.
#
# Defines:
#   PROJECT  code checkout in $HOME (small, backed up, 10 GiB quota)
#   HEAVY    group storage root — the venv only (250 GB SHARED by four people)
#   STORE    personal $SCRATCH — uv cache and any bulk churn (12 TiB, ours)
#   RESULTS  where exp4 actually writes (see the note below)
#   MIRROR   durable copy of RESULTS in group storage

PROJECT="${SLURM_SUBMIT_DIR:-${PWD}}"
# shellcheck source=plg-config.sh
source "${PROJECT}/plgrid/plg-config.sh"
plg_require PLG_GROUP PROJECT_NAME

: "${PLG_GROUPS_STORAGE:?PLG_GROUPS_STORAGE is not set — is this a PLGrid node?}"
# Site convention in plggcfsgenwro is <group>/<login>/, NOT <group>/users/<login>/.
HEAVY="${PLG_GROUPS_STORAGE}/${PLG_GROUP}/${USER}/${PROJECT_NAME}"
STORE="${SCRATCH:?SCRATCH is not set}/${PROJECT_NAME}"

if [ ! -d "${HEAVY}" ]; then
    echo "ERROR: ${HEAVY} does not exist — run plgrid/sync-to-plgrid.sh first." >&2
    exit 1
fi
mkdir -p "${STORE}"

# Caches. The uv cache alone reached 7.5 GB for a sibling project, so it lives
# in personal $SCRATCH rather than eating the shared 250 GB grant quota. Only
# the venv itself, which has to survive $SCRATCH's 30-day retention, is in
# group storage.
export UV_CACHE_DIR="${STORE}/cache/uv"
export UV_PYTHON_INSTALL_DIR="${STORE}/cache/uv-python"
export PIP_CACHE_DIR="${STORE}/cache/pip"
export XDG_CACHE_HOME="${STORE}/cache/xdg"
export TORCH_HOME="${STORE}/cache/torch"
export HF_HOME="${STORE}/cache/huggingface"
mkdir -p "${UV_CACHE_DIR}" "${UV_PYTHON_INSTALL_DIR}" "${XDG_CACHE_HOME}" \
         "${TORCH_HOME}" "${HF_HOME}"

module purge
if ! module load "${ML_BUNDLE}" 2>/dev/null; then
    echo "ERROR: cannot load ${ML_BUNDLE}. Try: module spider ML-bundle" >&2
    exit 1
fi

# ML-bundle pins OMP_NUM_THREADS=1 for its own reasons. The beam search does
# real CPU work outside TabPFN (LOF, the sklearn discriminator, the metric
# harness), and the local reference numbers were produced with many threads, so
# set it explicitly from the allocation. Pinned rather than left to per-node
# heuristics so timings stay comparable between cells.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"
export NUMEXPR_NUM_THREADS="${OMP_NUM_THREADS}"

VENV="${HEAVY}/envs/${VENV_NAME}"
if [ "${SKIP_VENV:-0}" != "1" ]; then
    if [ ! -x "${VENV}/bin/python" ]; then
        echo "ERROR: no environment at ${VENV}" >&2
        echo "       submit plgrid/00_setup_env.sbatch first." >&2
        exit 1
    fi
    # shellcheck disable=SC1091
    source "${VENV}/bin/activate"
fi

cd "${PROJECT}"
export PYTHONPATH="${PROJECT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

###############################################################################
# TabPFN: offline, v2, GPU.
#
# The experiment pins TabPFN **v2** on purpose. v2.5/v2.6/v3 live in *gated*
# HuggingFace repos (Prior-Labs/tabpfn_2_5 etc.) whose download path calls
# tabpfn.browser_auth.ensure_license_accepted() — an interactive browser
# license flow that cannot work on a compute node. The v2 repos
# (Prior-Labs/TabPFN-v2-clf, Prior-Labs/TabPFN-v2-reg) are ungated.
#
# We do not download anything at all: both .ckpt files (~70 MiB total) are
# rsynced from the workstation into experiments/zeroshot_cf/models/ by
# sync-to-plgrid.sh, and TABPFN_LOCAL_CACHE points there.
# experiments/zeroshot_cf/checkpoints.py reads TABPFN_LOCAL_CACHE, exports
# TABPFN_MODEL_CACHE_DIR from it, and passes an explicit absolute model_path=
# to both estimators — so no hub lookup happens on the happy path.
#
# HF_HUB_OFFLINE=1 is the safety net: tabpfn/base.py hard-codes
# download_if_not_exists=True, so a missing checkpoint would otherwise silently
# try to reach huggingface.co. Offline turns that into an immediate loud error
# instead of a hang or an unnoticed re-download.
###############################################################################
export TABPFN_LOCAL_CACHE="${PROJECT}/experiments/zeroshot_cf/models"
export TABPFN_MODEL_VERSION=v2
export TABPFN_DEVICE="${TABPFN_DEVICE:-cuda}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline

if [ "${SKIP_CKPT_CHECK:-0}" != "1" ]; then
    for ckpt in tabpfn-v2-classifier-finetuned-zk73skhh.ckpt tabpfn-v2-regressor.ckpt; do
        if [ ! -s "${TABPFN_LOCAL_CACHE}/${ckpt}" ]; then
            echo "ERROR: missing TabPFN checkpoint ${TABPFN_LOCAL_CACHE}/${ckpt}" >&2
            echo "       re-run plgrid/sync-to-plgrid.sh (they are gitignored," >&2
            echo "       so a git checkout alone never provides them)." >&2
            exit 1
        fi
    done
fi

# exp4_beam_search.py has no --results-dir and no results env var: RESULTS_DIR
# is computed from __file__, so output always lands next to the script. Total
# volume is ~400 KB, which is harmless in $HOME. MIRROR is the durable copy
# pushed to group storage at the end of each job.
RESULTS="${PROJECT}/experiments/zeroshot_cf/results"
MIRROR="${HEAVY}/results"
mkdir -p "${RESULTS}/arrays" "${MIRROR}"

# Pull previously completed cells back in, so a job that follows a crashed one
# (or a fresh checkout) sees the finished Law / HELOC-frozen arrays and
# recompute_metrics.py can rescore the whole grid.
rsync -a "${MIRROR}/" "${RESULTS}/" 2>/dev/null || true

mirror_results() {
    echo "--- mirroring results -> ${MIRROR}"
    rsync -a "${RESULTS}/" "${MIRROR}/" || true
}

echo "host      $(hostname)  arch $(uname -m)"
echo "project   ${PROJECT}"
echo "heavy     ${HEAVY}"
echo "store     ${STORE}"
echo "venv      ${VENV:-<skipped>}"
echo "results   ${RESULTS}"
echo "threads   OMP_NUM_THREADS=${OMP_NUM_THREADS}"
