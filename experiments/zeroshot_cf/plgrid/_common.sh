#!/bin/bash
# Shared paths and environment for CounterContEx jobs on Helios GH200.

PLG_GROUP="${PLG_GROUP:-plggcfsgenwro}"
PROJECT_NAME="${PROJECT_NAME:-countercontex-campaign}"
VENV_NAME="${VENV_NAME:-campaign}"

PROJECT="${SLURM_SUBMIT_DIR:-${PWD}}"
: "${PLG_GROUPS_STORAGE:?PLG_GROUPS_STORAGE is not set — is this a PLGrid node?}"
: "${SCRATCH:?SCRATCH is not set}"
HEAVY="${PLG_GROUPS_STORAGE}/${PLG_GROUP}/${USER}/${PROJECT_NAME}"
STORE="${SCRATCH}/${PROJECT_NAME}"
VENV="${HEAVY}/envs/${VENV_NAME}"

umask 077
mkdir -p "${HEAVY}"/{models,results} "${STORE}/cache"

export UV_CACHE_DIR="${STORE}/cache/uv"
export XDG_CACHE_HOME="${STORE}/cache/xdg"
export HF_HOME="${STORE}/cache/huggingface"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TABICL_DEVICE="${TABICL_DEVICE:-cuda}"
export WANDB_MODE=offline
mkdir -p "${UV_CACHE_DIR}" "${XDG_CACHE_HOME}" "${HF_HOME}"

cd "${PROJECT}"

link_heavy() {
    local link="$1" target="$2"
    if [ -L "${link}" ]; then
        [ "$(readlink -f "${link}")" = "$(readlink -f "${target}")" ] \
            || { echo "ERROR: ${link} points elsewhere: $(readlink "${link}")" >&2; exit 1; }
    elif [ -e "${link}" ]; then
        echo "ERROR: ${link} exists and is not a symlink; refusing to replace it" >&2
        exit 1
    else
        ln -s "${target}" "${link}"
    fi
}

link_heavy experiments/zeroshot_cf/models "${HEAVY}/models"
link_heavy experiments/zeroshot_cf/results "${HEAVY}/results"

[ -x "${VENV}/bin/python" ] \
    || { echo "ERROR: missing campaign environment at ${VENV}" >&2; exit 1; }
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

echo "host     $(hostname)  arch $(uname -m)"
echo "project  ${PROJECT}  commit $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "heavy    ${HEAVY}"
echo "venv     ${VENV}"
