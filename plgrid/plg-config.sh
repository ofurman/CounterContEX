#!/bin/bash
# PLGrid settings for the CounterContEX zero-shot counterfactual beam search.
# Sourced by sync-to-plgrid.sh, submit.sh and every job script.
#
# Anything already set in the environment wins, so a one-off run can override
# without editing this file.
#
# Verified live on Helios 2026-07-28:
#   grant     plgcountercontex, active to 2027-03-23
#   alloc     plgcountercontex-gpu-gh200, 10 000 h granted, 932.94 h consumed
#   storage   plggcfsgenwro, 250 GB limit, 32.5 GiB used, SHARED by four people
#   partition plgrid-gpu-gh200, 110 nodes, gpu:4/node, max walltime 2-00:00:00
#   $SCRATCH  /net/scratch/hscra/plgrid/plgllenkiewicz, 12 TiB, personal
#
# The grant holds ONLY a gpu-gh200 allocation: `sbatch --test-only` rejects the
# plgrid / plgrid-long / cpu partitions with "Invalid account or
# account/partition combination". CPU-only work therefore also runs here, just
# submitted without --gres.

PLG_LOGIN="${PLG_LOGIN:-plgllenkiewicz}"
PLG_HOST="${PLG_HOST:-login01.helios.cyfronet.pl}"
PLG_ACCOUNT="${PLG_ACCOUNT:-plgcountercontex-gpu-gh200}"
PLG_PARTITION="${PLG_PARTITION:-plgrid-gpu-gh200}"
PLG_GROUP="${PLG_GROUP:-plggcfsgenwro}"

PROJECT_NAME="${PROJECT_NAME:-countercontex}"

# Helios login nodes are x86_64, GH200 compute nodes are aarch64. The whole
# environment is therefore built inside a GPU job (00_setup_env.sbatch); a
# login-built venv installs wheels the compute node cannot import.
COMPUTE_ARCH="${COMPUTE_ARCH:-aarch64}"
ML_BUNDLE="${ML_BUNDLE:-ML-bundle/25.10}"

# Python for the project venv.
#
# NOT inherited from ML-bundle. That bundle is a CUDA/compiler stack whose only
# torch is a local cp313 wheelhouse
# (/net/software/aarch64/el9/wheels/ML-bundle/25.10), which would pin us to
# Python 3.13. We do not need it: PyPI publishes CUDA-enabled aarch64 torch
# wheels for this platform, and a sibling project on this exact partition
# (job 19930987) installed torch 2.11.0+cu130 that way on a GH200 node. uv
# fetches its own interpreter, so this version is free to differ from the
# module set's 3.13.
PY_VERSION="${PY_VERSION:-3.11}"
VENV_NAME="${VENV_NAME:-beam}"

# TabPFN checkpoints. Fetched from the HuggingFace Hub during 00_setup_env
# (compute nodes DO have outbound network — verified, see README) and pinned
# into group storage so every later job can run with HF_HUB_OFFLINE=1.
TABPFN_MODEL_SUBDIR="${TABPFN_MODEL_SUBDIR:-models/tabpfn}"

plg_require() {
    local missing=0
    for var in "$@"; do
        if [ -z "${!var:-}" ]; then
            echo "ERROR: ${var} is not set — edit plgrid/plg-config.sh" >&2
            missing=1
        fi
    done
    [ "${missing}" -eq 0 ] || exit 1
}
