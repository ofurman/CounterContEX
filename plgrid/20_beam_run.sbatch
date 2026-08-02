#!/bin/bash -l
###############################################################################
# The real beam-search run on the full test splits.
#
# Defaults to the ONE cell that is genuinely missing: HELOC / from-scratch
# (Set 2). Law frozen (444 pts) and HELOC frozen (2092 pts) were already
# computed locally and shipped by sync-to-plgrid.sh; Law has no immutable
# features, so its two regimes are the identical code path and Law is complete.
#
#   bash plgrid/submit.sh plgrid/20_beam_run.sbatch                  # missing cell
#   CELLS="law:frozen law:fromscratch heloc:frozen heloc:fromscratch" \
#       bash plgrid/submit.sh plgrid/20_beam_run.sbatch              # full 2x2 grid
#   FORCE=1 bash plgrid/submit.sh plgrid/20_beam_run.sbatch          # ignore resume
#
# Cells run sequentially in one job rather than as an array. exp4 has no
# --results-dir, so parallel array tasks would race on exp4_summary.md and on
# each other's metrics CSVs in a shared checkout.
#
# --chunk-size is pinned. TabPFN's predictions depend on the composition of the
# predict batch, so results are NOT chunk-invariant (measured: chunk=40 vs
# chunk=7 differed by ~1.0 on a one-hot column). 4096 = one call per class, and
# it must stay identical across every cell that appears in the same table.
###############################################################################

#SBATCH --job-name=cx_beam
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/cx_beam-%j.out
#SBATCH --error=logs/cx_beam-%j.err

set -euo pipefail
mkdir -p logs

# shellcheck source=_common.sh
source "${SLURM_SUBMIT_DIR:-${PWD}}/plgrid/_common.sh"

CELLS="${CELLS:-heloc:fromscratch}"
CHUNK="${CHUNK_SIZE:-4096}"
MAX_TEST="${MAX_TEST:--1}"   # -1 = full stratified test split

# Peak RSS was 13 GB locally; --mem=96G leaves generous headroom for the larger
# chunked predict batches on a 2092-point split.
trap mirror_results EXIT

nvidia-smi || echo "WARNING: nvidia-smi not on PATH"
echo ""
echo "cells      ${CELLS}"
echo "chunk-size ${CHUNK}"
echo "max-test   ${MAX_TEST}"

for cell in ${CELLS}; do
    dataset="${cell%%:*}"
    tag="${cell##*:}"
    npz="${RESULTS}/arrays/exp4_${dataset}_${tag}_cfs.npz"

    echo ""
    echo "############################################################"
    echo "# ${dataset} / ${tag}"
    echo "############################################################"

    # exp4 has no resume logic of its own: it recomputes and overwrites every
    # requested cell. The guard lives here so a requeued job keeps finished
    # work instead of paying for it twice.
    if [ -s "${npz}" ] && [ "${FORCE:-0}" != "1" ]; then
        echo "SKIP: ${npz} already exists ($(du -h "${npz}" | cut -f1)). FORCE=1 to redo."
        continue
    fi

    SECONDS=0
    srun python -m experiments.zeroshot_cf.exp4_beam_search \
        --dataset "${dataset}" \
        --set "${tag}" \
        --max-test "${MAX_TEST}" \
        --chunk-size "${CHUNK}"
    echo "--- ${dataset}/${tag} done in ${SECONDS}s"

    test -s "${npz}" || { echo "ERROR: ${npz} not written" >&2; exit 1; }
    # Mirror after every cell, not just at the end: a job that dies in cell 3
    # then keeps cells 1 and 2 in durable storage.
    mirror_results
done

echo ""
echo "=== rescoring every saved cell into one table ==="
# exp4_summary.md is rewritten from only the cells of the invocation that
# produced it, so after a resumed or partial run it under-reports. This reads
# every .npz on disk instead and is the table to trust.
srun python -m experiments.zeroshot_cf.recompute_metrics --min-n 100

echo ""
echo "=== results ==="
ls -la "${RESULTS}" "${RESULTS}/arrays"
if [ -s "${RESULTS}/exp4_recomputed_metrics.csv" ]; then
    echo "--- exp4_recomputed_metrics.csv"
    cat "${RESULTS}/exp4_recomputed_metrics.csv"
fi

echo ""
echo "Job ${SLURM_JOB_ID} finished at $(date)"
