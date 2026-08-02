#!/bin/bash -l
###############################################################################
# Exp 7: beam-search hyperparameter sweep on the full test splits.
#
# One job runs ONE cell (dataset:set) across MANY configs, sequentially. Configs
# are declared in the CONFIGS list below as
#
#     <run-id>|<extra exp4 CLI flags>
#
# and each writes its own config-tagged array:
#
#     results/arrays/sweep/exp4_<dataset>_<set>__<run-id>_cfs.npz
#
# so nothing overwrites anything — that was the blocker that made a sweep
# impossible on the old pipeline (every run rewrote the same npz and the same
# exp4_summary.md).
#
#   CELL=heloc:frozen bash plgrid/submit.sh plgrid/30_beam_sweep.sbatch
#   CELL=law:frozen   bash plgrid/submit.sh plgrid/30_beam_sweep.sbatch
#   CELL=heloc:frozen CONFIGS="bw16|--beam-width 16" \
#       bash plgrid/submit.sh plgrid/30_beam_sweep.sbatch     # subset
#   FORCE=1 CELL=... bash plgrid/submit.sh plgrid/30_beam_sweep.sbatch
#
# Sequential within one job, not a Slurm array: exp4 has no --results-dir, so
# parallel array tasks in a shared checkout would race on each other's output.
# Different CELLs are safe to run as concurrent jobs — their npz namespaces are
# disjoint (they differ in the <dataset>_<set> part of the filename).
#
# --chunk-size is PINNED at 4096 and is NOT a sweep axis. TabPFN's predictions
# depend on the composition of the predict batch, so results are not
# chunk-invariant (measured: chunk=40 vs chunk=7 differed by ~1.0 on a one-hot
# column). Varying it would confound every other axis.
###############################################################################

#SBATCH --job-name=cx_sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/cx_sweep-%j.out
#SBATCH --error=logs/cx_sweep-%j.err

set -euo pipefail
mkdir -p logs

# shellcheck source=_common.sh
source "${SLURM_SUBMIT_DIR:-${PWD}}/plgrid/_common.sh"

CELL="${CELL:-heloc:frozen}"
DATASET="${CELL%%:*}"
TAG="${CELL##*:}"
CHUNK="${CHUNK_SIZE:-4096}"
MAX_TEST="${MAX_TEST:--1}"   # -1 = full stratified test split

# One-factor-at-a-time around the BeamConfig defaults
# (beam_width=8, n_candidates=6, lambda_actionable=1.0, max_context=256,
#  candidate_probs=interior grid, n_estimators=4).
#
# `base` is the defaults re-run ON THIS BACKEND. Every sweep number is compared
# against it, never against the local MPS numbers — TabPFN's outputs are not
# bitwise portable across backends, so a cross-hardware comparison would confound
# the hyperparameter effect with the CUDA-vs-MPS difference.
#
# lambda_actionable=0 is the direct test of whether the proximity penalty is
# suppressing validity: it removes the penalty entirely, leaving pure
# log-density + the validity rerank.
#
# probs-tail reaches the distribution tails. The default interior grid hugs the
# mode, and the generated CFs travel a median 0.3% of the available margin — if
# step size is the binding constraint, this is the axis that shows it.
DEFAULT_CONFIGS="
base|
bw4|--beam-width 4
bw16|--beam-width 16
bw32|--beam-width 32
nc4|--n-candidates 4
nc10|--n-candidates 10
nc16|--n-candidates 16
lam0|--lambda-actionable 0.0
lam0.1|--lambda-actionable 0.1
lam3|--lambda-actionable 3.0
lam10|--lambda-actionable 10.0
ctx128|--max-context 128
ctx512|--max-context 512
ctx1024|--max-context 1024
probs-tail|--candidate-probs tail
"
CONFIGS="${CONFIGS:-${DEFAULT_CONFIGS}}"

trap mirror_results EXIT

nvidia-smi || echo "WARNING: nvidia-smi not on PATH"
echo ""
echo "cell        ${CELL}  (dataset=${DATASET} set=${TAG})"
echo "chunk-size  ${CHUNK}  [pinned, not a sweep axis]"
echo "max-test    ${MAX_TEST}"
echo "commit      $(git -C "${PROJECT}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "configs:"
echo "${CONFIGS}" | sed '/^[[:space:]]*$/d' | sed 's/^/    /'

n_run=0
n_skip=0
n_total=$(echo "${CONFIGS}" | sed '/^[[:space:]]*$/d' | wc -l | tr -d ' ')

while IFS= read -r line; do
    # Skip blank lines (the heredoc-style list has leading/trailing newlines).
    [ -z "${line//[[:space:]]/}" ] && continue

    RUN_ID="${line%%|*}"
    EXTRA="${line#*|}"
    npz="${RESULTS}/arrays/sweep/exp4_${DATASET}_${TAG}__${RUN_ID}_cfs.npz"

    echo ""
    echo "############################################################"
    echo "# [$((n_run + n_skip + 1))/${n_total}] ${DATASET}/${TAG}  run-id=${RUN_ID}"
    echo "#     extra flags: ${EXTRA:-<defaults>}"
    echo "############################################################"

    # exp4 recomputes and overwrites whatever it is asked for, so the resume guard
    # lives here: a requeued job keeps finished configs instead of paying twice.
    if [ -s "${npz}" ] && [ "${FORCE:-0}" != "1" ]; then
        echo "SKIP: ${npz} exists ($(du -h "${npz}" | cut -f1)). FORCE=1 to redo."
        n_skip=$((n_skip + 1))
        continue
    fi

    SECONDS=0
    # shellcheck disable=SC2086  # EXTRA is a deliberate word-split flag list
    srun python -m experiments.zeroshot_cf.exp4_beam_search \
        --dataset "${DATASET}" \
        --set "${TAG}" \
        --max-test "${MAX_TEST}" \
        --chunk-size "${CHUNK}" \
        --run-id "${RUN_ID}" \
        ${EXTRA}
    echo "--- ${DATASET}/${TAG} ${RUN_ID} done in ${SECONDS}s"

    test -s "${npz}" || { echo "ERROR: ${npz} not written" >&2; exit 1; }
    n_run=$((n_run + 1))
    # Mirror after every config, not just at the end: a job that dies on config 9
    # keeps configs 1-8 in durable storage.
    mirror_results
done <<< "${CONFIGS}"

echo ""
echo "=== sweep complete: ${n_run} generated, ${n_skip} skipped, ${n_total} total ==="
ls -la "${RESULTS}/arrays/sweep" 2>/dev/null || echo "no sweep arrays!"

# Deliberately NOT scoring here. exp7_sweep_table.py needs only the arrays and no
# GPU, so scoring runs locally against the pulled-back npz files (standing rule:
# generation on PLGrid, scoring local). Keeping it off the GPU node also means a
# metric bug never costs a regeneration.

echo ""
echo "Job ${SLURM_JOB_ID} finished at $(date)"
