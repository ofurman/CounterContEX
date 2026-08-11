#!/usr/bin/env bash
#SBATCH --job-name=tabular-fm-exp8
#SBATCH --account=plgmodalitiescfes-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=47:00:00
#SBATCH --array=0-3
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
CASE_FILE="${CASE_FILE:-$PROJECT_DIR/experiments/zeroshot_cf/athena/exp8_backend_cases.tsv}"
RESULTS_BASE="${RESULTS_BASE:-$PROJECT_DIR/experiments/zeroshot_cf/results/athena}"

cd "$PROJECT_DIR"

mapfile -t CASES < <(awk 'NF && $1 !~ /^#/ {print}' "$CASE_FILE")
if (( ${#CASES[@]} == 0 )); then
    echo "No runnable cases in $CASE_FILE" >&2
    exit 2
fi

if (( SLURM_ARRAY_TASK_ID >= ${#CASES[@]} )); then
    echo "Array index $SLURM_ARRAY_TASK_ID outside case count ${#CASES[@]}" >&2
    exit 2
fi

IFS=$'\t' read -r \
    DATASET BACKEND MAX_TEST N_ESTIMATORS TEMPERATURE TAG EXTRA \
    <<< "${CASES[$SLURM_ARRAY_TASK_ID]}"

if [[ -z "${TAG:-}" || -n "${EXTRA:-}" ]]; then
    echo "Expected 6 tab-separated case columns:" >&2
    echo "dataset backend max_test n_estimators temperature tag" >&2
    echo "Got: ${CASES[$SLURM_ARRAY_TASK_ID]}" >&2
    exit 2
fi

RESULTS_DIR="$RESULTS_BASE/$TAG"
mkdir -p "$RESULTS_DIR"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TABPFN_MODEL_VERSION="v3"
export TABPFN_DEVICE="${TABPFN_DEVICE:-cuda}"
export TABPFN_LOCAL_CACHE="${TABPFN_LOCAL_CACHE:-$PROJECT_DIR/models}"
export TABICL_DEVICE="${TABICL_DEVICE:-cuda}"
export TABICL_LOCAL_CACHE="${TABICL_LOCAL_CACHE:-$PROJECT_DIR/experiments/zeroshot_cf/models/tabicl}"
# Each array task gets its own tiny discriminator cache, avoiding concurrent
# writes when the two backends for one dataset start at the same time.
export ZEROSHOT_CF_MODELS_DIR="$RESULTS_DIR/models"

if [[ -n "${UV_CACHE_DIR:-}" ]]; then
    :
elif [[ -n "${SCRATCH:-}" ]]; then
    export UV_CACHE_DIR="$SCRATCH/uv-cache"
fi

if [[ -n "${PYTHON_BIN:-}" ]]; then
    PYTHON_CMD=("$PYTHON_BIN")
elif [[ -x "$PROJECT_DIR/.venv/bin/python" ]]; then
    PYTHON_CMD=("$PROJECT_DIR/.venv/bin/python")
elif command -v uv >/dev/null 2>&1; then
    PYTHON_CMD=(uv run python)
else
    PYTHON_CMD=(python)
fi

echo "[$(date --iso-8601=seconds)] Starting Exp8 backend comparison"
echo "project=$PROJECT_DIR"
echo "dataset=$DATASET backend=$BACKEND max_test=$MAX_TEST"
echo "n_estimators=$N_ESTIMATORS temperature=$TEMPERATURE tag=$TAG"
echo "results_dir=$RESULTS_DIR"
echo "python=${PYTHON_CMD[*]}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
else
    echo "nvidia-smi not found on PATH"
fi

"${PYTHON_CMD[@]}" -m experiments.zeroshot_cf.exp8_backend_comparison \
    --dataset "$DATASET" \
    --backend "$BACKEND" \
    --max-test "$MAX_TEST" \
    --n-estimators "$N_ESTIMATORS" \
    --temperature "$TEMPERATURE" \
    --tabpfn-cache-dir "$TABPFN_LOCAL_CACHE" \
    --tabicl-cache-dir "$TABICL_LOCAL_CACHE" \
    --results-dir "$RESULTS_DIR"

echo "[$(date --iso-8601=seconds)] Finished Exp8 comparison: $TAG"
