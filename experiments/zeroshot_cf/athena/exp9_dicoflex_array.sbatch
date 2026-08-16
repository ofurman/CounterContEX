#!/usr/bin/env bash
#SBATCH --job-name=tabicl-exp9
#SBATCH --account=plgfoundationeconom-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-4
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
CASE_FILE="${CASE_FILE:-$PROJECT_DIR/experiments/zeroshot_cf/athena/exp9_dicoflex_cases.tsv}"
RESULTS_DIR="${RESULTS_DIR:-$PROJECT_DIR/experiments/zeroshot_cf/results/athena/exp9_dicoflex}"
TABICL_CACHE_DIR="${TABICL_CACHE_DIR:-$PROJECT_DIR/experiments/zeroshot_cf/models/tabicl}"
MAX_VALIDITY_STEPS="${MAX_VALIDITY_STEPS:-100}"

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

IFS=$'\t' read -r DATASET MAX_TEST N_ESTIMATORS TAG EXTRA \
    <<< "${CASES[$SLURM_ARRAY_TASK_ID]}"
if [[ -z "${TAG:-}" || -n "${EXTRA:-}" ]]; then
    echo "Expected 4 tab-separated columns:" >&2
    echo "dataset max_test n_estimators tag" >&2
    echo "Got: ${CASES[$SLURM_ARRAY_TASK_ID]}" >&2
    exit 2
fi

mkdir -p "$RESULTS_DIR" "$RESULTS_DIR/models/$TAG"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TABICL_DEVICE="${TABICL_DEVICE:-cuda}"
export TABICL_LOCAL_CACHE="$TABICL_CACHE_DIR"
export ZEROSHOT_CF_MODELS_DIR="$RESULTS_DIR/models/$TAG"

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

echo "[$(date --iso-8601=seconds)] Starting Exp9 TabICL benchmark"
echo "project=$PROJECT_DIR"
echo "dataset=$DATASET max_test=$MAX_TEST n_estimators=$N_ESTIMATORS max_validity_steps=$MAX_VALIDITY_STEPS"
echo "split=64/16/20 seed=42 test_selection=stratified"
echo "tag=$TAG results_dir=$RESULTS_DIR"
echo "python=${PYTHON_CMD[*]}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi

"${PYTHON_CMD[@]}" -m experiments.zeroshot_cf.exp9_dicoflex_benchmark \
    --dataset "$DATASET" \
    --max-test "$MAX_TEST" \
    --n-estimators "$N_ESTIMATORS" \
    --max-validity-steps "$MAX_VALIDITY_STEPS" \
    --tabicl-cache-dir "$TABICL_CACHE_DIR" \
    --results-dir "$RESULTS_DIR"

echo "[$(date --iso-8601=seconds)] Finished Exp9 dataset: $DATASET"
