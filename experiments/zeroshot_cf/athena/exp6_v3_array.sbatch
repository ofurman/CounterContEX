#!/usr/bin/env bash
#SBATCH --job-name=tabpfn-v3-exp6
#SBATCH --account=plgmodalitiescfes-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=47:00:00
#SBATCH --array=0-0
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
CASE_FILE="${CASE_FILE:-$PROJECT_DIR/experiments/zeroshot_cf/athena/exp6_v3_cases.tsv}"
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
    DATASET SELECTOR SIZES STRATEGIES MAX_TEST N_PERMUTATIONS TEMPERATURE TAG EXTRA \
    <<< "${CASES[$SLURM_ARRAY_TASK_ID]}"

if [[ -z "${TAG:-}" || -n "${EXTRA:-}" ]]; then
    echo "Expected 8 tab-separated case columns:" >&2
    echo "dataset selector sizes strategies max_test n_permutations temperature tag" >&2
    echo "Got: ${CASES[$SLURM_ARRAY_TASK_ID]}" >&2
    exit 2
fi

RESULTS_DIR="$RESULTS_BASE/$TAG"
mkdir -p "$RESULTS_DIR"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TABPFN_MODEL_VERSION="v3"
export TABPFN_DEVICE="${TABPFN_DEVICE:-cuda}"
export TABPFN_LOCAL_CACHE="${TABPFN_LOCAL_CACHE:-$PROJECT_DIR/models}"
export ZEROSHOT_CF_RESULTS_DIR="$RESULTS_DIR"

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

echo "[$(date --iso-8601=seconds)] Starting Exp6 v3 case"
echo "project=$PROJECT_DIR"
echo "case_file=$CASE_FILE"
echo "dataset=$DATASET selector=$SELECTOR sizes=$SIZES"
echo "strategies=$STRATEGIES max_test=$MAX_TEST"
echo "n_permutations=$N_PERMUTATIONS temperature=$TEMPERATURE tag=$TAG"
echo "results_dir=$RESULTS_DIR"
echo "python=${PYTHON_CMD[*]}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
else
    echo "nvidia-smi not found on PATH"
fi

"${PYTHON_CMD[@]}" -m experiments.zeroshot_cf.exp6_context_ablation \
    --dataset "$DATASET" \
    --selector "$SELECTOR" \
    --sizes "$SIZES" \
    --strategies "$STRATEGIES" \
    --max-test "$MAX_TEST" \
    --n-permutations "$N_PERMUTATIONS" \
    --temperature "$TEMPERATURE" \
    --results-dir "$RESULTS_DIR"

echo "[$(date --iso-8601=seconds)] Finished Exp6 v3 case: $TAG"
