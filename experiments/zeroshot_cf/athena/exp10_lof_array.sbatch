#!/usr/bin/env bash
#SBATCH --job-name=tabicl-lof-ablation
#SBATCH --account=plgfoundationeconom-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-1
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
CASE_FILE="${CASE_FILE:-$PROJECT_DIR/experiments/zeroshot_cf/athena/exp10_lof_cases.tsv}"
RESULTS_DIR="${RESULTS_DIR:-$PROJECT_DIR/experiments/zeroshot_cf/results/athena/exp10_lof_ablation_r3}"
TABICL_CACHE_DIR="${TABICL_CACHE_DIR:-$PROJECT_DIR/experiments/zeroshot_cf/models/tabicl}"

cd "$PROJECT_DIR"
mapfile -t CASES < <(awk 'NF && $1 !~ /^#/ {print}' "$CASE_FILE")
if (( SLURM_ARRAY_TASK_ID >= ${#CASES[@]} )); then
    echo "Array index outside case count ${#CASES[@]}" >&2
    exit 2
fi
IFS=$'\t' read -r VARIANT MAX_TEST N_ESTIMATORS TAG EXTRA \
    <<< "${CASES[$SLURM_ARRAY_TASK_ID]}"
if [[ -z "${TAG:-}" || -n "${EXTRA:-}" ]]; then
    echo "Expected: variant max_test n_estimators tag" >&2
    exit 2
fi

mkdir -p "$RESULTS_DIR" "$RESULTS_DIR/models/$TAG"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TABICL_DEVICE="${TABICL_DEVICE:-cuda}"
export TABICL_LOCAL_CACHE="$TABICL_CACHE_DIR"
export ZEROSHOT_CF_MODELS_DIR="$RESULTS_DIR/models/$TAG"
PYTHON_CMD=("${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}")

echo "[$(date --iso-8601=seconds)] variant=$VARIANT max_test=$MAX_TEST"
nvidia-smi
"${PYTHON_CMD[@]}" -m experiments.zeroshot_cf.exp10_lof_ablation \
    --variant "$VARIANT" \
    --max-test "$MAX_TEST" \
    --n-estimators "$N_ESTIMATORS" \
    --max-validity-steps 100 \
    --tabicl-cache-dir "$TABICL_CACHE_DIR" \
    --results-dir "$RESULTS_DIR"
echo "[$(date --iso-8601=seconds)] Finished variant=$VARIANT"
