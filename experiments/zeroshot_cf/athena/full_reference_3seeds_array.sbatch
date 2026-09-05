#!/usr/bin/env bash
#SBATCH --job-name=countercontex-full3seed
#SBATCH --account=plgfoundationeconom-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --array=0-71
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --error=slurm-%x-%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
SUITE_DIR="${SUITE_DIR:-$PROJECT_DIR/experiments/zeroshot_cf}"
BASE_CONFIG="${BASE_CONFIG:-$SUITE_DIR/configs/matrices/full_reference_3seeds.yaml}"
RESULTS_DIR="${RESULTS_DIR:-$SUITE_DIR/results/athena/full_reference_3seeds}"
TABICL_CACHE_DIR="${TABICL_CACHE_DIR:-$SUITE_DIR/models/tabicl}"
CELL_CONFIG_DIR="${CELL_CONFIG_DIR:-$RESULTS_DIR/generated_configs}"
COUNTERCONTEX_SLURM_WALLTIME="${COUNTERCONTEX_SLURM_WALLTIME:-10:00:00}"
export COUNTERCONTEX_SLURM_WALLTIME

cd "$PROJECT_DIR"

mkdir -p "$RESULTS_DIR" "$RESULTS_DIR/models" "$CELL_CONFIG_DIR" "$RESULTS_DIR/wandb"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TABICL_DEVICE="${TABICL_DEVICE:-cuda}"
# Shared (not per-task) so every dataset's classifier checkpoint is trained or
# loaded from the exact same cache file across all methods and seeds.
export ZEROSHOT_CF_MODELS_DIR="$RESULTS_DIR/models"
# Compute nodes have no outbound network. GenericRunner still logs every
# computed metric via wandb.log() the moment each cell finishes -- offline
# mode just writes those runs to disk instead of uploading them. Sync with
# `wandb sync "$RESULTS_DIR/wandb"` from a networked host once tasks finish.
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="$RESULTS_DIR/wandb"

if [[ -n "${UV_CACHE_DIR:-}" ]]; then
    :
elif [[ -n "${SCRATCH:-}" ]]; then
    export UV_CACHE_DIR="$SCRATCH/uv-cache"
fi

if [[ -n "${PYTHON_BIN:-}" ]]; then
    PYTHON_CMD=("$PYTHON_BIN")
elif [[ -x "$SUITE_DIR/.venv/bin/python" ]]; then
    PYTHON_CMD=("$SUITE_DIR/.venv/bin/python")
elif command -v uv >/dev/null 2>&1; then
    PYTHON_CMD=(uv run --project "$SUITE_DIR" python)
else
    PYTHON_CMD=(python)
fi

CELL_CONFIG="$CELL_CONFIG_DIR/cell_${SLURM_ARRAY_TASK_ID}.yaml"
"${PYTHON_CMD[@]}" -m experiments.zeroshot_cf.select_matrix_cell \
    --config "$BASE_CONFIG" \
    --index "$SLURM_ARRAY_TASK_ID" \
    --tabicl-cache-dir "$TABICL_CACHE_DIR" \
    --device "$TABICL_DEVICE" \
    --output "$CELL_CONFIG"

echo "[$(date --iso-8601=seconds)] Starting full-reference 3-seed benchmark cell $SLURM_ARRAY_TASK_ID"
echo "project=$PROJECT_DIR"
echo "suite=$SUITE_DIR"
echo "base_config=$BASE_CONFIG"
echo "cell_config=$CELL_CONFIG"
echo "results_dir=$RESULTS_DIR models_dir=$ZEROSHOT_CF_MODELS_DIR"
echo "requested_walltime=$COUNTERCONTEX_SLURM_WALLTIME"
echo "python=${PYTHON_CMD[*]}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi

"${PYTHON_CMD[@]}" -m experiments.zeroshot_cf.cli matrix \
    --config "$CELL_CONFIG" \
    --resume

echo "[$(date --iso-8601=seconds)] Finished cell $SLURM_ARRAY_TASK_ID"
