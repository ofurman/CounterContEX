#!/bin/bash
#SBATCH --job-name=countercontex-full3seed
#SBATCH --qos=batch
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --cpus-per-task=16
#SBATCH --partition=rtx4090_batch
#SBATCH --time=1-00:00:00
#SBATCH --array=0-71
#SBATCH --output=logs/slurm-%x-%A_%a.out
#SBATCH --error=logs/slurm-%x-%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/shared/results/marszale/CounterContEX}"
SUITE_DIR="${SUITE_DIR:-$PROJECT_DIR/experiments/zeroshot_cf}"
BASE_CONFIG="${BASE_CONFIG:-$SUITE_DIR/configs/matrices/full_reference_3seeds_rtx4090.yaml}"
RESULTS_DIR="${RESULTS_DIR:-$SUITE_DIR/results/rtx4090/full_reference_3seeds}"
TABICL_CACHE_DIR="${TABICL_CACHE_DIR:-$SUITE_DIR/models/tabicl}"
CELL_CONFIG_DIR="${CELL_CONFIG_DIR:-$RESULTS_DIR/generated_configs}"
CONDA_ENV="${CONDA_ENV:-/shared/results/marszale/anaconda3/envs/ccex}"

cd "$PROJECT_DIR"
mkdir -p logs "$RESULTS_DIR" "$RESULTS_DIR/models" "$CELL_CONFIG_DIR" "$RESULTS_DIR/wandb"

source /home/marszale/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TABICL_DEVICE="${TABICL_DEVICE:-cuda}"
# Shared (not per-task) so every dataset's classifier checkpoint is trained or
# loaded from the exact same cache file across all methods and seeds.
export ZEROSHOT_CF_MODELS_DIR="$RESULTS_DIR/models"
# Treat this cluster's GPU nodes as offline too: GenericRunner still logs
# every computed metric via wandb.log() the moment each cell finishes --
# offline mode just writes those runs to disk instead of uploading them.
# Sync with `wandb sync "$RESULTS_DIR/wandb"` from a networked host.
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="$RESULTS_DIR/wandb"

CELL_CONFIG="$CELL_CONFIG_DIR/cell_${SLURM_ARRAY_TASK_ID}.yaml"
uv run --project "$SUITE_DIR" python -m experiments.zeroshot_cf.select_matrix_cell \
    --config "$BASE_CONFIG" \
    --index "$SLURM_ARRAY_TASK_ID" \
    --tabicl-cache-dir "$TABICL_CACHE_DIR" \
    --device "$TABICL_DEVICE" \
    --output "$CELL_CONFIG"

echo "[$(date --iso-8601=seconds)] Starting full-reference 3-seed benchmark cell $SLURM_ARRAY_TASK_ID"
echo "project=$PROJECT_DIR"
echo "suite=$SUITE_DIR"
echo "conda_env=$CONDA_ENV"
echo "base_config=$BASE_CONFIG"
echo "cell_config=$CELL_CONFIG"
echo "results_dir=$RESULTS_DIR models_dir=$ZEROSHOT_CF_MODELS_DIR"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi

uv run --project "$SUITE_DIR" python -m experiments.zeroshot_cf.cli matrix \
    --config "$CELL_CONFIG" \
    --resume

echo "[$(date --iso-8601=seconds)] Finished cell $SLURM_ARRAY_TASK_ID"
