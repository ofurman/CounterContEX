#!/bin/bash
#SBATCH --job-name=countercontex-warm-cache
#SBATCH --qos=batch
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=rtx4090_batch
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err

# Standalone job: trains and disk-caches the classifier checkpoint for every
# dataset in the matrix config, once, on a real compute node -- so it never
# hits the login node's restricted CPU (see the NumPy x86-64-v2 crash this
# was split out to avoid), and so the 72-task benchmark array below never
# races multiple tasks into training/overwriting the same checkpoint file.
#
# Run this first and wait for it to finish, then submit the benchmark array
# with the warm step skipped:
#   sbatch experiments/zeroshot_cf/rtx4090/warm_classifier_cache.sbatch
#   squeue -u "$USER"   # wait until it's done
#   SKIP_WARM=1 bash experiments/zeroshot_cf/rtx4090/submit_full_reference_3seeds.sh

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/shared/results/marszale/CounterContEX}"
SUITE_DIR="${SUITE_DIR:-$PROJECT_DIR/experiments/zeroshot_cf}"
BASE_CONFIG="${BASE_CONFIG:-$SUITE_DIR/configs/matrices/full_reference_3seeds_rtx4090.yaml}"
RESULTS_DIR="${RESULTS_DIR:-$SUITE_DIR/results/rtx4090/full_reference_3seeds}"
CONDA_ENV="${CONDA_ENV:-/shared/results/marszale/anaconda3/envs/ccex}"

cd "$PROJECT_DIR"
mkdir -p logs "$RESULTS_DIR/models"

source /home/marszale/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
# Same models dir the benchmark array will read from -- must match
# ZEROSHOT_CF_MODELS_DIR in full_reference_3seeds_array.sbatch exactly.
export ZEROSHOT_CF_MODELS_DIR="$RESULTS_DIR/models"

echo "[$(date --iso-8601=seconds)] Warming classifier checkpoint cache"
echo "project=$PROJECT_DIR"
echo "base_config=$BASE_CONFIG"
echo "models_dir=$ZEROSHOT_CF_MODELS_DIR"

uv run --project "$SUITE_DIR" python -m experiments.zeroshot_cf.warm_classifier_cache \
    --config "$BASE_CONFIG"

echo "[$(date --iso-8601=seconds)] Classifier checkpoint cache warm for every dataset"
