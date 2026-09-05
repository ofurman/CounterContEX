#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/shared/results/marszale/CounterContEX}"
SUITE_DIR="${SUITE_DIR:-$PROJECT_DIR/experiments/zeroshot_cf}"
SBATCH_FILE="${SBATCH_FILE:-$SUITE_DIR/rtx4090/full_reference_3seeds_array.sbatch}"
BASE_CONFIG="${BASE_CONFIG:-$SUITE_DIR/configs/matrices/full_reference_3seeds_rtx4090.yaml}"
RESULTS_DIR="${RESULTS_DIR:-$SUITE_DIR/results/rtx4090/full_reference_3seeds}"
CONDA_ENV="${CONDA_ENV:-/shared/results/marszale/anaconda3/envs/ccex}"
WALLTIME="${WALLTIME:-1-00:00:00}"
SKIP_WARM="${SKIP_WARM:-0}"

source /home/marszale/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

N_CELLS=$(
    cd "$PROJECT_DIR" && uv run --project "$SUITE_DIR" python \
        -m experiments.zeroshot_cf.select_matrix_cell \
        --config "$BASE_CONFIG" --count
)
if [[ "$N_CELLS" -lt 1 ]]; then
    echo "Matrix config $BASE_CONFIG expands to no cells" >&2
    exit 2
fi

if [[ "$SKIP_WARM" != "1" ]]; then
    echo "Warming the shared classifier checkpoint cache for every dataset before"
    echo "submitting the array, so every method and seed loads the identical"
    echo "checkpoint instead of racing to train and overwrite it concurrently."
    mkdir -p "$RESULTS_DIR/models"
    (
        cd "$PROJECT_DIR"
        ZEROSHOT_CF_MODELS_DIR="$RESULTS_DIR/models" \
        HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" \
            uv run --project "$SUITE_DIR" python \
                -m experiments.zeroshot_cf.warm_classifier_cache \
                --config "$BASE_CONFIG"
    )
else
    echo "SKIP_WARM=1: assuming $RESULTS_DIR/models already holds every checkpoint"
fi

mkdir -p "$PROJECT_DIR/logs"

echo "Submitting $N_CELLS independent rtx4090 full-reference benchmark task(s)"
echo "(all available methods x all available datasets x 3 seeds)"
echo "partition=rtx4090_batch qos=batch, walltime=$WALLTIME"
echo "conda_env=$CONDA_ENV"
echo "base_config=$BASE_CONFIG"
echo "sbatch_file=$SBATCH_FILE"

sbatch \
    --array="0-$((N_CELLS - 1))" \
    --time="$WALLTIME" \
    "$SBATCH_FILE"

echo "After all tasks finish, aggregate with:"
echo "conda activate \"$CONDA_ENV\" && uv run --project \"$SUITE_DIR\" python -m experiments.zeroshot_cf.cli aggregate --config \"$BASE_CONFIG\""
echo "Then sync metrics with:"
echo "wandb sync \"$RESULTS_DIR/wandb\""
