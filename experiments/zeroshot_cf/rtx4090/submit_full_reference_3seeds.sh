#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/shared/results/marszale/CounterContEX}"
SUITE_DIR="${SUITE_DIR:-$PROJECT_DIR/experiments/zeroshot_cf}"
SBATCH_FILE="${SBATCH_FILE:-$SUITE_DIR/rtx4090/full_reference_3seeds_array.sbatch}"
WARM_SBATCH_FILE="${WARM_SBATCH_FILE:-$SUITE_DIR/rtx4090/warm_classifier_cache.sbatch}"
BASE_CONFIG="${BASE_CONFIG:-$SUITE_DIR/configs/matrices/full_reference_3seeds_rtx4090.yaml}"
RESULTS_DIR="${RESULTS_DIR:-$SUITE_DIR/results/rtx4090/full_reference_3seeds}"
CONDA_ENV="${CONDA_ENV:-/shared/results/marszale/anaconda3/envs/ccex}"
WALLTIME="${WALLTIME:-1-00:00:00}"
SKIP_WARM="${SKIP_WARM:-0}"

# select_matrix_cell.py only touches yaml/argparse/itertools -- no numpy/torch
# -- so it's safe to run right here on the login node just to count cells.
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

mkdir -p "$PROJECT_DIR/logs"

# The actual classifier warm-up imports numpy/sklearn, which needs a real
# compute node's CPU (the login node's is too restricted -- see the
# x86-64-v2 NumPy crash this was split out to avoid). So it's always
# submitted as its own Slurm job, never run inline here.
SBATCH_ARGS=(
    --array="0-$((N_CELLS - 1))"
    --time="$WALLTIME"
)
if [[ "$SKIP_WARM" != "1" ]]; then
    echo "Submitting the classifier checkpoint warm-up job first. The benchmark"
    echo "array is queued right behind it with --dependency=afterok, so it only"
    echo "starts once every dataset's checkpoint has been trained and cached --"
    echo "no array task can race another into training/overwriting it."
    WARM_JOB_ID=$(sbatch --parsable "$WARM_SBATCH_FILE")
    echo "warm job id=$WARM_JOB_ID"
    SBATCH_ARGS+=(--dependency="afterok:$WARM_JOB_ID")
else
    echo "SKIP_WARM=1: assuming $RESULTS_DIR/models already holds every checkpoint"
fi

echo "Submitting $N_CELLS independent rtx4090 full-reference benchmark task(s)"
echo "(all available methods x all available datasets x 3 seeds)"
echo "partition=rtx4090_batch qos=batch, walltime=$WALLTIME"
echo "conda_env=$CONDA_ENV"
echo "base_config=$BASE_CONFIG"
echo "sbatch_file=$SBATCH_FILE"

ARRAY_JOB_ID=$(sbatch --parsable "${SBATCH_ARGS[@]}" "$SBATCH_FILE")
echo "array job id=$ARRAY_JOB_ID"
if [[ "$SKIP_WARM" != "1" ]]; then
    echo "(pending until warm job $WARM_JOB_ID completes successfully)"
fi

echo "After all tasks finish, aggregate with:"
echo "conda activate \"$CONDA_ENV\" && uv run --project \"$SUITE_DIR\" python -m experiments.zeroshot_cf.cli aggregate --config \"$BASE_CONFIG\""
echo "Then sync metrics with:"
echo "wandb sync \"$RESULTS_DIR/wandb\""
