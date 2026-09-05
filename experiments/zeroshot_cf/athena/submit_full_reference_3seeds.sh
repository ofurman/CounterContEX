#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
SUITE_DIR="${SUITE_DIR:-$PROJECT_DIR/experiments/zeroshot_cf}"
SBATCH_FILE="${SBATCH_FILE:-$SUITE_DIR/athena/full_reference_3seeds_array.sbatch}"
BASE_CONFIG="${BASE_CONFIG:-$SUITE_DIR/configs/matrices/full_reference_3seeds.yaml}"
RESULTS_DIR="${RESULTS_DIR:-$SUITE_DIR/results/athena/full_reference_3seeds}"
WALLTIME="${WALLTIME:-10:00:00}"
SKIP_WARM="${SKIP_WARM:-0}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
    PYTHON_CMD=("$PYTHON_BIN")
elif [[ -x "$SUITE_DIR/.venv/bin/python" ]]; then
    PYTHON_CMD=("$SUITE_DIR/.venv/bin/python")
elif command -v uv >/dev/null 2>&1; then
    PYTHON_CMD=(uv run --project "$SUITE_DIR" python)
else
    PYTHON_CMD=(python)
fi

N_CELLS=$(
    cd "$PROJECT_DIR" && "${PYTHON_CMD[@]}" -m experiments.zeroshot_cf.select_matrix_cell \
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
            "${PYTHON_CMD[@]}" -m experiments.zeroshot_cf.warm_classifier_cache \
                --config "$BASE_CONFIG"
    )
else
    echo "SKIP_WARM=1: assuming $RESULTS_DIR/models already holds every checkpoint"
fi

echo "Submitting $N_CELLS independent Athena full-reference benchmark task(s)"
echo "(all available methods x all available datasets x 3 seeds)"
echo "account=plgfoundationeconom-gpu-a100 partition=plgrid-gpu-a100"
echo "resources per task: 1xA100, 8 CPUs, 64G RAM, walltime $WALLTIME"
echo "base_config=$BASE_CONFIG"
echo "sbatch_file=$SBATCH_FILE"

sbatch \
    --array="0-$((N_CELLS - 1))" \
    --time="$WALLTIME" \
    --export="ALL,COUNTERCONTEX_SLURM_WALLTIME=$WALLTIME" \
    "$SBATCH_FILE"

echo "After all tasks finish, aggregate with:"
echo "uv run --project \"$SUITE_DIR\" python -m experiments.zeroshot_cf.cli aggregate --config \"$BASE_CONFIG\""
