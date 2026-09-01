#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
SUITE_DIR="${SUITE_DIR:-$PROJECT_DIR/experiments/zeroshot_cf}"
CASE_FILE="${CASE_FILE:-$SUITE_DIR/athena/exp9_countercontex_cases.tsv}"
SBATCH_FILE="${SBATCH_FILE:-$SUITE_DIR/athena/exp9_countercontex_array.sbatch}"
WALLTIME="${WALLTIME:-10:00:00}"

N_CASES=$(awk 'NF && $1 !~ /^#/ {c++} END {print c + 0}' "$CASE_FILE")
if [[ "$N_CASES" -lt 1 ]]; then
    echo "No runnable cases in $CASE_FILE" >&2
    exit 2
fi

echo "Submitting $N_CASES independent Athena Exp9 dataset task(s)"
echo "account=plgfoundationeconom-gpu-a100 partition=plgrid-gpu-a100"
echo "resources per task: 1xA100, 8 CPUs, 64G RAM, walltime $WALLTIME"
echo "case_file=$CASE_FILE"
echo "sbatch_file=$SBATCH_FILE"
echo "suite_dir=$SUITE_DIR"

sbatch \
    --array="0-$((N_CASES - 1))" \
    --time="$WALLTIME" \
    --export="ALL,COUNTERCONTEX_SLURM_WALLTIME=$WALLTIME" \
    "$SBATCH_FILE"

echo "After all tasks finish, aggregate with:"
echo "uv run --project \"$SUITE_DIR\" python -m experiments.zeroshot_cf.exp9_countercontex_benchmark --dataset aggregate"
