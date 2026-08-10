#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
CASE_FILE="${CASE_FILE:-$PROJECT_DIR/experiments/zeroshot_cf/athena/exp8_backend_cases.tsv}"
SBATCH_FILE="${SBATCH_FILE:-$PROJECT_DIR/experiments/zeroshot_cf/athena/exp8_backend_array.sbatch}"

N_CASES=$(awk 'NF && $1 !~ /^#/ {c++} END {print c + 0}' "$CASE_FILE")
if [[ "$N_CASES" -lt 1 ]]; then
    echo "No runnable cases in $CASE_FILE" >&2
    exit 2
fi

echo "Submitting $N_CASES Athena Exp8 backend-comparison task(s)"
echo "case_file=$CASE_FILE"
echo "sbatch_file=$SBATCH_FILE"

sbatch --array="0-$((N_CASES - 1))" "$SBATCH_FILE"
