#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
SBATCH_FILE="$PROJECT_DIR/experiments/zeroshot_cf/athena/exp10_lof_array.sbatch"

echo "Submitting paired 50-point HELOC LOF ablation"
echo "Two independent A100 tasks, 8 CPUs, 64G RAM, 2 hours each"
sbatch --array=0-1 "$SBATCH_FILE"
