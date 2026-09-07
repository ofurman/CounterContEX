#!/bin/bash -l
# Run, strictly aggregate, and build the paired analysis for clean E3.

#SBATCH --job-name=cx_e3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/cx_e3-%j.out
#SBATCH --error=logs/cx_e3-%j.err

set -euo pipefail
mkdir -p logs
# shellcheck source=_common.sh
source "${SLURM_SUBMIT_DIR:-${PWD}}/experiments/zeroshot_cf/plgrid/_common.sh"

MATRIX="experiments/zeroshot_cf/configs/matrices/campaign_e3_clean_backend.yaml"
test -f "${MATRIX}" || { echo "ERROR: no matrix at ${MATRIX}" >&2; exit 1; }
python experiments/zeroshot_cf/vendor_setup.py --check
nvidia-smi || true

SECONDS=0
python -m experiments.zeroshot_cf.cli matrix --config "${MATRIX}" --resume
echo "generation finished in ${SECONDS}s"
python -m experiments.zeroshot_cf.cli aggregate --config "${MATRIX}"
python -m experiments.zeroshot_cf.cli analyze-e3 --config "${MATRIX}" \
    --output experiments/zeroshot_cf/results/campaign/e3_clean_backend_analysis

echo "Job ${SLURM_JOB_ID} finished at $(date)"
