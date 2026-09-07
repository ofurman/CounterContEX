#!/bin/bash
# Inject the live allocation at submission time; keep it out of SBATCH headers.
set -euo pipefail
cd "$(dirname "$0")/../../.."

PLG_ACCOUNT="${PLG_ACCOUNT:-plgcountercontex-gpu-gh200}"
PLG_PARTITION="${PLG_PARTITION:-plgrid-gpu-gh200}"
[ "$#" -gt 0 ] || { echo "usage: submit.sh [sbatch flags] <script.sbatch>" >&2; exit 1; }
mkdir -p logs
exec sbatch --account="${PLG_ACCOUNT}" --partition="${PLG_PARTITION}" --export=ALL "$@"
