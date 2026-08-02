#!/bin/bash
###############################################################################
# Submit a PLGrid job with the account and partition from plg-config.sh.
#
# The .sbatch files deliberately carry no --account/--partition: those are grant
# identifiers that change between users and allocations, and a stale one in a
# committed header fails at 3am. They come from config, here, every time.
#
#   bash plgrid/submit.sh --test-only plgrid/00_setup_env.sbatch
#   bash plgrid/submit.sh plgrid/00_setup_env.sbatch
#   MAX_TEST=20 bash plgrid/submit.sh plgrid/10_smoke.sbatch
#
# Any extra sbatch flags are passed through, so dependencies work as usual:
#   bash plgrid/submit.sh --dependency=afterok:12345 plgrid/20_beam_run.sbatch
###############################################################################
set -euo pipefail
cd "$(dirname "$0")/.."

# shellcheck source=plg-config.sh
source "${PWD}/plgrid/plg-config.sh"
plg_require PLG_ACCOUNT PLG_PARTITION PLG_GROUP

if [ "$#" -eq 0 ]; then
    echo "usage: bash plgrid/submit.sh [sbatch flags] <script.sbatch>" >&2
    exit 1
fi

# These have to reach the job. Slurm exports the submission environment by
# default, but be explicit so a site policy change cannot silently break it.
export PLG_GROUP PROJECT_NAME ML_BUNDLE VENV_NAME COMPUTE_ARCH PY_VERSION
export TABPFN_MODEL_SUBDIR

set -x
exec sbatch \
    --account="${PLG_ACCOUNT}" \
    --partition="${PLG_PARTITION}" \
    --export=ALL \
    "$@"
