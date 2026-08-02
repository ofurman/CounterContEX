#!/bin/bash -l
###############################################################################
# 20-point smoke on GH200, before spending real allocation.
#
# Runs in a THROWAWAY COPY of the repo on $SCRATCH, never in place. This is not
# tidiness — exp4_beam_search.py derives its output directory from __file__ and
# has no --results-dir, so an in-place smoke would overwrite
# results/arrays/exp4_heloc_fromscratch_cfs.npz with a 20-row array and rewrite
# exp4_summary.md from that single run. The already-finished Law and
# HELOC-frozen cells would survive (different filenames) but the real target
# cell would be silently destroyed.
#
# Proves: CUDA visible, TabPFN v2 loads offline from the staged checkpoints,
# cel + data load, the beam search produces a well-formed .npz, and
# recompute_metrics.py can rescore it.
#
#   bash plgrid/submit.sh plgrid/10_smoke.sbatch
#   MAX_TEST=40 DATASET=law bash plgrid/submit.sh plgrid/10_smoke.sbatch
###############################################################################

#SBATCH --job-name=cx_smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/cx_smoke-%j.out
#SBATCH --error=logs/cx_smoke-%j.err

set -euo pipefail
mkdir -p logs

# shellcheck source=_common.sh
source "${SLURM_SUBMIT_DIR:-${PWD}}/plgrid/_common.sh"

DATASET="${DATASET:-heloc}"
SET="${SET:-fromscratch}"
MAX_TEST="${MAX_TEST:-20}"
CHUNK="${CHUNK_SIZE:-4096}"

SANDBOX="${STORE}/smoke-${SLURM_JOB_ID}"
echo ""
echo "=== sandbox ${SANDBOX} ==="
mkdir -p "${SANDBOX}"
rsync -a \
    --exclude '.git' --exclude '__pycache__' --exclude '*.egg-info' \
    --exclude 'experiments/zeroshot_cf/results' \
    "${PROJECT}/" "${SANDBOX}/"
mkdir -p "${SANDBOX}/experiments/zeroshot_cf/results/arrays"

# Everything the experiment resolves from __file__ now points into the sandbox;
# only the venv (and the editable tabpfn install inside it, which still points
# at $HOME/projects) stays shared.
cd "${SANDBOX}"
export PYTHONPATH="${SANDBOX}"
export TABPFN_LOCAL_CACHE="${SANDBOX}/experiments/zeroshot_cf/models"

nvidia-smi || echo "WARNING: nvidia-smi not on PATH"

echo ""
echo "=== beam search: ${DATASET} / ${SET} / ${MAX_TEST} points ==="
SECONDS=0
srun python -m experiments.zeroshot_cf.exp4_beam_search \
    --dataset "${DATASET}" \
    --set "${SET}" \
    --max-test "${MAX_TEST}" \
    --chunk-size "${CHUNK}"
echo "elapsed ${SECONDS}s for ${MAX_TEST} points"

echo ""
echo "=== checking artefacts ==="
python - "${SANDBOX}" "${DATASET}" "${SET}" "${MAX_TEST}" <<'PY'
import sys
from pathlib import Path

import numpy as np

sandbox, dataset, tag, max_test = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
results = Path(sandbox) / "experiments" / "zeroshot_cf" / "results"

npz_path = results / "arrays" / f"exp4_{dataset}_{tag}_cfs.npz"
if not npz_path.exists():
    print(f"FAIL: {npz_path} was not written")
    sys.exit(1)

with np.load(npz_path) as d:
    keys = sorted(d.keys())
    x_cf = d["X_cf"]
    print(f"{npz_path.name}: keys={keys}")
    print(f"  X_cf {x_cf.shape} {x_cf.dtype}")
    expected = {
        "X_cf", "X_test", "y_test", "y_target",
        "y_pred", "immutable_idx", "chosen_valid", "immutable_drift",
    }
    missing = expected - set(keys)
    if missing:
        print(f"FAIL: missing keys {sorted(missing)}")
        sys.exit(1)
    if x_cf.shape[0] != max_test:
        print(f"FAIL: expected {max_test} rows, got {x_cf.shape[0]}")
        sys.exit(1)
    if not np.isfinite(x_cf).all():
        print("FAIL: X_cf contains non-finite values")
        sys.exit(1)
    print(f"  valid CFs {int(d['chosen_valid'].sum())}/{x_cf.shape[0]}")

csv_path = results / f"exp4_{dataset}_{tag}_metrics.csv"
if not csv_path.exists():
    print(f"FAIL: {csv_path} was not written")
    sys.exit(1)
print(f"{csv_path.name}:")
print("  " + "\n  ".join(csv_path.read_text().strip().splitlines()))
print("OK: artefacts well-formed")
PY

echo ""
echo "=== recompute_metrics.py (the salvage path) ==="
srun python -m experiments.zeroshot_cf.recompute_metrics --min-n 1
test -s "${SANDBOX}/experiments/zeroshot_cf/results/exp4_recomputed_metrics.csv" || {
    echo "FAIL: recompute_metrics.py wrote nothing" >&2
    exit 1
}
echo "recompute ok"

echo ""
echo "SMOKE PASS — sandbox left at ${SANDBOX}"
echo "Job ${SLURM_JOB_ID} finished at $(date)"
