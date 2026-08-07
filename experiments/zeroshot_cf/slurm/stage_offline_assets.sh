#!/bin/bash
# Run ONCE on a login node (needs network) before submitting run_benchmark_array.sbatch —
# SLURM compute nodes are usually offline, and TabPFN checkpoints / project
# dependencies must already be present locally when the array job runs.
#
# Usage: bash experiments/zeroshot_cf/slurm/stage_offline_assets.sh
set -euo pipefail

cd "$(dirname "$0")/../../.."   # repo root

echo "=== Installing dependencies (uv) ==="
uv pip install -r experiments/zeroshot_cf/requirements.txt

echo "=== Staging TabPFN v2 checkpoints into experiments/zeroshot_cf/models/ ==="
uv run python -c "from experiments.zeroshot_cf.checkpoints import stage_checkpoints; stage_checkpoints()"

echo "=== Verifying the 10 ported datasets are present ==="
missing=0
for name in german adult admission student adult_dicoflex bank default gmc lending-club sba; do
    d="experiments/zeroshot_cf/datasets/$name"
    for f in config.json train.csv test.csv; do
        if [ ! -f "$d/$f" ]; then
            echo "  MISSING: $d/$f"
            missing=1
        fi
    done
done
if [ "$missing" -eq 1 ]; then
    echo
    echo "Some dataset files are missing. They are gitignored (large CSVs/checkpoints)"
    echo "— copy experiments/zeroshot_cf/datasets/ over from a machine that has them"
    echo "(e.g. rsync -av datasets/ user@cluster:.../experiments/zeroshot_cf/datasets/),"
    echo "or re-run the porting step against a local ../CETGFN checkout if available"
    echo "on the cluster too."
    exit 1
fi

echo "=== Offline smoke test (1 point, no network, no W&B) ==="
HF_HUB_OFFLINE=1 TABPFN_DEVICE=cpu uv run python experiments/zeroshot_cf/run_full_benchmark.py \
    --dataset german --max-test 1 --n-repeats 1 --no-wandb

echo
echo "=== W&B login (optional — only needed if you want run_benchmark_array.sbatch's"
echo "    offline W&B runs synced to the cloud later) ==="
if uv run python -c "import wandb" 2>/dev/null; then
    uv run wandb login || echo "  Skipped — run 'wandb login' manually later if you want W&B."
else
    echo "  wandb not installed — skipping (it's in requirements.txt if you want it)."
fi

echo
echo "All set. Submit the array job with:"
echo "  sbatch experiments/zeroshot_cf/slurm/run_benchmark_array.sbatch"
