#!/bin/bash
# SLURM array job: run_full_benchmark.py --max-test 256 for one dataset per
# array task (10 tasks total — the datasets in run_full_benchmark.BENCHMARK_CONFIG).
#
# Configured for PLGrid (plgrid-gpu-a100 / plgtabanomdet-gpu-a100).
#
# Before submitting:
#   1. Run stage_offline_assets.sh once on a LOGIN NODE (network access) to
#      download the TabPFN v2 checkpoints — compute nodes are usually offline.
#   2. Make sure experiments/zeroshot_cf/datasets/<name>/{train,val,test}.csv
#      exist on the cluster (they're gitignored — copy them over, or re-run
#      the porting step from ../CETGFN if that repo is available there too).
#   3. `wandb login` once on a login node (writes ~/.netrc, which compute
#      nodes will also see) if you want the offline runs below synced later.
#
# Each dataset is logged to W&B as its own run (name: "<dataset>-<disc_type>-
# <suite>-mt256-nr5-seed42"), but compute nodes are usually offline — this
# script sets WANDB_MODE=offline (writes locally under <repo_root>/wandb/,
# since run_full_benchmark.py is invoked from the repo root below) and prints
# a `wandb sync` command per run. From a login node afterward, sync everything
# from this job at once:
#   wandb sync wandb/offline-run-*
#
# Submit with:  sbatch run_benchmark_array.sbatch
# Single dataset only:  sbatch --array=0 run_benchmark_array.sbatch   (0=german)
#
#SBATCH --job-name=ccex
#SBATCH --qos=batch
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --partition=rtx4090_batch
#SBATCH --mem=12G
#SBATCH --array=0-9
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

cd /shared/results/marszale/CounterContEX
source activate /shared/results/marszale/anaconda3/envs/ccex

# Array index -> dataset name (must match run_full_benchmark.BENCHMARK_CONFIG's
# key order: german, adult, admission, student, adult_dicoflex, bank, default,
# gmc, lending-club, sba).
DATASETS=(german adult admission student adult_dicoflex bank default gmc lending-club sba)
DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"

mkdir -p logs
# cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../../..}"   # repo root (script lives in experiments/zeroshot_cf/slurm/)

echo "=== $(date) : starting $DATASET (array task $SLURM_ARRAY_TASK_ID) on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Conda env (activated above) — see slurm/README.md "Conda setup" for the
# one-time `conda create`/`pip install` commands.
export HF_HUB_OFFLINE=1
export TABPFN_DEVICE=cuda
export WANDB_MODE=online               # compute nodes are usually offline; `wandb sync` later
export WANDB_PROJECT=CounterContEX

python experiments/zeroshot_cf/run_full_benchmark.py \
    --dataset "$DATASET" \
    --max-test 256 \
    --n-permutations 1 \
    --n-repeats 5 \
    --seed 42 \
    --wandb-project "$WANDB_PROJECT"

echo "=== $(date) : finished $DATASET ==="
