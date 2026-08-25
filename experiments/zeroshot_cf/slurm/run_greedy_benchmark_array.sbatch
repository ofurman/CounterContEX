#!/bin/bash
# SLURM array job: run_greedy_benchmark.py (exp4, classifier-in-the-loop greedy
# CF) for one dataset per array task (10 tasks — same BENCHMARK_CONFIG dataset
# list as run_benchmark_array.sbatch / run_full_benchmark.py).
#
# Configured for PLGrid (plgrid-gpu-a100 / plgtabanomdet-gpu-a100) — same
# account/partition/conda env as run_benchmark_array.sbatch.
#
# *** --time is a rough extrapolation, not a calibrated number — see below. ***
#
# Before submitting: same prerequisites as run_benchmark_array.sbatch (stage
# TabPFN checkpoints + copy experiments/zeroshot_cf/datasets/ over + `wandb
# login` once) — see slurm/README.md.
#
# Why this is a SEPARATE, more cautious script from run_benchmark_array.sbatch:
# exp4's default "prob_ascent" selector calls sampler.sample_feature once per
# REMAINING actionable candidate at EVERY greedy step (greedy.py's
# _select_prob_ascent) — worst case (budget exhausted without an early flip)
# is ~n_actionable*(n_actionable+1)/2 individual, UNBATCHED TabPFN calls per
# point, vs. exp2's one batched call across the whole test set. One-point
# local calibration (RTX 4070 Ti): admission (6 actionable) exhausted its
# full budget WITHOUT flipping in ~25s; german (16 actionable) flipped after
# only 2 of 16 steps in ~26s. Cost and success are inversely correlated here —
# the datasets exp2 already struggled with (gmc, lending-club, bank, default)
# are plausibly both the slowest (full budget, every time) AND the least
# likely to flip. --max-test therefore defaults to 16 here, not 256 — see
# run_greedy_benchmark.py's module docstring for the extrapolated per-dataset
# worst-case cost table. RECALIBRATE (--max-test 1-4 on `default` specifically,
# the worst case) on the actual A100 before trusting --time below, or before
# ever raising --max-test past 16.
#
# Submit with:  sbatch run_greedy_benchmark_array.sbatch
# Single dataset only:  sbatch --array=0 run_greedy_benchmark_array.sbatch   (0=german)
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

# Array index -> dataset name (must match run_greedy_benchmark.BENCHMARK_CONFIG's
# key order: german, adult, admission, student, adult_dicoflex, bank, default,
# gmc, lending-club, sba).
DATASETS=(german adult admission student adult_dicoflex bank default gmc lending-club sba)
DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"

mkdir -p logs

echo "=== $(date) : starting greedy/$DATASET (array task $SLURM_ARRAY_TASK_ID) on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

export HF_HUB_OFFLINE=1
export TABPFN_DEVICE=cuda
export WANDB_MODE=online
export WANDB_PROJECT=CounterContEX

python experiments/zeroshot_cf/run_greedy_benchmark.py \
    --dataset "$DATASET" \
    --max-test 16 \
    --selector prob_ascent \
    --tau 0.5 \
    --n-repeats 1 \
    --seed 42 \
    --wandb-project "$WANDB_PROJECT"

echo "=== $(date) : finished greedy/$DATASET ==="
