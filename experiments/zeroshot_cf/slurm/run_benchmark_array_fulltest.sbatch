#!/bin/bash
# SLURM array job: run_full_benchmark.py --max-test -1 (the FULL test split,
# not a 256-point cap) for one dataset per array task — same 10-dataset
# BENCHMARK_CONFIG as run_benchmark_array.sbatch, same cluster/conda setup.
#
# *** Read this before submitting — full-test-set sizes vary ~100x across
# these datasets, and several are very likely to blow past common per-job
# time limits at --n-repeats 5. ***
#
# Per-dataset test-set size (experiments/zeroshot_cf/datasets/<name>/test.csv):
#   admission 100, german 200, student 226, sba 1159, default 3000,
#   adult 9045, adult_dicoflex/bank/gmc/lending-club 10000
#
# Extrapolating from the 256-point calibration in slurm/README.md (exp2
# internally chunks large batches at 256 rows via robust_impute, so this
# extrapolation is chunk-count x per-chunk-cost, not a blind linear guess)
# at --n-repeats 5:
#   admission   ~4 min      student       ~27 min     sba          ~5.4h
#   german      ~1.6h       adult         ~12h        gmc          ~13.3h
#   adult_dicoflex ~16.7h   lending-club  ~20h         default      ~45h (~1.9 days)
#   bank        ~63h (~2.6 days)  <- likely the worst case
# Serial total (all 10, one GPU): ~7.4 days. As this array, wall-clock is
# bounded by the slowest single task (bank, ~63h) — check your cluster's
# actual max job/QOS time limit (`sacctmgr show qos` or equivalent) before
# submitting; 63h may exceed it. These are still extrapolations from a
# 256-point sample, not measured at this scale — consider a smaller
# --n-repeats (e.g. 1) for the four 10,000-row datasets specifically if the
# full 5-repeat run doesn't fit your job-time budget; edit the DATASETS loop
# below or submit those four with a separate, shorter override.
#
# Before submitting: same prerequisites as run_benchmark_array.sbatch (stage
# TabPFN checkpoints, copy experiments/zeroshot_cf/datasets/ over, `wandb
# login` once) — see slurm/README.md.
#
# Submit with:  sbatch run_benchmark_array_fulltest.sbatch
# Single (cheap) dataset only:  sbatch --array=2 run_benchmark_array_fulltest.sbatch   (2=admission)
#
#SBATCH --job-name=ccex-full
#SBATCH --qos=batch
#SBATCH --time=72:00:00
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

echo "=== $(date) : starting FULL/$DATASET (array task $SLURM_ARRAY_TASK_ID) on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

export HF_HUB_OFFLINE=1
export TABPFN_DEVICE=cuda
export WANDB_MODE=online
export WANDB_PROJECT=CounterContEX

python experiments/zeroshot_cf/run_full_benchmark.py \
    --dataset "$DATASET" \
    --max-test -1 \
    --n-permutations 1 \
    --n-repeats 5 \
    --seed 42 \
    --wandb-project "$WANDB_PROJECT"

echo "=== $(date) : finished FULL/$DATASET ==="
