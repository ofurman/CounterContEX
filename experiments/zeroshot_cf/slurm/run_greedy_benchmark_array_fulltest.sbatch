#!/bin/bash
# SLURM array job: run_greedy_benchmark.py --max-test -1 (the FULL test split)
# for one dataset per array task — exp4's classifier-in-the-loop greedy
# search, same 10-dataset BENCHMARK_CONFIG, same cluster/conda setup as
# run_greedy_benchmark_array.sbatch.
#
# *** READ THIS — 6 of the 10 datasets are very likely INFEASIBLE at this
# configuration, not just slow. This is not a --time tuning problem. ***
#
# exp4's default selector is ~n_actionable^2/2 unbatched TabPFN calls per
# point WORST CASE (budget exhausted, no early flip — plausible for most
# points on the datasets exp2 already struggled with). Extrapolating from the
# admission calibration (~0.8s/call) to each dataset's actual full test-set
# size, at --n-repeats 5, WORST CASE:
#
#   admission    ~2.3h   (0.1 days)   <- fine
#   student     ~13.8h   (0.6 days)   <- fine
#   german      ~30.2h   (1.3 days)   <- fine with --time below
#   sba        ~117.2h   (4.9 days)   <- needs --time raised
#   gmc        ~611.1h  (25.5 days)   <- NOT FEASIBLE as configured
#   adult      ~361.8h  (15.1 days)   <- NOT FEASIBLE as configured
#   lending-club ~866.7h (36.1 days)  <- NOT FEASIBLE as configured
#   adult_dicoflex ~866.7h (36.1 days) <- NOT FEASIBLE as configured
#   default     ~920.0h  (38.3 days)  <- NOT FEASIBLE as configured
#   bank       ~1511.1h  (63.0 days)  <- NOT FEASIBLE as configured
#
# (real cost will be lower wherever points flip before exhausting budget —
# german's own calibration point flipped in 2/16 steps, well under this
# worst case — but datasets with near-zero exp2 validity are exactly the
# ones plausible to hit worst case often, so treat this table as realistic,
# not merely pessimistic.)
#
# This script is submitted AS REQUESTED (whole test set, all 10 datasets,
# n_repeats=5) — but --time below is only sized for the datasets that are
# actually feasible (german/student/admission/sba). The other six array
# tasks WILL be killed by the time limit before finishing, producing no
# result for those datasets, not just a slow one. Before running this for
# real, pick one:
#   - Submit only the feasible subset:  sbatch --array=0,2,3,9 run_greedy_benchmark_array_fulltest.sbatch
#     (0=german 2=admission 3=student 9=sba)
#   - Cap --max-test for the other six specifically (edit per-dataset below,
#     or run them separately with e.g. --max-test 500).
#   - Accept partial/killed results for those six and treat this as a
#     best-effort exploratory run.
#
# Before submitting: same prerequisites as run_greedy_benchmark_array.sbatch
# (stage TabPFN checkpoints, copy experiments/zeroshot_cf/datasets/ over,
# `wandb login` once) — see slurm/README.md.
#
# Submit with:  sbatch run_greedy_benchmark_array_fulltest.sbatch
# Feasible subset only:  sbatch --array=0,2,3,9 run_greedy_benchmark_array_fulltest.sbatch
#
#SBATCH --job-name=ccex-greedy-full
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

# Array index -> dataset name (must match run_greedy_benchmark.BENCHMARK_CONFIG's
# key order: german, adult, admission, student, adult_dicoflex, bank, default,
# gmc, lending-club, sba).
DATASETS=(german adult admission student adult_dicoflex bank default gmc lending-club sba)
DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"

mkdir -p logs

echo "=== $(date) : starting greedy-FULL/$DATASET (array task $SLURM_ARRAY_TASK_ID) on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

export HF_HUB_OFFLINE=1
export TABPFN_DEVICE=cuda
export WANDB_MODE=online
export WANDB_PROJECT=CounterContEX

python experiments/zeroshot_cf/run_greedy_benchmark.py \
    --dataset "$DATASET" \
    --max-test -1 \
    --selector prob_ascent \
    --tau 0.5 \
    --n-repeats 5 \
    --seed 42 \
    --wandb-project "$WANDB_PROJECT"

echo "=== $(date) : finished greedy-FULL/$DATASET ==="
