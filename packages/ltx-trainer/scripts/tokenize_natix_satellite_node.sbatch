#!/bin/bash
# One full node (4 GPUs / 288 cores): 4 processes, one per GPU, each with CPU workers feeding it.
# Tokenizes the Natix satellite crops, one still image = a 1-frame video. Rerun later to pick up zips that were still being generated.
# Usage: sbatch [--array=0-N%K] [--export=ALL,OUT=...,RES=WxH,EXTRA="--num-sessions 80"] tokenize_natix_satellite_node.sbatch
# Each array task takes 4 shards; total shards = 4 * array size.
#SBATCH --job-name=natix-sat-tok
#SBATCH --account=ab036
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=72
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/capstor/store/cscs/swissai/ab036/processed_datasets/natix/tokenized/logs/natix_sat_%A_%a.log

# Use the following:
# cd ~/LTX-2
# sbatch --array=0-1 packages/ltx-trainer/scripts/tokenize_natix_satellite_node.sbatch

set -euo pipefail
CAP=/capstor/store/cscs/swissai/ab036
RES=${RES:-768x768}
OUT=${OUT:-/capstor/store/cscs/swissai/ab036/processed_datasets/natix/tokenized/natix_satellite_${RES}}
WORKERS=${WORKERS:-14}
EXTRA=${EXTRA:-}
cd "$HOME/LTX-2"

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
TASK_COUNT=${SLURM_ARRAY_TASK_COUNT:-1}
export NUM_SHARDS=$((4 * TASK_COUNT))
export BASE_SHARD=$((4 * TASK_ID))
export CAP RES OUT WORKERS EXTRA

# --kill-on-bad-exit=0: a single failing GPU process must not take the others down; rerun resumes anyway.
srun --kill-on-bad-exit=0 bash -c '
exec .venv/bin/python packages/ltx-trainer/scripts/tokenize_natix_satellite.py \
    --sat-root $CAP/processed_datasets/natix/satellite/images \
    --output-dir $OUT \
    --vae-path $CAP/checkpoints/ltx-2.5/vae/ltx-2.5-video-vae-bf16.safetensors \
    --resolution $RES \
    --shard-index $((BASE_SHARD + SLURM_PROCID)) --num-shards $NUM_SHARDS \
    --num-workers $WORKERS --prefetch 1 $EXTRA'
