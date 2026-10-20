#!/bin/bash
# One full node (4 GPUs / 288 cores): 4 processes, one per GPU, each with CPU workers feeding it.
# Usage: sbatch [--array=0-N%K] [--export=ALL,OUT=...,RES=WxH,EXTRA="--num-videos 80"] tokenize_natix_node.sbatch
# Each array task takes 4 shards; total shards = 4 * array size.
#SBATCH --job-name=natix-tok
#SBATCH --account=ab036
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=72
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/capstor/store/cscs/swissai/ab036/processed_datasets/natix/tokenized/logs/natix_%A_%a.log

# Use the following:
# cd ~/LTX-2
# RES=768x512 WORKERS=14 sbatch --array=0-7 packages/ltx-trainer/scripts/tokenize_natix_node.sbatch
# RES=1248x832 WORKERS=4 sbatch --array=0-15 packages/ltx-trainer/scripts/tokenize_natix_node.sbatch

set -euo pipefail
CAP=/capstor/store/cscs/swissai/ab036
RES=${RES:-768x512}
OUT=${OUT:-/capstor/store/cscs/swissai/ab036/processed_datasets/natix/tokenized/natix_${RES}}
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
exec .venv/bin/python packages/ltx-trainer/scripts/tokenize_natix.py \
    --manifest $CAP/processed_datasets/natix/manifest/samples_24fps.parquet \
    --data-root $CAP/datasets/natix \
    --output-dir $OUT \
    --vae-path $CAP/checkpoints/ltx-2.5/vae/ltx-2.5-video-vae-bf16.safetensors \
    --resolution $RES \
    --shard-index $((BASE_SHARD + SLURM_PROCID)) --num-shards $NUM_SHARDS \
    --num-workers $WORKERS --prefetch 1 \
    --tmp-dir "${SLURM_TMPDIR:-/tmp}" $EXTRA'
