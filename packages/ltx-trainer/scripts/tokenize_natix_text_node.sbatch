#!/bin/bash
# One full node (4 GPUs): 4 processes, one per GPU. Encodes the Natix route captions (osm_bev/text_predicted) with Gemma + LTX embeddings processor.
# Usage: sbatch [--array=0-N%K] [--export=ALL,OUT=...,EXTRA="--num-sessions 80"] tokenize_natix_text_node.sbatch
# Each array task takes 4 shards; total shards = 4 * array size.
#SBATCH --job-name=natix-text-tok
#SBATCH --account=ab036
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=72
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/capstor/store/cscs/swissai/ab036/processed_datasets/natix/tokenized/logs/natix_text_%A_%a.log

# Use the following:
# cd ~/LTX-2
# sbatch --array=0-3 packages/ltx-trainer/scripts/tokenize_natix_text_node.sbatch

set -euo pipefail
CAP=/capstor/store/cscs/swissai/ab036
OUT=${OUT:-$CAP/processed_datasets/natix/tokenized/text_route_predicted}
EXTRA=${EXTRA:-}
cd "$HOME/LTX-2"

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
TASK_COUNT=${SLURM_ARRAY_TASK_COUNT:-1}
export NUM_SHARDS=$((4 * TASK_COUNT))
export BASE_SHARD=$((4 * TASK_ID))
export CAP OUT EXTRA

srun --kill-on-bad-exit=0 bash -c '
exec .venv/bin/python packages/ltx-trainer/scripts/tokenize_natix_text.py \
    --text-root $CAP/processed_datasets/natix/osm_bev/text_predicted \
    --output-dir $OUT \
    --model-path $CAP/ltx-2-models/ltx-2.5/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors \
    --text-encoder-path $CAP/ltx-2-models/ltx-2.5/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
    --shard-index $((BASE_SHARD + SLURM_PROCID)) --num-shards $NUM_SHARDS $EXTRA'
