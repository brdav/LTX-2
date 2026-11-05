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
DATA=${DATA:-$CAP/processed_datasets/natix}  # dataset root, e.g. $CAP/processed_datasets/natix-jp-uk
TEXT=${TEXT:-text_predicted}  # captions dir under $DATA/osm_bev, e.g. text_predicted_v2
OUT=${OUT:-$DATA/tokenized/text_route_${TEXT#text_}}
EXTRA=${EXTRA:-}
cd "$HOME/LTX-2"

# TASKS="0 1 2 3" TASK_COUNT=4: run several array tasks one after another in a single job (e.g. on debug, which
# limits submitted jobs per user). Default: this array task only.
TASK_COUNT=${TASK_COUNT:-${SLURM_ARRAY_TASK_COUNT:-1}}
TASKS=${TASKS:-${SLURM_ARRAY_TASK_ID:-0}}
export NUM_SHARDS=$((4 * TASK_COUNT))
export CAP DATA TEXT OUT EXTRA

for TASK_ID in $TASKS; do
export BASE_SHARD=$((4 * TASK_ID))
srun --kill-on-bad-exit=0 bash -c '
exec .venv/bin/python packages/ltx-trainer/scripts/tokenize_natix_text.py \
    --text-root $DATA/osm_bev/$TEXT \
    --output-dir $OUT \
    --model-path $CAP/ltx-2-models/ltx-2.5/diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors \
    --text-encoder-path $CAP/ltx-2-models/ltx-2.5/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
    --shard-index $((BASE_SHARD + SLURM_PROCID)) --num-shards $NUM_SHARDS $EXTRA'
done
