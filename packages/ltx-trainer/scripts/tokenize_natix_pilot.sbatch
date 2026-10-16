#!/bin/bash
#SBATCH --job-name=natix-pilot
#SBATCH --account=ab036
#SBATCH --partition=preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --output=/iopsstor/scratch/cscs/%u/tokenized_datasets/natix_pilot_%j.log

set -euo pipefail
CAP=/capstor/store/cscs/swissai/ab036
cd "$HOME/LTX-2"

.venv/bin/python packages/ltx-trainer/scripts/tokenize_natix.py \
    --manifest $CAP/processed_datasets/natix/manifest/samples_24fps.parquet \
    --data-root $CAP/datasets/natix \
    --output-dir /iopsstor/scratch/cscs/$USER/tokenized_datasets/natix_pilot_768x512 \
    --vae-path $CAP/checkpoints/ltx-2.5/vae/ltx-2.5-video-vae-bf16.safetensors \
    --resolution 768x512 \
    --max-samples 100 \
    --tmp-dir "${SLURM_TMPDIR:-/tmp}"
