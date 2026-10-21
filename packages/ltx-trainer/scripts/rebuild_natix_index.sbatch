#!/bin/bash
# CPU-only job that rebuilds the clip index of a finished tokenization run (see rebuild_natix_index.py).
# Usage: RES=1248x832 NUM_SHARDS=64 sbatch [--dependency=afterok:<tokenization jobid>] rebuild_natix_index.sbatch
#SBATCH --job-name=natix-index
#SBATCH --account=ab036
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=03:00:00
#SBATCH --output=/capstor/store/cscs/swissai/ab036/processed_datasets/natix/tokenized/logs/natix_index_%j.log

set -euo pipefail
CAP=/capstor/store/cscs/swissai/ab036
RES=${RES:?set RES, e.g. 1248x832}
NUM_SHARDS=${NUM_SHARDS:?set NUM_SHARDS (4 x array size of the tokenization job)}
cd "$HOME/LTX-2"

.venv/bin/python -I packages/ltx-trainer/scripts/rebuild_natix_index.py \
    $CAP/processed_datasets/natix/tokenized/natix_${RES} \
    --num-shards $NUM_SHARDS \
    --manifest $CAP/processed_datasets/natix/manifest/samples_24fps.parquet \
    --backup-dir /iopsstor/scratch/cscs/$USER/tokenized_datasets/_index_backup_${RES} \
    --threads 128
