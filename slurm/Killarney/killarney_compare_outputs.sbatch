#!/bin/bash
#SBATCH -A aip-rgrosse
#SBATCH -D /project/6105522/memoozd/harmful-agents-meta-dataset
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --job-name=cb_compare
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Quick diagnostic: see actual model outputs side by side

set -euo pipefail
mkdir -p logs

source /project/6105522/memoozd/.venvs/cb_env/bin/activate
cd /project/6105522/memoozd/harmful-agents-meta-dataset

export HF_HOME="/project/6105522/memoozd/cb_cache/hf"

# Find latest checkpoint
CB_CKPT=$(ls -td ~/scratch/cb_runs/*/outputs/cb_llama31_8b_instruct_v2/final 2>/dev/null | head -1)
echo "Using checkpoint: $CB_CKPT"

python scripts/circuit_breakers/compare_outputs.py \
  --base-model meta-llama/Llama-3.1-8B-Instruct \
  --adapter-path "$CB_CKPT"
