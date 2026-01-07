#!/bin/bash
#SBATCH -A aip-rgrosse
#SBATCH -D /project/6105522/memoozd/harmful-agents-meta-dataset
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --job-name=diagnose_cb_lite
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# =============================================================================
# Circuit Breakers Pipeline Diagnostic - Lite Version (1×L40S)
# =============================================================================
#
# This script runs data ingestion and tokenization checks without loading
# the full model onto GPU. Fast and lightweight.
#
# Checks:
# 1. Data ingestion and format validation
# 2. Tokenization and masking correctness
# 3. Sample harmful/benign examples
#
# Usage:
#   sbatch slurm/Killarney/killarney_diagnose_lite.sbatch
#
# =============================================================================

set -euo pipefail

mkdir -p logs

PROJECT_DIR="/project/6105522/memoozd"
REPO_DIR="/project/6105522/memoozd/harmful-agents-meta-dataset"
VENV_DIR="/project/6105522/memoozd/.venvs/cb_env"

cd "$REPO_DIR"

# Load modules
module --force purge || true
module load StdEnv/2023
module load cuda/12.6
module load python/3.11.5

if [[ ! -d "$VENV_DIR" ]]; then
  echo "ERROR: venv not found at $VENV_DIR"
  exit 1
fi

source "$VENV_DIR/bin/activate"

echo "Python: $(python -V)"
echo "Which:  $(which python)"

# =============================================================================
# Environment
# =============================================================================

# API Tokens
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "WARNING: HF_TOKEN not set. Model download may fail for gated models."
fi

CACHE_ROOT="$PROJECT_DIR/cb_cache"
mkdir -p "$CACHE_ROOT"/{hf,wandb,torch,xdg}
export HF_HOME="$CACHE_ROOT/hf"
export HF_DATASETS_CACHE="$CACHE_ROOT/hf/datasets"
export WANDB_DIR="$CACHE_ROOT/wandb"
export TORCH_HOME="$CACHE_ROOT/torch"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"

export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8

# =============================================================================
# Configuration
# =============================================================================

DATA_PATH="data/circuit_breakers/cb_training_batches.jsonl"
MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
MAX_SEQ_LENGTH=512
N_SAMPLES=5

echo "=========================================="
echo "CB Pipeline Diagnostic - Lite"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo "Data: $DATA_PATH"
echo "Model: $MODEL_NAME"
echo "Device: CPU (lightweight mode)"
echo "Max sequence length: $MAX_SEQ_LENGTH"
echo "Sample count: $N_SAMPLES"
echo "=========================================="

# =============================================================================
# Run Diagnostic (CPU mode - data checks only)
# =============================================================================

python scripts/diagnose_cb_pipeline.py \
  --data-path "$DATA_PATH" \
  --model-name "$MODEL_NAME" \
  --device cpu \
  --max-seq-length "$MAX_SEQ_LENGTH" \
  --n-samples "$N_SAMPLES"

echo ""
echo "=========================================="
echo "Diagnostic completed!"
echo "Date: $(date)"
echo "=========================================="
