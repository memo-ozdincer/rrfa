#!/bin/bash
#SBATCH -A aip-rgrosse
#SBATCH -D /project/6105522/memoozd/harmful-agents-meta-dataset
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --job-name=eval_cb_quick
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# =============================================================================
# QUICK Circuit Breakers Evaluation - 20 samples only
# =============================================================================

set -euo pipefail

mkdir -p logs

PROJECT_DIR="/project/6105522/memoozd"
SCRATCH_DIR="$HOME/scratch"
REPO_DIR="/project/6105522/memoozd/harmful-agents-meta-dataset"
VENV_DIR="/project/6105522/memoozd/.venvs/cb_env"

cd "$REPO_DIR"

module --force purge || true
module load StdEnv/2023
module load cuda/12.6
module load python/3.11.5

source "$VENV_DIR/bin/activate"

echo "Python: $(python -V)"

# Environment
CACHE_ROOT="$PROJECT_DIR/cb_cache"
export HF_HOME="$CACHE_ROOT/hf"
export WANDB_MODE=disabled  # Skip W&B for quick test

# Find checkpoint
CB_CHECKPOINT="${CB_CHECKPOINT:-}"
if [[ -z "$CB_CHECKPOINT" ]]; then
  LATEST_RUN=$(ls -td $SCRATCH_DIR/cb_runs/*/outputs/cb_llama31_8b_instruct_v2 2>/dev/null | head -1)
  if [[ -z "$LATEST_RUN" ]]; then
    LATEST_RUN=$(ls -td $SCRATCH_DIR/cb_runs/*/outputs/cb_llama31_8b_instruct 2>/dev/null | head -1)
  fi
  CB_CHECKPOINT="$LATEST_RUN/final"
fi

EVAL_OUTPUT_DIR="$SCRATCH_DIR/cb_eval/quick_$SLURM_JOB_ID"
mkdir -p "$EVAL_OUTPUT_DIR"

echo "========================================"
echo "QUICK Eval - 20 samples each"
echo "========================================"
echo "CB Checkpoint: $CB_CHECKPOINT"
echo "========================================"

# Generate minimal eval data
EVAL_DATA_DIR="$EVAL_OUTPUT_DIR/data"
mkdir -p "$EVAL_DATA_DIR"

python scripts/prepare_eval_data.py \
  --training-data data/circuit_breakers/cb_training_batches.jsonl \
  --output-dir "$EVAL_DATA_DIR" \
  --n-harmful 20 \
  --n-benign 20 \
  --seed 42

echo "Eval data: $(wc -l < $EVAL_DATA_DIR/harmful_eval.jsonl) harmful, $(wc -l < $EVAL_DATA_DIR/benign_eval.jsonl) benign"

# Baseline
echo ""
echo "=== Baseline Model ==="
python scripts/circuit_breakers/eval.py \
  --base-model meta-llama/Llama-3.1-8B-Instruct \
  --harmful-data "$EVAL_DATA_DIR/harmful_eval.jsonl" \
  --benign-data "$EVAL_DATA_DIR/benign_eval.jsonl" \
  --output "$EVAL_OUTPUT_DIR/baseline_results.json" \
  --max-samples 20 \
  --wandb-mode disabled

echo ""
echo "=== CB Model ==="
python scripts/circuit_breakers/eval.py \
  --base-model meta-llama/Llama-3.1-8B-Instruct \
  --adapter-path "$CB_CHECKPOINT" \
  --harmful-data "$EVAL_DATA_DIR/harmful_eval.jsonl" \
  --benign-data "$EVAL_DATA_DIR/benign_eval.jsonl" \
  --output "$EVAL_OUTPUT_DIR/cb_results.json" \
  --max-samples 20 \
  --wandb-mode disabled

# Compare
echo ""
echo "=== Comparison ==="
python scripts/analyze_cb_results.py \
  --baseline "$EVAL_OUTPUT_DIR/baseline_results.json" \
  --cb-model "$EVAL_OUTPUT_DIR/cb_results.json" \
  --output "$EVAL_OUTPUT_DIR/analysis.json"

echo ""
echo "========================================"
echo "QUICK EVAL COMPLETE"
echo "Results: $EVAL_OUTPUT_DIR"
cat "$EVAL_OUTPUT_DIR/analysis.json"
echo "========================================"
