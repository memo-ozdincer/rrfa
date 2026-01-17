#!/bin/bash
#SBATCH --account=def-zhijing
#SBATCH --job-name=stage2_advsafe
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/memoozd/logs/%x_%j.out
#SBATCH --error=/scratch/memoozd/logs/%x_%j.err

# =============================================================================
# STAGE 2: GENERATE ADVERSARIAL-SAFE SAMPLES (Critical Dr expansion)
# =============================================================================
#
# Generates samples where the model RESISTS injection attacks:
# - Input: B4 adversarial prompts (with malicious injections)
# - Generate: Model responses at low temperature (0.1)
# - Filter: Keep only samples where observed_tool == expected_tool
# - Output: High-value Dr samples for capability retention
#
# Expected yield: ~95% (baseline ASR was only 5.2%)
# Target: 500 samples minimum
#
# Submit from $SCRATCH:
#   cd /scratch/memoozd/harmful-agents-meta-dataset
#   sbatch slurm/Trillium/trillium_stage2_adversarial_safe.sbatch
#
# =============================================================================

set -euo pipefail

PROJECT_DIR="/project/def-zhijing/memoozd"
SCRATCH_DIR="/scratch/memoozd"
REPO_DIR="/project/def-zhijing/memoozd/harmful-agents-meta-dataset"
VENV_DIR="/project/def-zhijing/memoozd/.venvs/cb_env"

cd "$REPO_DIR"

echo "=== Module setup ==="
module --force purge || true
module load StdEnv/2023
module load cuda/12.6
module load python/3.11.5
module list

echo "=== Venv activation ==="
if [[ ! -d "$VENV_DIR" ]]; then
  echo "ERROR: venv not found at $VENV_DIR"
  exit 1
fi
source "$VENV_DIR/bin/activate"
echo "Python: $(python -V)"

echo "=== Environment setup ==="
export OMP_NUM_THREADS=8

RUN_DIR="$SCRATCH_DIR/stage2_adversarial_safe/$SLURM_JOB_ID"
mkdir -p "$RUN_DIR"

CACHE_ROOT="$SCRATCH_DIR/cb_cache"
mkdir -p "$CACHE_ROOT"/{hf,vllm,flashinfer}
export HF_HOME="$CACHE_ROOT/hf"
export VLLM_USAGE_STATS_DIR="$CACHE_ROOT/vllm"
export FLASHINFER_WORKSPACE_DIR="$CACHE_ROOT/flashinfer"
export SCRATCH="$SCRATCH_DIR"

echo ""
echo "GENERATING ADVERSARIAL-SAFE SAMPLES"
echo ""

python scripts/cb_data_generation/generate_adversarial_safe.py \
    --b4-data data/fujitsu/orchestrator_attacks_combined_deduplicated.jsonl \
    --tool-schema configs/tool_schemas/b4_standard_v1.json \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --output data/circuit_breakers/retain/adversarial_safe.jsonl \
    --target-n 500 \
    --temperature 0.1 \
    --batch-size 32
