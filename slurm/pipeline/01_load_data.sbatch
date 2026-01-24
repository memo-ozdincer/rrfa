#!/bin/bash
#SBATCH --job-name=etl_a_load
#SBATCH --nodes=1
#SBATCH --gpus-per-node=0
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/memoozd/cb-scratch/logs/%x_%j.out
#SBATCH --error=/scratch/memoozd/cb-scratch/logs/%x_%j.err
#SBATCH --account=def-zhijing

# =============================================================================
# Stage 1: Load Raw Data (ETL_A)
# Converts Tier A raw data -> Tier B trace_v1 format
# CPU-only (no GPU needed)
# =============================================================================
#
# This script:
# 1. Converts raw Fujitsu B4 data to B1 skeleton traces
# 2. Converts raw AgentDojo data to B2 complete traces
# 3. Outputs trace_v1 JSONL files ready for downstream processing
#
# Configuration via environment variables:
#   FUJITSU_B4_PATH - Path to Fujitsu B4 orchestrator attacks JSONL
#   AGENTDOJO_PATH  - Path to AgentDojo JSONL file
#   OUTPUT_DIR      - Directory for output traces (default: $CB_SCRATCH/data/traces)
#   SPLIT           - Split assignment (default: train)
#
# Submit:
#   sbatch slurm/pipeline/01_load_data.sbatch
#
# Or with custom paths:
#   FUJITSU_B4_PATH=/path/to/data.jsonl sbatch slurm/pipeline/01_load_data.sbatch
#
# =============================================================================

set -euo pipefail

PROJECT_DIR="/project/def-zhijing/memoozd"
CB_SCRATCH="/scratch/memoozd/cb-scratch"
REPO_DIR="$PROJECT_DIR/rrfa"
VENV_DIR="$PROJECT_DIR/.venvs/cb_env"

cd "$REPO_DIR"

module --force purge || true
module load StdEnv/2023
module load python/3.11.5

if [[ ! -d "$VENV_DIR" ]]; then
    echo "ERROR: venv not found at $VENV_DIR"
    exit 1
fi

source "$VENV_DIR/bin/activate"

# =============================================================================
# Cache Setup
# =============================================================================
CACHE_DIR="$CB_SCRATCH/cache"
mkdir -p "$CACHE_DIR"/{hf/hub,hf/datasets}
mkdir -p "$CB_SCRATCH/logs"

export HF_HOME="$CACHE_DIR/hf"
export HF_HUB_CACHE="$CACHE_DIR/hf/hub"
export HF_DATASETS_CACHE="$CACHE_DIR/hf/datasets"

echo "========================================"
echo "Stage 1: Load Raw Data (ETL_A)"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Date: $(date)"
echo ""

# =============================================================================
# Configuration
# =============================================================================
# Default paths - override via environment variables
FUJITSU_B4_PATH="${FUJITSU_B4_PATH:-$REPO_DIR/data/fujitsu/orchestrator_attacks_combined_deduplicated.jsonl}"
AGENTDOJO_PATH="${AGENTDOJO_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-$CB_SCRATCH/data/traces}"
SPLIT="${SPLIT:-train}"

mkdir -p "$OUTPUT_DIR"

echo "Configuration:"
echo "  FUJITSU_B4_PATH: $FUJITSU_B4_PATH"
echo "  AGENTDOJO_PATH: $AGENTDOJO_PATH"
echo "  OUTPUT_DIR: $OUTPUT_DIR"
echo "  SPLIT: $SPLIT"
echo ""

# =============================================================================
# Build ETL_A command
# =============================================================================
ETL_A_ARGS=""

# Add Fujitsu B4 if path exists
if [[ -n "$FUJITSU_B4_PATH" && -f "$FUJITSU_B4_PATH" ]]; then
    echo "Found Fujitsu B4 data: $FUJITSU_B4_PATH"
    echo "  Lines: $(wc -l < "$FUJITSU_B4_PATH")"
    ETL_A_ARGS="$ETL_A_ARGS --fujitsu-b4 $FUJITSU_B4_PATH"
    FUJITSU_OUTPUT="$OUTPUT_DIR/fujitsu_b4_skeletons.jsonl"
elif [[ -n "$FUJITSU_B4_PATH" ]]; then
    echo "WARNING: Fujitsu B4 path specified but file not found: $FUJITSU_B4_PATH"
fi

# Add AgentDojo if path exists
if [[ -n "$AGENTDOJO_PATH" && -f "$AGENTDOJO_PATH" ]]; then
    echo "Found AgentDojo data: $AGENTDOJO_PATH"
    echo "  Lines: $(wc -l < "$AGENTDOJO_PATH")"
    ETL_A_ARGS="$ETL_A_ARGS --agentdojo $AGENTDOJO_PATH"
    AGENTDOJO_OUTPUT="$OUTPUT_DIR/agentdojo_complete.jsonl"
elif [[ -n "$AGENTDOJO_PATH" ]]; then
    echo "WARNING: AgentDojo path specified but file not found: $AGENTDOJO_PATH"
fi

if [[ -z "$ETL_A_ARGS" ]]; then
    echo "ERROR: No input data files found!"
    echo "Please set FUJITSU_B4_PATH and/or AGENTDOJO_PATH environment variables."
    exit 1
fi

echo ""

# =============================================================================
# Run ETL_A for each dataset separately (better tracking)
# =============================================================================

# Process Fujitsu B4
if [[ -n "$FUJITSU_B4_PATH" && -f "$FUJITSU_B4_PATH" ]]; then
    echo "========================================"
    echo "Converting Fujitsu B4 -> B1 Skeletons"
    echo "========================================"

    python src/schemas/tools/ETL_A.py \
        --fujitsu-b4 "$FUJITSU_B4_PATH" \
        --output "$OUTPUT_DIR/fujitsu_b4_skeletons.jsonl" \
        --split "$SPLIT"

    echo ""
    echo "Fujitsu B4 conversion complete!"
    echo "  Output: $OUTPUT_DIR/fujitsu_b4_skeletons.jsonl"
    echo "  Lines: $(wc -l < "$OUTPUT_DIR/fujitsu_b4_skeletons.jsonl")"
    echo ""
fi

# Process AgentDojo
if [[ -n "$AGENTDOJO_PATH" && -f "$AGENTDOJO_PATH" ]]; then
    echo "========================================"
    echo "Converting AgentDojo -> B2 Complete"
    echo "========================================"

    python src/schemas/tools/ETL_A.py \
        --agentdojo "$AGENTDOJO_PATH" \
        --output "$OUTPUT_DIR/agentdojo_complete.jsonl" \
        --split "$SPLIT"

    echo ""
    echo "AgentDojo conversion complete!"
    echo "  Output: $OUTPUT_DIR/agentdojo_complete.jsonl"
    echo "  Lines: $(wc -l < "$OUTPUT_DIR/agentdojo_complete.jsonl")"
    echo ""
fi

# =============================================================================
# Summary
# =============================================================================
echo "========================================"
echo "Stage 1 Complete: Load Raw Data"
echo "========================================"
echo ""
echo "Output directory: $OUTPUT_DIR"
echo "Files created:"
ls -la "$OUTPUT_DIR"/*.jsonl 2>/dev/null || echo "  (no files found)"
echo ""
echo "Next step: Run 02_fill_skeletons.sbatch to generate DS/DR completions"
