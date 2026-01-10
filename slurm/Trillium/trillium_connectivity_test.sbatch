#!/bin/bash
#SBATCH --job-name=connectivity_test
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/project/def-zhijing/memoozd/harmful-agents-meta-dataset/logs/%x_%j.out
#SBATCH --error=/project/def-zhijing/memoozd/harmful-agents-meta-dataset/logs/%x_%j.err
#SBATCH --account=def-zhijing

# =============================================================================
# Trillium Connectivity Test (CPU Only)
# =============================================================================
#
# Tests:
# 1. Internet connectivity (ping, curl)
# 2. HuggingFace Hub access (API and model download test)
#
# Submit from $SCRATCH:
#   cd /scratch/memoozd/harmful-agents-meta-dataset
#   sbatch slurm/Trillium/trillium_connectivity_test.sbatch
#
# =============================================================================

set -euo pipefail

PROJECT_DIR="/project/def-zhijing/memoozd"
SCRATCH_DIR="/scratch/memoozd"
REPO_DIR="$PROJECT_DIR/harmful-agents-meta-dataset"
VENV_DIR="$PROJECT_DIR/.venvs/cb_env"

cd "$REPO_DIR"
mkdir -p "$PROJECT_DIR/harmful-agents-meta-dataset/logs"

# Load modules
module --force purge || true
module load StdEnv/2023
module load python/3.11.5

echo "========================================"
echo "Trillium Connectivity Test"
echo "========================================"
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURM_NODELIST"
echo "Date:     $(date)"
echo "Mode:     CPU only (no GPU)"
echo "========================================"

# =============================================================================
# Test 1: Basic Internet Connectivity
# =============================================================================
echo ""
echo "TEST 1: Basic Internet Connectivity"
echo "========================================"

# Test DNS resolution
echo -n "DNS Resolution (google.com): "
if host google.com > /dev/null 2>&1; then
    echo "✓ SUCCESS"
else
    echo "✗ FAILED"
fi

# Test ping (may be blocked by firewall)
echo -n "Ping (8.8.8.8): "
if ping -c 1 -W 2 8.8.8.8 > /dev/null 2>&1; then
    echo "✓ SUCCESS"
else
    echo "✗ FAILED (may be blocked by firewall)"
fi

# Test HTTP/HTTPS
echo -n "HTTPS (google.com): "
if curl -s -o /dev/null -w "%{http_code}" --max-time 10 https://www.google.com | grep -q "200"; then
    echo "✓ SUCCESS"
else
    echo "✗ FAILED"
fi

# =============================================================================
# Test 2: HuggingFace Hub Access
# =============================================================================
echo ""
echo "TEST 2: HuggingFace Hub Access"
echo "========================================"

# Test HF website
echo -n "HuggingFace website: "
if curl -s -o /dev/null -w "%{http_code}" --max-time 10 https://huggingface.co | grep -q "200"; then
    echo "✓ SUCCESS"
else
    echo "✗ FAILED"
fi

# Test HF API
echo -n "HuggingFace API: "
if curl -s --max-time 10 https://huggingface.co/api/models/gpt2 | grep -q "modelId"; then
    echo "✓ SUCCESS"
else
    echo "✗ FAILED"
fi

# =============================================================================
# Test 3: HuggingFace Python Access
# =============================================================================
echo ""
echo "TEST 3: HuggingFace Python Access"
echo "========================================"

if [[ ! -d "$VENV_DIR" ]]; then
    echo "WARNING: venv not found at $VENV_DIR"
    echo "Skipping Python tests"
else
    source "$VENV_DIR/bin/activate"

    echo "Python: $(python -V)"
    echo ""

    # Test huggingface_hub installation
    echo -n "huggingface_hub installed: "
    if python -c "import huggingface_hub; print(huggingface_hub.__version__)" > /dev/null 2>&1; then
        VERSION=$(python -c "import huggingface_hub; print(huggingface_hub.__version__)")
        echo "✓ SUCCESS (v$VERSION)"
    else
        echo "✗ FAILED"
    fi

    # Test model info fetch (lightweight API call)
    echo -n "Fetch model info (gpt2): "
    if python -c "
from huggingface_hub import model_info
info = model_info('gpt2', timeout=10)
print('✓ SUCCESS (model found)')
" 2>/dev/null; then
        :  # Success message printed by Python
    else
        echo "✗ FAILED"
    fi

    # Test small model download (tokenizer only, ~500KB)
    echo "Testing small download (gpt2 tokenizer):"
    CACHE_DIR="$SCRATCH_DIR/connectivity_test_cache"
    mkdir -p "$CACHE_DIR"
    export HF_HOME="$CACHE_DIR"

    if timeout 60 python -c "
from transformers import AutoTokenizer
import os
print('  Downloading...')
tokenizer = AutoTokenizer.from_pretrained('gpt2')
print('  ✓ SUCCESS - Download completed')
print(f'  Vocab size: {tokenizer.vocab_size}')
" 2>/dev/null; then
        :  # Success message printed by Python
    else
        echo "  ✗ FAILED - Could not download"
    fi

    # Cleanup test cache
    rm -rf "$CACHE_DIR"
fi

# =============================================================================
# Test 4: Other Common Endpoints
# =============================================================================
echo ""
echo "TEST 4: Other Common Endpoints"
echo "========================================"

echo -n "PyPI: "
if curl -s -o /dev/null -w "%{http_code}" --max-time 10 https://pypi.org | grep -q "200"; then
    echo "✓ SUCCESS"
else
    echo "✗ FAILED"
fi

echo -n "GitHub: "
if curl -s -o /dev/null -w "%{http_code}" --max-time 10 https://github.com | grep -q "200"; then
    echo "✓ SUCCESS"
else
    echo "✗ FAILED"
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "========================================"
echo "Connectivity Test Complete!"
echo "========================================"
echo "Date: $(date)"
echo ""
echo "If all tests passed, Trillium can:"
echo "  - Access the internet"
echo "  - Download from HuggingFace Hub"
echo "  - Use transformers/huggingface_hub libraries"
echo ""
echo "If tests failed, check:"
echo "  - Compute node firewall rules"
echo "  - Proxy settings (HTTP_PROXY, HTTPS_PROXY)"
echo "  - Network configuration"
echo "========================================"
