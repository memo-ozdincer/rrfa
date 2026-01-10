#!/bin/bash

# =============================================================================
# Rebuild Python venv on Trillium (RECOMMENDED APPROACH)
# =============================================================================
#
# This script rebuilds the Python virtual environment on Trillium.
# This is the SAFER and RECOMMENDED approach compared to transferring
# the venv from Killarney, as it ensures all dependencies are compiled
# for Trillium's architecture and CUDA version.
#
# Run this on the Trillium login node:
#   ssh trillium.alliancecan.ca
#   cd /project/def-zhijing/memoozd/harmful-agents-meta-dataset
#   bash scripts/rebuild_venv_on_trillium.sh
#
# =============================================================================

set -euo pipefail

PROJECT_DIR="/project/def-zhijing/memoozd"
VENV_DIR="$PROJECT_DIR/.venvs/cb_env"
REPO_DIR="$PROJECT_DIR/harmful-agents-meta-dataset"

echo "========================================"
echo "Rebuild venv on Trillium"
echo "========================================"
echo ""
echo "This will create a fresh Python virtual environment at:"
echo "  $VENV_DIR"
echo ""
echo "Press Ctrl+C to cancel, or Enter to continue..."
read

# =============================================================================
# Step 1: Load Modules
# =============================================================================
echo ""
echo "Step 1: Loading modules..."
module --force purge || true
module load StdEnv/2023
module load cuda/12.6
module load python/3.11.5

echo "Python version: $(python -V)"

# =============================================================================
# Step 2: Install uv (if not already installed)
# =============================================================================
echo ""
echo "Step 2: Installing/updating uv..."

# Install uv to user directory if not available
if ! command -v uv &> /dev/null; then
    echo "  uv not found, installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
else
    echo "  uv already installed: $(uv --version)"
fi

# =============================================================================
# Step 3: Create venv directory
# =============================================================================
echo ""
echo "Step 3: Creating venv directory..."

mkdir -p "$(dirname $VENV_DIR)"

if [[ -d "$VENV_DIR" ]]; then
    echo "  WARNING: venv already exists at $VENV_DIR"
    read -p "  Remove and recreate? (yes/no): " CONFIRM
    if [[ "$CONFIRM" == "yes" ]]; then
        rm -rf "$VENV_DIR"
        echo "  Removed old venv"
    else
        echo "  Keeping existing venv, exiting."
        exit 0
    fi
fi

# =============================================================================
# Step 4: Create virtual environment with uv
# =============================================================================
echo ""
echo "Step 4: Creating virtual environment with uv..."

uv venv "$VENV_DIR" --python python3.11

echo "  Virtual environment created!"

# =============================================================================
# Step 5: Activate venv
# =============================================================================
echo ""
echo "Step 5: Activating virtual environment..."

source "$VENV_DIR/bin/activate"

echo "  Python: $(which python)"
echo "  Pip: $(which pip)"

# =============================================================================
# Step 6: Install PyTorch with CUDA support
# =============================================================================
echo ""
echo "Step 6: Installing PyTorch with CUDA 12.6 support..."

# Install PyTorch with CUDA 12.6 support
# Using uv pip for faster installation
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

echo ""
echo "  Verifying PyTorch installation..."
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')"

# =============================================================================
# Step 7: Install remaining dependencies
# =============================================================================
echo ""
echo "Step 7: Installing remaining dependencies from requirements.txt..."

cd "$REPO_DIR"

# Use uv pip for faster installation
uv pip install -r requirements.txt

echo "  All dependencies installed!"

# =============================================================================
# Step 8: Install vLLM (for data generation)
# =============================================================================
echo ""
echo "Step 8: Installing vLLM..."

uv pip install vllm

echo "  vLLM installed!"

# =============================================================================
# Step 9: Verify installation
# =============================================================================
echo ""
echo "Step 9: Verifying installation..."
echo ""

python - << 'PY'
import sys

def check_package(name):
    try:
        mod = __import__(name)
        version = getattr(mod, '__version__', 'unknown')
        print(f"  ✓ {name:20s} {version}")
        return True
    except ImportError as e:
        print(f"  ✗ {name:20s} MISSING: {e}")
        return False

packages = [
    'torch',
    'torchvision',
    'transformers',
    'peft',
    'accelerate',
    'datasets',
    'vllm',
    'wandb',
    'bitsandbytes',
]

print("Checking installed packages:")
all_ok = all(check_package(pkg) for pkg in packages)

if all_ok:
    print("\n✓ All packages installed successfully!")

    # Check CUDA
    import torch
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.version.cuda}")
        print(f"✓ GPU count: {torch.cuda.device_count()}")
    else:
        print("⚠ WARNING: CUDA not available!")
        print("  This is expected on login nodes.")
        print("  GPUs will be available in compute jobs.")
else:
    print("\n✗ Some packages failed to install!")
    sys.exit(1)
PY

# =============================================================================
# Done!
# =============================================================================
echo ""
echo "========================================"
echo "Virtual environment setup complete!"
echo "========================================"
echo ""
echo "Virtual environment location:"
echo "  $VENV_DIR"
echo ""
echo "To activate the venv:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "To test on a GPU node, run the debug script:"
echo "  cd /scratch/memoozd/harmful-agents-meta-dataset"
echo "  sbatch slurm/Trillium/trillium_debug.sbatch"
echo ""
