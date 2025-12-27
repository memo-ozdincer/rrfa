#!/bin/bash
# =============================================================================
# Circuit Breakers Trillium Setup Script
# =============================================================================
#
# Trillium-specific setup that:
# - Installs everything to $SCRATCH (writable from compute nodes)
# - Uses StdEnv/2023 modules (Trillium best practice)
# - Sets caches to $SCRATCH to avoid $HOME bottlenecks
#
# IMPORTANT: Run this on the GPU login node or in a GPU allocation,
# as CUDA modules aren't available on CPU-only Trillium nodes.
#
# Usage:
#   ssh memoozd@trillium-gpu.scinet.utoronto.ca
#   cd $SCRATCH
#   git clone <repo> harmful-agents-meta-dataset
#   cd harmful-agents-meta-dataset
#   bash scripts/hpc_setup_trillium.sh
#
# =============================================================================

set -euo pipefail

echo "=============================================="
echo "  CB Trillium Setup (Alliance/SciNet)"
echo "=============================================="

# Always build env on scratch (fast + writable from compute jobs)
: "${ENV_ROOT:=$SCRATCH/.venvs}"
: "${ENV_NAME:=cb_env}"
VENV_DIR="$ENV_ROOT/$ENV_NAME"

mkdir -p "$ENV_ROOT"

echo "Scratch:   $SCRATCH"
echo "Env root:  $ENV_ROOT"
echo "Venv dir:  $VENV_DIR"

# Use Trillium modules (recommended; be explicit)
source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module purge
module load StdEnv/2023
module load cuda/12.6
module load python/3.11.5

echo "Modules loaded:"
module list

# uv install (user-space)
if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
echo "uv: $(uv --version)"

# Create venv
if [[ ! -d "$VENV_DIR" ]]; then
  echo "Creating venv at $VENV_DIR ..."
  uv venv "$VENV_DIR" --python 3.11
fi
source "$VENV_DIR/bin/activate"
python -V
which python

# Keep all caches on scratch (important on Trillium)
: "${CACHE_ROOT:=$SCRATCH/cb_cache}"
mkdir -p "$CACHE_ROOT"/{hf,wandb,torch,xdg}
export HF_HOME="$CACHE_ROOT/hf"
export TRANSFORMERS_CACHE="$CACHE_ROOT/hf/transformers"
export HF_DATASETS_CACHE="$CACHE_ROOT/hf/datasets"
export WANDB_DIR="$CACHE_ROOT/wandb"
export TORCH_HOME="$CACHE_ROOT/torch"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"

echo "Cache root: $CACHE_ROOT"

uv pip install --upgrade pip setuptools wheel

# PyTorch (wheel build; should work with cuda module loaded)
echo "Installing PyTorch..."
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

uv pip install \
  "transformers>=4.45.0" \
  "peft>=0.13.0" \
  "accelerate>=1.0.0" \
  "bitsandbytes>=0.44.0" \
  "deepspeed>=0.15.0" \
  "sentencepiece>=0.2.0" \
  "protobuf>=4.25.0" \
  "wandb>=0.18.0" \
  "tensorboard>=2.17.0" \
  "pandas>=2.0.0" \
  "numpy>=1.24.0" \
  "pyarrow>=15.0.0" \
  "datasets>=2.19.0" \
  "huggingface-hub>=0.25.0" \
  "tqdm>=4.66.0"

# Optional
echo "Installing flash-attn (optional, may take time)..."
uv pip install flash-attn --no-build-isolation || true

python - << 'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("num gpus:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"  gpu {i}:", torch.cuda.get_device_name(i))
PY

echo "=============================================="
echo "✅ Trillium setup complete"
echo "Activate with:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "Next: Submit jobs from $SCRATCH using:"
echo "  cd $SCRATCH/harmful-agents-meta-dataset"
echo "  sbatch slurm/trillium_cb_llama4_4xh100.sbatch"
echo "=============================================="
