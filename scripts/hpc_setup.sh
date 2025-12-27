#!/bin/bash
# =============================================================================
# Circuit Breakers HPC Setup Script (Generic)
# =============================================================================
#
# Works on: CSLab ML cluster, general HPC systems
# For Trillium (Alliance): use scripts/hpc_setup_trillium.sh
#
# Usage:
#   chmod +x scripts/hpc_setup.sh
#   ./scripts/hpc_setup.sh
#
# =============================================================================

set -euo pipefail

echo "=============================================="
echo "  Circuit Breakers HPC Environment Setup"
echo "=============================================="

# ---- User-tunable defaults ----
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

# Prefer fast scratch locations if available
DEFAULT_ENV_ROOT=""
if [[ -n "${SCRATCH:-}" ]]; then
  DEFAULT_ENV_ROOT="$SCRATCH/.venvs"
elif [[ -d "/mfs1/u/$USER" ]]; then
  DEFAULT_ENV_ROOT="/mfs1/u/$USER/.venvs"
else
  DEFAULT_ENV_ROOT="$PROJECT_DIR"
fi

ENV_ROOT="${ENV_ROOT:-$DEFAULT_ENV_ROOT}"
ENV_NAME="${ENV_NAME:-cb_env}"
VENV_DIR="${VENV_DIR:-$ENV_ROOT/$ENV_NAME}"

mkdir -p "$ENV_ROOT"

echo "Project:   $PROJECT_DIR"
echo "Env root:  $ENV_ROOT"
echo "Venv dir:  $VENV_DIR"
echo "Python:    $PYTHON_VERSION"

# ---- 0) Modules (Alliance-style, safe no-op elsewhere) ----
# Alliance clusters commonly use the module system for Python.
if [[ -f /cvmfs/soft.computecanada.ca/config/profile/bash.sh ]]; then
  source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
  module purge || true
  module load python/"$PYTHON_VERSION" || module load python || true
fi

# ---- 1) Install uv (user-space) ----
if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
echo "uv: $(uv --version)"

# ---- 2) Create venv ----
if [[ ! -d "$VENV_DIR" ]]; then
  echo "Creating venv at $VENV_DIR ..."
  uv venv "$VENV_DIR" --python "$PYTHON_VERSION"
fi

# Activate
source "$VENV_DIR/bin/activate"
python -V
which python

# ---- 3) Put caches on scratch (avoid $HOME bottlenecks) ----
CACHE_ROOT="${CACHE_ROOT:-${SCRATCH:-/tmp}/cb_cache/$USER}"
mkdir -p "$CACHE_ROOT"/{hf,wandb,torch,xdg}

export HF_HOME="$CACHE_ROOT/hf"
export TRANSFORMERS_CACHE="$CACHE_ROOT/hf/transformers"
export HF_DATASETS_CACHE="$CACHE_ROOT/hf/datasets"
export WANDB_DIR="$CACHE_ROOT/wandb"
export TORCH_HOME="$CACHE_ROOT/torch"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"

echo "Cache root: $CACHE_ROOT"

# ---- 4) Install PyTorch ----
uv pip install --upgrade pip setuptools wheel

echo "Installing PyTorch..."
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

python -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available())"

# ---- 5) Core deps ----
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

# ---- 6) Flash-Attn (optional) ----
echo "Optional: flash-attn (may fail on some HPC setups)"
uv pip install flash-attn --no-build-isolation || true

# ---- 7) Verify ----
python - << 'PY'
import torch, transformers, peft, accelerate
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("num gpus:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print("gpu", i, torch.cuda.get_device_name(i))
print("transformers:", transformers.__version__)
print("peft:", peft.__version__)
print("accelerate:", accelerate.__version__)
PY

echo "=============================================="
echo "✅ Setup complete"
echo "Activate with:"
echo "  source $VENV_DIR/bin/activate"
echo "=============================================="
