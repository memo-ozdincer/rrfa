#!/bin/bash

# =============================================================================
# Transfer venv from Killarney to Trillium
# =============================================================================
#
# WARNING: Transferring Python virtual environments between clusters is risky!
#
# Potential issues:
# - Different CPU architectures (AMD Zen 5 on Trillium vs Zen 4 on Killarney)
# - Different CUDA versions or drivers
# - Binary wheels compiled for different systems may not work
# - System libraries may be incompatible
#
# RECOMMENDATION: Rebuild the venv on Trillium instead (see rebuild script)
#
# If you still want to transfer (at your own risk), run:
#   bash scripts/transfer_venv_to_trillium.sh
#
# =============================================================================

set -euo pipefail

# Source and destination paths
KILLARNEY_VENV="/project/6105522/memoozd/.venvs/cb_env"
TRILLIUM_VENV_DIR="/project/def-zhijing/memoozd/.venvs"
TRILLIUM_HOST="trillium.alliancecan.ca"

echo "========================================"
echo "Transfer venv: Killarney → Trillium"
echo "========================================"
echo ""
echo "WARNING: This may not work due to:"
echo "  - Different CPU architectures"
echo "  - Different CUDA versions"
echo "  - Binary incompatibilities"
echo ""
echo "Source: $KILLARNEY_VENV"
echo "Destination: $TRILLIUM_HOST:$TRILLIUM_VENV_DIR/cb_env"
echo ""

read -p "Are you sure you want to proceed? (yes/no): " CONFIRM

if [[ "$CONFIRM" != "yes" ]]; then
  echo "Transfer cancelled."
  exit 0
fi

echo ""
echo "Step 1: Creating destination directory on Trillium..."
ssh "$TRILLIUM_HOST" "mkdir -p $TRILLIUM_VENV_DIR"

echo ""
echo "Step 2: Compressing venv on Killarney..."
TEMP_ARCHIVE="/tmp/cb_env_$(date +%s).tar.gz"
tar -czf "$TEMP_ARCHIVE" -C "$(dirname $KILLARNEY_VENV)" "$(basename $KILLARNEY_VENV)"

echo ""
echo "Step 3: Transferring to Trillium..."
echo "  Archive size: $(du -h $TEMP_ARCHIVE | cut -f1)"
scp "$TEMP_ARCHIVE" "$TRILLIUM_HOST:$TRILLIUM_VENV_DIR/"

echo ""
echo "Step 4: Extracting on Trillium..."
ssh "$TRILLIUM_HOST" "cd $TRILLIUM_VENV_DIR && tar -xzf $(basename $TEMP_ARCHIVE) && rm $(basename $TEMP_ARCHIVE)"

echo ""
echo "Step 5: Cleaning up local archive..."
rm "$TEMP_ARCHIVE"

echo ""
echo "========================================"
echo "Transfer complete!"
echo "========================================"
echo ""
echo "Now test the venv on Trillium:"
echo "  ssh $TRILLIUM_HOST"
echo "  source $TRILLIUM_VENV_DIR/cb_env/bin/activate"
echo "  python -c 'import torch; print(torch.__version__)'"
echo ""
echo "If you encounter errors, rebuild the venv instead:"
echo "  bash scripts/rebuild_venv_on_trillium.sh"
echo ""
