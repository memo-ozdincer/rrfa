#!/bin/bash
# =============================================================================
# Resume Circuit Breaker MVP Pipeline
# =============================================================================
# Continues pipeline from where it left off
# Use this when Ds/Dr generation completed but downstream jobs didn't run
# =============================================================================

set -euo pipefail

echo "========================================"
echo "Resume Circuit Breaker MVP Pipeline"
echo "========================================"
echo ""

# Verify Ds/Dr exist
DATA_DIR="/scratch/memoozd/cb_mvp_data"
if [[ ! -f "$DATA_DIR/ds_stage1.jsonl" ]]; then
    echo "ERROR: $DATA_DIR/ds_stage1.jsonl not found"
    exit 1
fi

if [[ ! -f "$DATA_DIR/dr_stage1.jsonl" ]]; then
    echo "ERROR: $DATA_DIR/dr_stage1.jsonl not found"
    exit 1
fi

echo "✓ Found Ds: $(wc -l < "$DATA_DIR/ds_stage1.jsonl") samples"
echo "✓ Found Dr: $(wc -l < "$DATA_DIR/dr_stage1.jsonl") samples"
echo ""

# 1. Validation (optional but recommended)
echo "Submitting validation..."
VALIDATE_JOB=$(sbatch --parsable slurm/Trillium/trillium_mvp_validate.sbatch)
echo "  Job ID: $VALIDATE_JOB"

# 2. Create eval set (required for training)
echo "Submitting eval set creation..."
EVAL_SET_JOB=$(sbatch --parsable slurm/Trillium/trillium_mvp_create_eval.sbatch)
echo "  Job ID: $EVAL_SET_JOB"

# 3. Training (depends on validation and eval set)
echo "Submitting training (depends on $VALIDATE_JOB and $EVAL_SET_JOB)..."
TRAIN_JOB=$(sbatch --parsable --dependency=afterok:$VALIDATE_JOB:$EVAL_SET_JOB slurm/Trillium/trillium_mvp_train.sbatch)
echo "  Job ID: $TRAIN_JOB"

# 4. Sanity check (depends on training)
echo "Submitting sanity check (depends on $TRAIN_JOB)..."
SANITY_JOB=$(sbatch --parsable --dependency=afterok:$TRAIN_JOB slurm/Trillium/trillium_mvp_sanity_check.sbatch)
echo "  Job ID: $SANITY_JOB"

# 5. Evaluation (depends on training)
echo "Submitting evaluation (depends on $TRAIN_JOB)..."
EVAL_JOB=$(sbatch --parsable --dependency=afterok:$TRAIN_JOB slurm/Trillium/trillium_mvp_eval.sbatch)
echo "  Job ID: $EVAL_JOB"

echo ""
echo "========================================"
echo "Pipeline resumed!"
echo "========================================"
echo "Job chain:"
echo "  1. Validation:    $VALIDATE_JOB"
echo "  2. Eval set:      $EVAL_SET_JOB"
echo "  3. Training:      $TRAIN_JOB (after validation + eval set)"
echo "  4. Sanity check:  $SANITY_JOB (after training)"
echo "  5. Evaluation:    $EVAL_JOB (after training)"
echo ""
echo "Monitor: squeue -u \$USER"
echo "Cancel:  scancel $VALIDATE_JOB $EVAL_SET_JOB $TRAIN_JOB $SANITY_JOB $EVAL_JOB"
echo "========================================"
