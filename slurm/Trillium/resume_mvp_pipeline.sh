#!/bin/bash
# =============================================================================
# Resume Circuit Breaker MVP Pipeline from Validation Step
# =============================================================================
# Use this when Ds/Dr generation completed but pipeline didn't continue
# =============================================================================

set -euo pipefail

echo "========================================"
echo "Resuming Circuit Breaker MVP Pipeline"
echo "========================================"
echo "Starting from validation step..."
echo ""

# 1. Validate - no dependencies
echo "Submitting validation..."
VALIDATE_JOB=$(sbatch --parsable slurm/Trillium/trillium_mvp_validate.sbatch)
echo "  Job ID: $VALIDATE_JOB"

# 2. Create Eval Set - no dependencies (can run parallel with validate)
echo "Submitting eval set creation..."
EVAL_SET_JOB=$(sbatch --parsable slurm/Trillium/trillium_mvp_create_eval.sbatch)
echo "  Job ID: $EVAL_SET_JOB"

# 3. Train - depends on both validate and eval set
echo "Submitting training (depends on $VALIDATE_JOB and $EVAL_SET_JOB)..."
TRAIN_JOB=$(sbatch --parsable --dependency=afterok:$VALIDATE_JOB:$EVAL_SET_JOB slurm/Trillium/trillium_mvp_train.sbatch)
echo "  Job ID: $TRAIN_JOB"

# 4. Sanity Check - depends on training
echo "Submitting sanity check (depends on $TRAIN_JOB)..."
SANITY_JOB=$(sbatch --parsable --dependency=afterok:$TRAIN_JOB slurm/Trillium/trillium_mvp_sanity_check.sbatch)
echo "  Job ID: $SANITY_JOB"

# 5. Full Evaluation - depends on training (can run parallel with sanity)
echo "Submitting evaluation (depends on $TRAIN_JOB)..."
EVAL_JOB=$(sbatch --parsable --dependency=afterok:$TRAIN_JOB slurm/Trillium/trillium_mvp_eval.sbatch)
echo "  Job ID: $EVAL_JOB"

echo ""
echo "========================================"
echo "Pipeline resumed!"
echo "========================================"
echo "Job dependency chain:"
echo "  1. Validation:    $VALIDATE_JOB"
echo "  2. Eval set:      $EVAL_SET_JOB"
echo "  3. Training:      $TRAIN_JOB (after $VALIDATE_JOB, $EVAL_SET_JOB)"
echo "  4. Sanity check:  $SANITY_JOB (after $TRAIN_JOB)"
echo "  5. Evaluation:    $EVAL_JOB (after $TRAIN_JOB)"
echo ""
echo "Monitor with: squeue -u \$USER"
echo "Cancel all with: scancel $VALIDATE_JOB $EVAL_SET_JOB $TRAIN_JOB $SANITY_JOB $EVAL_JOB"
echo "========================================"
