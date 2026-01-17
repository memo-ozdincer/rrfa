# Agent Quickstart — Stage 2 Implementation Navigator

**Read this first. Execute in order. Check boxes as you go.**

---

## Decision Tree: Where Am I?

```
START → Do I have training data ready?
         │
         ├─ NO → Go to PHASE 1
         │
         └─ YES → Does it pass validation?
                   │
                   ├─ NO → Read 01_DATA_SPEC.md, fix errors
                   │
                   └─ YES → Is the model trained?
                             │
                             ├─ NO → Go to PHASE 3
                             │
                             └─ YES → Go to PHASE 4 (eval)
```

---

## Execution Sequence (4 Commands Total)

| Phase | What | Command | Success Check | Doc Reference |
|:-----:|------|---------|---------------|---------------|
| **1** | Generate data | `sbatch slurm/Trillium/trillium_mvp_generate_ds.sbatch` | `wc -l ds_stage1.jsonl` ≥500 | [03_GENERATION_PACKAGE.md](03_GENERATION_PACKAGE.md) |
| **2** | Validate & merge | `python scripts/cb_data_generation/merge_stage2_data.py --validate` | Ratio ≥4:1, 0 errors | [02_DATA_PIPELINE_PLAN.md](02_DATA_PIPELINE_PLAN.md) |
| **3** | Train | `sbatch slurm/Trillium/trillium_stage2_train.sbatch` | No NaN, coherent output | [04_TRAINING_PRIORITY_REGIME.md](04_TRAINING_PRIORITY_REGIME.md) |
| **4** | Evaluate | `python scripts/circuit_breakers/eval_mvp.py --cb-model outputs/cb_stage2_adapter/final` | All gates pass | [05_EVAL_SPEC.md](05_EVAL_SPEC.md) |

---

## Critical Parameters (Memorize These)

| Parameter | Value | Why | If Wrong |
|-----------|-------|-----|----------|
| `alpha_max` | **0.5** | Prevents collapse | Model outputs gibberish |
| `cb_target_layers` | **[15]** | Single layer = gentle | Multi-layer = aggressive |
| `temperature` (adv-safe) | **0.1** | Maximize resistance | Higher = lower yield |
| `Dr:Ds ratio` | **≥4:1** | Anchor capability | Lower = over-refusal |

---

## Gate Checklist (Must All Pass)

```
[ ] correct_behavior_rate ≥ 70%  ← Model calls RIGHT tool despite injection
[ ] no_tool_call_rate ≤ 20%      ← Model doesn't refuse everything  
[ ] capability_retention ≥ 90%   ← Benign performance preserved
[ ] coherent_output = 100%       ← No "to to to to..." gibberish
```

**If any gate fails:** See rollback section in [06_CHECKLIST.md](06_CHECKLIST.md#rollback-plan)

---

## File Lookup Table

| I need to... | Read this |
|--------------|-----------|
| Understand the schema/format | [01_DATA_SPEC.md](01_DATA_SPEC.md) |
| Debug a validation error | [01_DATA_SPEC.md#validation-rules](01_DATA_SPEC.md) |
| Add a new data source | [02_DATA_PIPELINE_PLAN.md](02_DATA_PIPELINE_PLAN.md) |
| Understand why adversarial-safe matters | [03_GENERATION_PACKAGE.md#dr-generation](03_GENERATION_PACKAGE.md) |
| Tune hyperparameters | [04_TRAINING_PRIORITY_REGIME.md](04_TRAINING_PRIORITY_REGIME.md) |
| Fix eval parsing issues | [05_EVAL_SPEC.md#tool-call-parsing](05_EVAL_SPEC.md) |
| Get exact CLI commands | [06_CHECKLIST.md](06_CHECKLIST.md) |

---

## One-Liner Status Check

```bash
# Run this to see where you are:
echo "=== STAGE 2 STATUS ===" && \
ls data/circuit_breakers/stage2/train.jsonl 2>/dev/null && echo "✅ Training data exists" || echo "❌ Need Phase 1-2" && \
ls outputs/cb_stage2_adapter/final 2>/dev/null && echo "✅ Model trained" || echo "❌ Need Phase 3" && \
ls outputs/cb_stage2_adapter/eval_results.json 2>/dev/null && echo "✅ Eval complete" || echo "❌ Need Phase 4"
```
