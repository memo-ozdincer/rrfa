# 02_DATA_PIPELINE_PLAN.md — Pipeline DAG & Execution

**Objective:** Define the complete data pipeline from raw sources to training-ready shards, with explicit dependencies, validators, and stop-the-line checks.

---

## Inputs/Outputs

### Inputs (Raw — Tier A)
| Source | Path | Records |
|--------|------|--------:|
| Fujitsu B4 | `data/fujitsu/orchestrator_attacks_combined_deduplicated.jsonl` | 13,246 |
| AgentDojo | `data/agent_dojo/agentdojo-*.jsonl` | 3,315 |
| TAU2 | `data/tau2_repo/data/tau2/domains/*/tasks.json` | ~2,458 |
| UltraChat | HuggingFace `HuggingFaceH4/ultrachat_200k` | (download) |
| XSTest | GitHub CSV | (download) |

### Outputs (Training — Tier C)
| Artifact | Path | Expected Size |
|----------|------|---------------|
| Harmful (Ds) | `data/circuit_breakers/stage2/harmful.jsonl` | ~1,000 samples |
| Retain (Dr) | `data/circuit_breakers/stage2/retain.jsonl` | ~4,500 samples |
| Merged training | `data/circuit_breakers/stage2/train.jsonl` | ~5,500 samples |
| Eval holdout | `data/circuit_breakers/stage2/eval.jsonl` | ~500 samples |

---

## Pipeline DAG

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 1: DATA GENERATION                                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  [Raw B4] ──┬──▶ [generate_ds_mvp.py] ──▶ [ds_stage1.jsonl]                 │
│             │         (GPU, abliterated model)                               │
│             │                                                                │
│             └──▶ [generate_adversarial_safe.py] ──▶ [adversarial_safe.jsonl]│
│                       (GPU, instruct model, temp=0.1)    ▲ CRITICAL         │
│                                                                              │
│  [Raw AgentDojo] ──▶ [ingest_agentdojo_splits.py] ──┬▶ [agentdojo_failures]│
│                                                     └▶ [agentdojo_resisted]│
│                                                                              │
│  [Raw TAU2] ──▶ [ingest_tau2_traces.py] ──▶ [tau2_traces.jsonl]            │
│                                                                              │
│  [HuggingFace] ──▶ [ingest_ultrachat.py] ──▶ [ultrachat_subset.jsonl]      │
│                                                                              │
│  [GitHub] ──▶ [ingest_xstest.py] ──▶ [xstest_borderline.jsonl]             │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│ PHASE 2: VALIDATION & MERGE                                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  [All curated JSONL] ──▶ [validate_format.py --strict] ──▶ PASS/FAIL       │
│                                    │                                         │
│                                    ▼ (if PASS)                               │
│                          [merge_stage2_data.py]                              │
│                                    │                                         │
│                     ┌──────────────┼──────────────┐                          │
│                     ▼              ▼              ▼                          │
│              [train.jsonl]  [eval.jsonl]  [stats.json]                      │
│                                    │                                         │
│                                    ▼                                         │
│                       [preflight_check.py] ──▶ PASS/FAIL                    │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│ PHASE 3: TRAINING                                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  [train.jsonl] + [Stage 2 Config] ──▶ [train_circuit_breaker.py]           │
│                                              │                               │
│                                              ▼                               │
│                                     [cb_stage2_adapter/]                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Execution Order (Slurm Jobs)

### Phase 1a: Ds Generation (GPU required)

```bash
# EXISTING: Generate Ds with behavioral filtering
sbatch slurm/Trillium/trillium_mvp_generate_ds.sbatch
# Output: $SCRATCH/cb_mvp_data/ds_stage1.jsonl
```

### Phase 1b: Adversarial-Safe Dr Generation (GPU required, CRITICAL)

```bash
# NEW: Generate adversarial-safe samples
sbatch slurm/Trillium/trillium_stage2_adversarial_safe.sbatch
# Output: data/circuit_breakers/retain/adversarial_safe.jsonl
```

**Script:** `scripts/cb_data_generation/generate_adversarial_safe.py`

```bash
python scripts/cb_data_generation/generate_adversarial_safe.py \
    --b4-data data/fujitsu/orchestrator_attacks_combined_deduplicated.jsonl \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --tool-schema configs/tool_schemas/b4_standard_v1.json \
    --output data/circuit_breakers/retain/adversarial_safe.jsonl \
    --target-n 500 \
    --temperature 0.1 \
    --backend vllm \
    --tensor-parallel 4
```

### Phase 1c: CPU-only Ingestion (parallel)

```bash
# AgentDojo split
python scripts/cb_data_generation/ingest_agentdojo_splits.py \
    --input-dir data/agent_dojo/ \
    --output-harmful data/circuit_breakers/harmful/agentdojo_failures.jsonl \
    --output-retain data/circuit_breakers/retain/agentdojo_resisted.jsonl \
    --max-each 500

# TAU2 traces
python scripts/cb_data_generation/ingest_tau2_traces.py \
    --tau2-path data/tau2_repo/ \
    --output data/circuit_breakers/retain/tau2_traces.jsonl \
    --target-n 500

# UltraChat (requires network on login node)
python scripts/cb_data_generation/ingest_ultrachat.py \
    --output data/circuit_breakers/retain/ultrachat_subset.jsonl \
    --n-samples 2000

# XSTest (requires network)
python scripts/cb_data_generation/ingest_xstest.py \
    --output data/circuit_breakers/retain/xstest_borderline.jsonl \
    --target-n 500
```

### Phase 2: Validation & Merge

```bash
# Validate all sources
python scripts/cb_data_generation/validate_format.py \
    --data-dir data/circuit_breakers/ \
    --strict \
    --report data/circuit_breakers/stage2/validation_report.json

# Merge into training file
python scripts/cb_data_generation/merge_stage2_data.py \
    --harmful-dir data/circuit_breakers/harmful/ \
    --retain-dir data/circuit_breakers/retain/ \
    --output data/circuit_breakers/stage2/train.jsonl \
    --eval-output data/circuit_breakers/stage2/eval.jsonl \
    --eval-fraction 0.1 \
    --dr-ratio 4.5 \
    --validate

# Final preflight check
python scripts/circuit_breakers/preflight_check.py \
    --train-data data/circuit_breakers/stage2/train.jsonl \
    --tool-schema configs/tool_schemas/b4_standard_v1.json \
    --strict
```

### Phase 3: Training

```bash
sbatch slurm/Trillium/trillium_stage2_train.sbatch
```

---

## Validators (Stop-the-Line Checks)

### V1: Per-Source Validation

| Source | Validator | Fail Condition |
|--------|-----------|----------------|
| Ds | `validate_format.py --split harmful` | Any R1-R6 error |
| Adversarial-safe | Check `is_adversarial_safe: true` | <400 samples |
| AgentDojo | Check `security` field exists | >10% missing |
| TAU2 | Check message format | <300 samples |
| UltraChat | Check `apply_chat_template` renders | Any error |

### V2: Merge-Time Checks

```python
# In merge_stage2_data.py
def validate_merge(ds_samples, dr_samples):
    # Check 1: Ratio
    ratio = len(dr_samples) / max(len(ds_samples), 1)
    assert ratio >= 4.0, f"Dr:Ds ratio {ratio:.2f} < 4.0"
    
    # Check 2: Adversarial-safe count
    adv_safe = sum(1 for s in dr_samples 
                   if s.get("labels", {}).get("is_adversarial_safe"))
    assert adv_safe >= 400, f"Adversarial-safe count {adv_safe} < 400"
    
    # Check 3: No duplicate IDs
    all_ids = [s["id"] for s in ds_samples + dr_samples]
    assert len(all_ids) == len(set(all_ids)), "Duplicate IDs found"
    
    # Check 4: Tool samples have tools field
    for s in ds_samples + dr_samples:
        if s.get("labels", {}).get("priority_class") != "general_conversation":
            assert s.get("tools"), f"Missing tools for {s['id']}"
```

### V3: Preflight Check (Before Training)

```bash
python scripts/circuit_breakers/preflight_check.py \
    --train-data data/circuit_breakers/stage2/train.jsonl \
    --tool-schema configs/tool_schemas/b4_standard_v1.json \
    --check-tokenization \
    --model meta-llama/Llama-3.1-8B-Instruct
```

Checks:
1. All samples tokenize without error
2. Assistant boundary tokens detected for loss masking
3. No samples exceed `max_seq_length` (2048)
4. Tool JSON parses correctly

---

## Output File Structure

```
data/circuit_breakers/
├── harmful/                           # Ds sources
│   ├── b4_flips.jsonl                # Stage 1 (keep)
│   └── agentdojo_failures.jsonl      # Stage 2 (new)
├── retain/                            # Dr sources
│   ├── b4_benign_twins.jsonl         # Stage 1 (keep)
│   ├── adversarial_safe.jsonl        # Stage 2 (CRITICAL)
│   ├── agentdojo_resisted.jsonl      # Stage 2 (new)
│   ├── tau2_traces.jsonl             # Stage 2 (new)
│   ├── ultrachat_subset.jsonl        # Stage 2 (new)
│   └── xstest_borderline.jsonl       # Stage 2 (optional)
├── stage2/                            # Final merged
│   ├── train.jsonl                   # Training data
│   ├── eval.jsonl                    # Held-out eval
│   ├── validation_report.json        # Validation results
│   └── stats.json                    # Dataset statistics
└── schemas/
    └── canonical_sample_schema.json   # JSON Schema for validation
```

---

## Steps to Implement

| Step | File(s) to Create/Modify | Action |
|------|--------------------------|--------|
| 1 | `slurm/Trillium/trillium_stage2_adversarial_safe.sbatch` | Create sbatch for adversarial-safe generation |
| 2 | `scripts/cb_data_generation/generate_adversarial_safe.py` | Already exists; verify output format |
| 3 | `scripts/cb_data_generation/ingest_agentdojo_splits.py` | Already exists; add `security` field check |
| 4 | `scripts/cb_data_generation/ingest_tau2_traces.py` | Already exists; verify output format |
| 5 | `scripts/cb_data_generation/ingest_ultrachat.py` | Already exists; add `apply_chat_template` validation |
| 6 | `scripts/cb_data_generation/validate_format.py` | Add `--strict` mode, JSON schema validation |
| 7 | `scripts/cb_data_generation/merge_stage2_data.py` | Add ratio enforcement, dedup check |
| 8 | `slurm/Trillium/trillium_mvp_validate.sbatch` | Update to call Stage 2 validation |

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| GPU job fails mid-generation | Use `--resume` flag; checkpoint every 100 samples |
| Network unavailable for UltraChat | Pre-download on login node; cache in `$SCRATCH` |
| AgentDojo format varies across files | Handle both old/new metadata locations |
| Merge creates unbalanced batches | Stratified sampling in dataloader, not merge |

---

## Definition of Done

- [ ] All Phase 1 scripts complete without error
- [ ] `validation_report.json` shows 0 errors
- [ ] `train.jsonl` exists with ≥5,000 samples
- [ ] Dr:Ds ratio ≥4.0 in `stats.json`
- [ ] `preflight_check.py` passes with `--strict`
