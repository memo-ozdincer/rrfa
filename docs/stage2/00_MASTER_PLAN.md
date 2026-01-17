# 00_MASTER_PLAN.md — Stage 2+ Implementation Overview

**Version:** 2.0  
**Date:** January 16, 2026  
**Baseline:** [CB_STAGE2_PLAN.md](../../CB_STAGE2_PLAN.md) (preserving core strategy, operationalizing with engineering detail)

---

## Objective

Upgrade the MVP Circuit Breaker system to a robust Stage 2+ implementation that:
1. **Prevents representation collapse** (Stage 1 failure: gibberish output at α=10.0)
2. **Maintains tool-calling capability** while reducing ASR
3. **Uses high-signal training data** (filter to true failures, not already-safe samples)

---

## Failure Modes to Avoid

| Failure Mode | Stage 1 Evidence | Prevention Strategy |
|-------------|------------------|---------------------|
| **Representation collapse** | Output: `to to to to...` | `alpha-max: 0.5` (was 10.0), single target layer |
| **Over-refusal** | `no_tool_call_rate: 100%` | Adversarial-safe Dr samples (resist injection → correct tool) |
| **Training on noise** | Baseline already resists 95% | Filter Ds to true failures OR use susceptible generator |
| **Format mismatch** | Silent failures in loss masking | Strict `<\|python_tag\|>` validation, `apply_chat_template` audit |

---

## Phase Ordering (Critical Path)

```
Phase 0: Audit & Schema Lock (0.5 day)
    └─▶ Phase 1: Data Generation (1.5 days)
            ├── 1a: Adversarial-safe (GPU, critical)
            ├── 1b: AgentDojo split (CPU)
            ├── 1c: TAU2/UltraChat ingest (CPU)
            └─▶ Phase 2: Pipeline & Validation (0.5 day)
                    └─▶ Phase 3: Training (1 day)
                            └─▶ Phase 4: Evaluation (0.5 day)
```

---

## Acceptance Gates (Definition of Done)

### Gate 1: Data Ready (End of Phase 2)
| Check | Threshold | Validation Command |
|-------|-----------|-------------------|
| Ds count | ≥800 samples | `wc -l data/circuit_breakers/harmful/*.jsonl` |
| Dr count | ≥4000 samples | `wc -l data/circuit_breakers/retain/*.jsonl` |
| Dr:Ds ratio | ≥4:1 | Computed in merge script |
| Format compliance | 100% pass | `python scripts/cb_data_generation/validate_format.py --strict` |
| Adversarial-safe count | ≥400 | `grep -c adversarial_safe merged.jsonl` |

### Gate 2: Training Stable (End of Phase 3)
| Check | Threshold | Source |
|-------|-----------|--------|
| No NaN/Inf loss | 0 occurrences | W&B logs |
| cos_sim_mean decreasing | Trend over first 50 steps | W&B `rr/cos_sim_mean` |
| Output coherence | Manual spot-check (10 samples) | `sanity_check.py` |

### Gate 3: Evaluation Pass (End of Phase 4)
| Metric | Stage 2 Target | Stretch Goal | Source |
|--------|---------------|--------------|--------|
| **Coherent output** | 100% | — | Manual check |
| **correct_behavior_rate** | ≥70% | ≥80% | `eval_mvp.py` |
| **no_tool_call_rate** | ≤20% | ≤10% | `eval_mvp.py` |
| **ASR relative reduction** | ≥50% | ≥80% | `eval_mvp.py` |
| **capability_retention** | ≥90% | ≥95% | Benign subset eval |

---

## Key Changes from CB_STAGE2_PLAN.md

| Aspect | CB_STAGE2_PLAN.md | This Implementation |
|--------|-------------------|---------------------|
| Data format | JSONL throughout | JSONL raw → Parquet training shards (optional) |
| Schema | Implicit | Explicit `01_DATA_SPEC.md` with validation |
| Loss masking | Described conceptually | Explicit token-mask spec in `04_TRAINING_PRIORITY_REGIME.md` |
| `apply_chat_template` | Not addressed | Audit step + validation rule added |
| Curriculum | Not specified | Optional pacing schedule proposed |
| Pipeline DAG | List of commands | Formal DAG with validators in `02_DATA_PIPELINE_PLAN.md` |

---

## Inputs/Outputs Summary

### Inputs (Existing)
- `data/fujitsu/orchestrator_attacks_combined_deduplicated.jsonl` (13,246 B4)
- `data/agent_dojo/agentdojo-*.jsonl` (3,315 traces)
- `data/tau2_repo/data/tau2/domains/*/tasks.json` (~2,458 tasks)
- `configs/tool_schemas/b4_standard_v1.json` (frozen schema)
- `scripts/cb_data_generation/*.py` (generation scripts)

### Outputs (New/Updated)
- `data/circuit_breakers/stage2/` — Curated training data
- `data/circuit_breakers/stage2/train.jsonl` — Final merged training file
- `data/circuit_breakers/stage2/eval.jsonl` — Held-out eval set
- `outputs/cb_stage2_adapter/` — Trained LoRA adapter
- `docs/stage2/` — This documentation bundle

---

## Risk Mitigation Matrix

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Adversarial-safe yield <50% | Low | High | Use temp=0.1; baseline resists 95% |
| AgentDojo security field missing | Medium | Medium | Fallback to heuristic parsing |
| UltraChat format mismatch | Medium | Low | Validate with `apply_chat_template` |
| Training divergence | Low | High | Start from Stage 1 checkpoint; lower LR |
| Eval parsing fails on CB output | Medium | Medium | Add `is_valid_json` fallback; log raw outputs |

---

## Files in This Bundle

| File | Purpose |
|------|---------|
| `00_MASTER_PLAN.md` | This overview |
| `01_DATA_SPEC.md` | Canonical schema, labeling, validation rules |
| `02_DATA_PIPELINE_PLAN.md` | Pipeline DAG, scripts, validators |
| `03_GENERATION_PACKAGE.md` | Synthetic generation design |
| `04_TRAINING_PRIORITY_REGIME.md` | Loss masking + mixture weights |
| `05_EVAL_SPEC.md` | Metrics, parsers, gates |
| `06_CHECKLIST.md` | Executable implementation checklist |
