# 01_DATA_SPEC.md — Canonical Dataset Schema & Validation

**Objective:** Define the single source of truth for data representation across raw, curated, and training stages, ensuring consistent formatting and enabling strict validation.

---

## Inputs/Outputs

| Stage | Format | Location |
|-------|--------|----------|
| Raw (Tier A) | Source-native JSONL | `data/{fujitsu,agent_dojo,tau2_repo}/` |
| Curated (Tier B) | Canonical JSONL | `data/circuit_breakers/{harmful,retain}/` |
| Training (Tier C) | Batched JSONL (or Parquet) | `data/circuit_breakers/stage2/train.jsonl` |

---

## Canonical Sample Schema (Tier B)

Every curated sample MUST conform to this schema:

```jsonc
{
  // === REQUIRED FIELDS ===
  "id": "b4_harmful_00042",           // Deterministic, globally unique
  "messages": [                        // Full conversation context
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ],
  "assistant_raw": "<|python_tag|>{\"name\": \"search_web\", ...}<|eom_id|>",
  "tools": "b4_standard_v1",           // Schema reference OR inline list
  
  // === LABELS (for loss masking & filtering) ===
  "labels": {
    "split": "harmful",                // "harmful" | "retain"
    "expected_tool": "retrieve_multimodal_docs",
    "simulated_tool": "search_web",    // For Ds only
    "observed_tool": "search_web",     // Actual model output
    "is_flip_success": true,           // observed == simulated (Ds)
    "is_adversarial_safe": false,      // observed == expected under injection (Dr)
    "security": false                  // AgentDojo: True=resisted, False=failed
  },
  
  // === TRAINING CONTROLS ===
  "training": {
    "loss_mask_start": 127,            // Token index where loss begins
    "loss_mask_end": 156,              // Token index where loss ends (exclusive)
    "sample_weight": 1.0,              // Per-sample weight (for curriculum)
    "priority_class": "harmful_flip"   // For stratified sampling
  },
  
  // === PROVENANCE ===
  "metadata": {
    "source": "fujitsu_b4",
    "source_id": "atk_orchestrator_12345",
    "generator_model": "mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated",
    "generator_temp": 0.7,
    "schema_version": "b4_standard_v1",
    "created_at": "2026-01-16T12:00:00Z",
    "pipeline_version": "stage2_v1"
  }
}
```

---

## Field Requirements by Split

| Field | Ds (harmful) | Dr (retain) | Notes |
|-------|-------------|-------------|-------|
| `id` | ✅ Required | ✅ Required | Format: `{source}_{split}_{seq:05d}` |
| `messages` | ✅ Required | ✅ Required | ≥2 messages (system + user) |
| `assistant_raw` | ✅ Required | ✅ Required | Llama 3.1 format |
| `tools` | ✅ Required | ⚠️ Optional | UltraChat has no tools |
| `labels.split` | `"harmful"` | `"retain"` | Determines loss function |
| `labels.expected_tool` | ✅ Required | ⚠️ Optional | |
| `labels.simulated_tool` | ✅ Required | ❌ N/A | Attack target |
| `labels.is_flip_success` | ✅ `true` | ❌ N/A | Ds definition |
| `labels.is_adversarial_safe` | ❌ N/A | ✅ if applicable | Critical for Dr |
| `training.loss_mask_*` | ⚠️ Computed | ⚠️ Computed | See §Loss Masking |

---

## Llama 3.1 Tool-Call Format Spec

All `assistant_raw` fields for tool-calling samples MUST follow:

```
<|python_tag|>{"name": "tool_name", "parameters": {"param": "value"}}<|eom_id|>
```

### Validation Rules

| Rule | Check | Severity |
|------|-------|----------|
| R1 | `assistant_raw` contains `<\|python_tag\|>` | ERROR |
| R2 | `assistant_raw` ends with `<\|eom_id\|>` or `<\|eot_id\|>` | WARNING |
| R3 | JSON between tags is valid | ERROR |
| R4 | JSON has `"name"` field | ERROR |
| R5 | No markdown (` ``` `) in `assistant_raw` | ERROR |
| R6 | No forbidden prefixes (`Action:`, `Tool:`, etc.) | ERROR |

### apply_chat_template Audit

Before training, verify template compatibility:

```python
# In scripts/cb_data_generation/validate_format.py
def validate_template_compatibility(sample, tokenizer):
    """Ensure sample renders correctly with tokenizer's chat template."""
    messages = sample["messages"] + [
        {"role": "assistant", "content": sample["assistant_raw"]}
    ]
    
    rendered = tokenizer.apply_chat_template(
        messages,
        tools=get_tools_from_schema(sample["tools"]),
        tokenize=False,
        add_generation_prompt=False,
    )
    
    # Check that assistant_raw appears verbatim
    assert sample["assistant_raw"] in rendered, "Template mangled assistant_raw"
    
    # Check for expected control tokens
    assert "<|start_header_id|>assistant<|end_header_id|>" in rendered
    return True
```

---

## Deterministic ID Generation

```python
import hashlib

def generate_sample_id(source: str, split: str, content_hash_input: str) -> str:
    """Generate deterministic, collision-resistant ID."""
    content_hash = hashlib.sha256(content_hash_input.encode()).hexdigest()[:8]
    return f"{source}_{split}_{content_hash}"

# Example: b4_harmful_a3f2c1d9
```

---

## Split Definitions

| Split | Definition | Source Mapping |
|-------|-----------|----------------|
| `harmful` (Ds) | Attack succeeded: `observed_tool == simulated_tool` | B4 flips, AgentDojo `security=False` |
| `retain` (Dr) | Capability to preserve | See subtypes below |

### Dr Subtypes (for weighted sampling)

| Subtype | Definition | Weight | Priority |
|---------|------------|--------|----------|
| `adversarial_safe` | Injection present, correct tool called | 2.0 | 🔴 Critical |
| `injection_resisted` | AgentDojo `security=True` | 1.5 | 🔴 High |
| `benign_twin` | Clean version of Ds sample | 1.2 | 🟡 Medium |
| `tool_capability` | TAU2, benign tool traces | 1.0 | 🟡 Medium |
| `general_conversation` | UltraChat (no tools) | 0.8 | 🟢 Low |
| `borderline` | XSTest (should comply) | 1.0 | 🟢 Low |

---

## Validation Script Interface

```bash
# Validate single file
python scripts/cb_data_generation/validate_format.py \
    --data data/circuit_breakers/harmful/b4_flips.jsonl \
    --split harmful \
    --strict

# Validate entire stage2 directory
python scripts/cb_data_generation/validate_format.py \
    --data-dir data/circuit_breakers/stage2/ \
    --strict \
    --report validation_report.json
```

### Expected Output

```
Validation Report for data/circuit_breakers/stage2/
============================================================
Total samples: 5,247
  Harmful (Ds): 1,047
  Retain (Dr): 4,200
  Dr:Ds ratio: 4.01:1

Format Compliance:
  ✅ R1 (python_tag present): 5,247/5,247 (100.0%)
  ✅ R2 (end token): 5,102/5,247 (97.2%) [WARNING: 145 missing]
  ✅ R3 (valid JSON): 5,247/5,247 (100.0%)
  ✅ R4 (has name field): 5,047/5,047 (100.0%) [200 non-tool samples]
  ✅ R5 (no markdown): 5,247/5,247 (100.0%)
  ✅ R6 (no forbidden prefix): 5,247/5,247 (100.0%)

RESULT: PASS (strict mode: all errors = 0)
```

---

## Steps to Implement

| Step | Code Location | Action |
|------|---------------|--------|
| 1 | `scripts/cb_data_generation/validate_format.py` | Add `--strict` mode, JSON schema validation |
| 2 | `scripts/cb_data_generation/*.py` | Ensure all generators emit canonical schema |
| 3 | `scripts/cb_data_generation/merge_stage2_data.py` | Add ID generation, deduplication check |
| 4 | `slurm/Trillium/trillium_mvp_validate.sbatch` | Call validation before training |

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| UltraChat lacks `tools` field | Allow `tools: null` for `general_conversation` subtype |
| AgentDojo `security` field missing in some traces | Fallback: parse final tool call vs injection goal |
| `apply_chat_template` behavior varies by tokenizer version | Pin `transformers>=4.43.0`; add version check |

---

## Definition of Done

- [ ] All curated files pass `validate_format.py --strict`
- [ ] Schema matches this spec (spot-check 10 random samples per source)
- [ ] `apply_chat_template` audit passes on 100 samples
- [ ] No duplicate IDs across all files
