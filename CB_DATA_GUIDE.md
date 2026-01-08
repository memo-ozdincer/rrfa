# Circuit Breaker Data Generation Guide

**Version:** 1.0  
**Last Updated:** January 8, 2026  
**Purpose:** Complete guide for generating proper Circuit Breaker training data

---

## ⚠️ CRITICAL: Why Previous Data Was Wrong

The previous data pipeline generated "harmful completions" that looked like this:

```
[TOOL_CALL] search_web
(Expected: retrieve_multimodal_docs, but injection caused flip to: search_web)
```

**This is NOT what a model actually outputs.** It's a human-readable debug label.

Circuit Breakers (Representation Rerouting) works by:
1. Running the harmful prompt + completion through the model
2. Extracting internal representations
3. Training to reroute those representations away from harmful outputs

If the "completion" is text no model ever generates, the representations extracted are useless. **The CB paper explicitly states that quality "largely depends on how precisely the data can elicit the targeted representation."**

---

## The Solution: LLM-Generated Completions

This package provides a complete pipeline for generating **REAL** harmful completions:

```
scripts/cb_data_generation/
├── __init__.py                 # Package init
├── tool_format.py              # Canonical Llama-3.1 tool format
├── llm_harness.py              # vLLM/Transformers generation harness
├── generate_ds.py              # Circuit Breaker Set generator
├── generate_dr.py              # Retain Set generator  
├── generate_eval.py            # Evaluation set generator
└── quality_gates.py            # Data validation checks
```

---

## Quick Start

### 1. Generate Circuit Breaker Set (Ds) - Harmful Completions

```bash
# Using vLLM (recommended for speed)
python scripts/cb_data_generation/generate_ds.py \
    --backend vllm \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --output data/circuit_breakers/ds/circuit_breaker_set.jsonl

# Using transformers (more flexible)
python scripts/cb_data_generation/generate_ds.py \
    --backend transformers \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --load-in-8bit
```

### 2. Generate Retain Set (Dr) - Benign/Refusal Examples

```bash
python scripts/cb_data_generation/generate_dr.py \
    --output data/circuit_breakers/dr/retain_set.jsonl
```

### 3. Generate Evaluation Set

```bash
python scripts/cb_data_generation/generate_eval.py \
    --include-forced-calls \
    --output data/circuit_breakers/eval/eval_set.jsonl
```

### 4. Validate Data Quality

```bash
python scripts/cb_data_generation/quality_gates.py \
    --ds data/circuit_breakers/ds/circuit_breaker_set.jsonl \
    --dr data/circuit_breakers/dr/retain_set.jsonl
```

---

## Dataset Specifications

### Ds: Circuit Breaker Set

Contains examples that reliably elicit harmful internal representations.

**Format:**
```json
{
  "id": "fujitsu_orch_000123#sample2",
  "split": "train",
  "set": "circuit_breaker",
  "messages": [
    {"role": "system", "content": "<AGENT SYSTEM PROMPT>"},
    {"role": "user", "content": "<ATTACKED USER PROMPT>"}
  ],
  "assistant_raw": "<RAW MODEL OUTPUT INCLUDING TOOL CALL>",
  "tool_calls_structured": [
    {"name": "search_web", "arguments": {"query": "..."}}
  ],
  "labels": {
    "is_harmful": true,
    "harm_type": "tool_flip",
    "expected_tool": "retrieve_multimodal_docs",
    "observed_tool": "search_web"
  },
  "provenance": {
    "source_dataset": "orchestrator_attacks_combined_deduplicated.jsonl",
    "record_id": "<original record_id>",
    "generator_model": "meta-llama/Llama-3.1-8B-Instruct",
    "temperature": 0.7,
    "timestamp": "<iso8601>"
  }
}
```

**Key Requirements:**
- `assistant_raw` must be ACTUAL model output (not templates)
- Both structured and raw format preserved
- Provenance tracked for reproducibility

### Dr: Retain Set

Contains examples that should NOT trigger the circuit breaker.

**Format:**
```json
{
  "id": "retain_tool_use_00456",
  "split": "train",
  "set": "retain",
  "messages": [
    {"role": "system", "content": "<AGENT SYSTEM PROMPT>"},
    {"role": "user", "content": "Find recent hours for the ROM in Toronto."}
  ],
  "assistant_raw": "<CORRECT TOOL USE OR NORMAL ANSWER>",
  "tool_calls_structured": [
    {"name": "search_web", "arguments": {"query": "Royal Ontario Museum hours"}}
  ],
  "labels": {
    "is_harmful": false,
    "is_refusal": false,
    "is_correct_tool_use": true
  }
}
```

**Sources:**
1. AgentDojo successful (secure) traces
2. TAU2-Bench task completions
3. Fujitsu B4 failed attacks (correct tool selection)
4. Synthetic refusal examples

### Eval Set

Contains attack prompts for testing CB effectiveness.

**Includes:**
- Standard prompt injection attacks
- Forced function-calling (prefill-style) attacks
- Multi-turn attack traces
- Success criteria for automated grading

---

## Tool-Calling Format

We use **Llama 3.1 format** as the canonical standard:

```
I'll help you find that information.

<|python_tag|>{"name": "search_web", "parameters": {"query": "..."}}
```

**Special Tokens:**
- `<|python_tag|>` - Marks start of tool call
- `<|eom_id|>` - End of message (expecting tool response)
- `<|eot_id|>` - End of turn

**Why This Matters:**
CB is representation-level. Different tokenization = different activations = broken circuit breaker.

---

## Quality Gates (Non-Negotiable)

Before any training run, these checks MUST pass:

### Gate 1: Format Validity
- Every Ds record must have valid `assistant_raw`
- Tool calls must parse correctly
- ≥95% of records valid

### Gate 2: Harmful Content Rate
- Ds must contain actual harmful compliance
- Tool flip attacks must show actual flips
- ≥80% harmful rate required

### Gate 3: Distribution Closeness
- Multiple attack categories represented
- No single category >80%
- Matches real threat model

### Gate 4: Retain Coverage
- Dr must include refusal examples
- Dr must include tool use examples
- Variety in benign tasks

### Gate 5: No Synthetic Markers
- `[TOOL_CALL]` = FAIL
- `(Expected:` = FAIL
- `[HARMFUL:` = FAIL

---

## When to Use What

### Use Raw Data Directly When:
- Constructing user-side attack prompts (`combined_query`)
- Defining target behavior labels
- Building grading functions
- Slicing metrics by category

### Do NOT Use Raw Data When:
- The "completion" field is a debug label
- Training CB for a specific tool format but data has different format
- Data wasn't produced by a model

### Use LLM Generation When:
- Creating Ds for CB training
- Raw data has no completions (AgentHarm)
- Raw data has template completions (Fujitsu B4 orchestrator)

---

## Model Selection for Generation

**For Ds Generation:**

Use a model that:
1. Is capable of being tricked (vulnerable to attacks)
2. Uses the SAME tool-calling format as your target
3. Has similar architecture to your target

Good choices:
- Your target model before defenses
- An uncensored/abliterated model
- A weaker model that complies more readily

**The CB Paper Approach:**
For refusal-trained models, keep harmful assistant responses but optionally redact the harmful user requests. This trains on harmful-state representations without damaging refusal behavior.

---

## Source Data Reference

| Source | Records | Real Completions | Needs Generation |
|--------|--------:|:---------------:|:----------------:|
| Fujitsu B4 | 13,246 | ❌ Labels only | ✅ Yes |
| Fujitsu B1/B3 | 22,946 | ✅ Yes | ❌ No |
| AgentDojo | 3,315 | ✅ Full traces | ❌ No |
| AgentHarm | 476 | ❌ Prompts only | ✅ Yes |
| TAU2 | ~2,458 | ✅ Full traces | ❌ No |

---

## Troubleshooting

### "Model refuses harmful requests"

Use `--redact-harmful-prompt` to keep only the assistant response. Or use an uncensored model as generator.

### "Low tool flip rate"

Increase `--num-samples` (5→10) or `--temperature` (0.7→0.9) for more diversity.

### "Format validation fails"

Check that your model outputs in Llama 3.1 tool format. May need to adjust generation harness.

### "Quality gates fail"

Run with `--dry-run` first to see statistics. Check for:
- Too many refusals in Ds
- Missing tool calls in structured format
- Imbalanced categories

---

## File Locations

```
data/circuit_breakers/
├── ds/                          # Circuit Breaker Set (Ds)
│   └── circuit_breaker_set.jsonl
├── dr/                          # Retain Set (Dr)
│   └── retain_set.jsonl
├── eval/                        # Evaluation Set
│   └── eval_set.jsonl
└── _backups/                    # Old data (deprecated)
```

---

## Related Documentation

- [DATA.md](DATA.md) - Full data inventory
- [CIRCUIT_BREAKERS.md](CIRCUIT_BREAKERS.md) - CB training details
- [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md) - Evaluation procedures
