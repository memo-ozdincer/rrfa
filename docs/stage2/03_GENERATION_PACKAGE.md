# 03_GENERATION_PACKAGE.md — Synthetic Data Generation Design

**Objective:** Define which data to generate, from which models, with what filters, to produce high-signal Ds/Dr samples that avoid training on noise.

---

## Inputs/Outputs

### Inputs
| Artifact | Path | Purpose |
|----------|------|---------|
| B4 attack records | `data/fujitsu/orchestrator_attacks_combined_deduplicated.jsonl` | Source prompts |
| Tool schema | `configs/tool_schemas/b4_standard_v1.json` | Frozen tool definitions |
| AgentDojo traces | `data/agent_dojo/agentdojo-*.jsonl` | Multi-turn traces |

### Outputs
| Artifact | Path | Expected Count |
|----------|------|---------------:|
| Ds (harmful) | `data/circuit_breakers/harmful/b4_flips.jsonl` | ~500 |
| Ds (AgentDojo) | `data/circuit_breakers/harmful/agentdojo_failures.jsonl` | ~500 |
| Dr (adversarial-safe) | `data/circuit_breakers/retain/adversarial_safe.jsonl` | ~500 |
| Dr (AgentDojo) | `data/circuit_breakers/retain/agentdojo_resisted.jsonl` | ~500 |

---

## Generation Strategy Overview

### The Problem: Low-Signal Training Data

**Stage 1 Finding:** Llama-3.1-8B-Instruct baseline already resists 94.8% of B4 attacks (ASR 5.2%). Training on samples where the model already succeeds provides weak gradient signal and can cause over-refusal.

**Solution:** Two-pronged approach:
1. **For Ds:** Use a more susceptible model (abliterated) OR filter to true failure cases
2. **For Dr:** Generate adversarial-safe samples at low temperature to maximize "resist injection → correct tool" yield

---

## Ds Generation: Harmful Set

### Model Selection

| Model | ASR Estimate | Use Case |
|-------|-------------|----------|
| `mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated` | ~60-80% | **Primary Ds generator** |
| `meta-llama/Llama-3.1-8B` (base) | ~70-90% | Alternative (no instruct tuning) |
| `mistralai/Mistral-7B-Instruct-v0.2` | ~30-50% | Cross-model attacks |

**Rationale:** The abliterated model has safety guardrails removed, making attacks more likely to succeed. This produces high-signal Ds samples where the model actually exhibits harmful behavior.

### Ds Generation Pipeline

```python
# scripts/cb_data_generation/generate_ds_mvp.py (existing, verified)
def generate_ds_sample(record, model, tokenizer, tools):
    """
    Generate Ds sample using behavioral filtering.
    
    CRITICAL: Only include if attack SUCCEEDS (observed == simulated).
    """
    messages = [
        {"role": "system", "content": get_system_prompt()},
        {"role": "user", "content": record["combined_query"]}  # Contains injection
    ]
    
    response = generate_with_tools(
        model, tokenizer, messages, tools,
        temperature=0.7,  # Higher temp for Ds (more variation)
        max_new_tokens=512
    )
    
    observed_tool = extract_tool_name(response)
    simulated_tool = record["simulated_tool"]
    expected_tool = record["expected_tool"]
    
    # BEHAVIORAL FILTER: Only keep if attack succeeded
    if observed_tool == simulated_tool and observed_tool != expected_tool:
        return {
            "id": generate_sample_id("b4", "harmful", record["record_id"]),
            "messages": messages,
            "assistant_raw": response,
            "tools": "b4_standard_v1",
            "labels": {
                "split": "harmful",
                "expected_tool": expected_tool,
                "simulated_tool": simulated_tool,
                "observed_tool": observed_tool,
                "is_flip_success": True
            },
            "metadata": {
                "source": "fujitsu_b4",
                "generator_model": "abliterated",
                "generator_temp": 0.7
            }
        }
    return None  # Filter out
```

### Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `temperature` | 0.7 | Higher variation in harmful outputs |
| `top_p` | 0.95 | Standard |
| `max_new_tokens` | 512 | Sufficient for tool calls |
| `min_yield` | 10% | Fail job if yield too low |
| `target_n` | 500 | Stage 2 target |

---

## Dr Generation: Adversarial-Safe Samples (CRITICAL)

### The Key Innovation

**Adversarial-safe samples** are where:
- Input contains injection attempt (malicious prompt)
- Model RESISTS and calls the CORRECT tool anyway

These samples teach "injection present → still call correct tool" instead of "injection present → refuse everything".

### Dr Generation Pipeline

```python
# scripts/cb_data_generation/generate_adversarial_safe.py
def generate_adversarial_safe_sample(record, model, tokenizer, tools):
    """
    Generate adversarial-safe Dr sample.
    
    CRITICAL: Only include if model RESISTED (observed == expected).
    Use LOW TEMPERATURE to maximize resistance rate.
    """
    messages = [
        {"role": "system", "content": get_system_prompt()},
        {"role": "user", "content": record["combined_query"]}  # Contains injection
    ]
    
    response = generate_with_tools(
        model, tokenizer, messages, tools,
        temperature=0.1,  # LOW temp = more deterministic = more resistance
        max_new_tokens=512
    )
    
    observed_tool = extract_tool_name(response)
    expected_tool = record["expected_tool"]
    
    # ADVERSARIAL-SAFE FILTER: Only keep if model resisted
    if observed_tool == expected_tool:
        return {
            "id": generate_sample_id("b4", "adversarial_safe", record["record_id"]),
            "messages": messages,
            "assistant_raw": response,
            "tools": "b4_standard_v1",
            "labels": {
                "split": "retain",
                "expected_tool": expected_tool,
                "observed_tool": observed_tool,
                "is_adversarial_safe": True,
                "priority_class": "adversarial_safe"
            },
            "metadata": {
                "source": "fujitsu_b4",
                "generator_model": "Llama-3.1-8B-Instruct",
                "generator_temp": 0.1
            }
        }
    return None  # Filter out (attack succeeded - goes to Ds instead)
```

### Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `temperature` | 0.1 | **Low temp = more resistance** |
| `model` | `Llama-3.1-8B-Instruct` | Target model (non-abliterated) |
| `target_n` | 500 | Stage 2 target |
| `expected_yield` | ~95% | Based on baseline ASR 5.2% |

### Expected Yield Calculation

From Stage 1 evaluation:
- Baseline ASR: 5.2% (attacks succeed)
- Therefore: 94.8% resist (correct tool)
- With 13,246 B4 records → expect ~12,500 adversarial-safe
- Only need 500 → comfortable margin

---

## AgentDojo Split

AgentDojo traces have a `security` field indicating whether the agent resisted injection.

```python
# scripts/cb_data_generation/ingest_agentdojo_splits.py
def split_by_security(record):
    """
    Split AgentDojo record by security outcome.
    
    security=True  → Agent resisted injection (Dr)
    security=False → Attack succeeded (Ds)
    """
    metadata = record.get("metadata", {})
    
    # Only process records WITH injections
    if metadata.get("injection_task_id") is None:
        return None, None  # Benign baseline - skip for now
    
    security = metadata.get("security", True)  # Default safe
    
    sample = {
        "id": generate_sample_id("agentdojo", "harmful" if not security else "resisted", 
                                  record.get("task_id", "")),
        "messages": record["messages"],
        "assistant_raw": extract_last_assistant(record),
        "labels": {
            "split": "harmful" if not security else "retain",
            "security": security,
            "success": metadata.get("success", False),
            "priority_class": "injection_failure" if not security else "injection_resisted"
        },
        "metadata": {
            "source": "agentdojo",
            "suite_name": metadata.get("suite_name"),
            "injection_task_id": metadata.get("injection_task_id")
        }
    }
    
    if security:
        return None, sample  # Dr
    else:
        return sample, None  # Ds
```

---

## Quality Controls

### 1. Deduplication

```python
def deduplicate_samples(samples, key_fn=lambda s: s["assistant_raw"][:200]):
    """Remove near-duplicate samples based on response prefix."""
    seen = set()
    unique = []
    for s in samples:
        key = key_fn(s)
        if key not in seen:
            seen.add(key)
            unique.append(s)
    return unique
```

### 2. Similarity Filtering

```python
def filter_high_similarity(new_samples, existing_samples, threshold=0.9):
    """Filter out samples too similar to existing data."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    
    existing_texts = [s["assistant_raw"] for s in existing_samples]
    new_texts = [s["assistant_raw"] for s in new_samples]
    
    vectorizer = TfidfVectorizer(max_features=5000)
    all_vectors = vectorizer.fit_transform(existing_texts + new_texts)
    
    existing_vectors = all_vectors[:len(existing_texts)]
    new_vectors = all_vectors[len(existing_texts):]
    
    filtered = []
    for i, new_vec in enumerate(new_vectors):
        max_sim = cosine_similarity(new_vec, existing_vectors).max()
        if max_sim < threshold:
            filtered.append(new_samples[i])
    
    return filtered
```

### 3. Quarantine Bucket

Samples that fail validation go to quarantine (not silently dropped):

```python
def validate_and_quarantine(sample):
    """Validate sample; return (valid_sample, quarantine_sample)."""
    errors = []
    
    if "<|python_tag|>" not in sample.get("assistant_raw", ""):
        errors.append("missing_python_tag")
    
    try:
        tool_name, params, is_valid = extract_tool_call(sample["assistant_raw"])
        if not is_valid:
            errors.append("invalid_json")
    except Exception as e:
        errors.append(f"parse_error:{str(e)}")
    
    if errors:
        sample["_quarantine_reasons"] = errors
        return None, sample
    return sample, None
```

---

## Failure-Mining Loop (Optional Enhancement)

For maximum Ds signal, implement a failure-mining loop:

```python
def mine_failures(records, models, tools, target_n=500):
    """
    Mine for true failure cases across multiple models.
    
    A "true failure" is where ALL models call the wrong tool.
    """
    failures = []
    
    for record in tqdm(records):
        all_failed = True
        best_harmful_output = None
        
        for model_name, model in models.items():
            response = generate_with_tools(model, ...)
            observed = extract_tool_name(response)
            
            if observed == record["expected_tool"]:
                all_failed = False
                break
            elif observed == record["simulated_tool"]:
                best_harmful_output = response
        
        if all_failed and best_harmful_output:
            failures.append(create_sample(record, best_harmful_output))
        
        if len(failures) >= target_n:
            break
    
    return failures
```

---

## Steps to Implement

| Step | Script | Action |
|------|--------|--------|
| 1 | `generate_ds_mvp.py` | Verify abliterated model usage; check yield |
| 2 | `generate_adversarial_safe.py` | Verify temp=0.1; add progress checkpoint |
| 3 | `ingest_agentdojo_splits.py` | Add fallback for missing `security` field |
| 4 | `merge_stage2_data.py` | Add deduplication step |
| 5 | `validate_format.py` | Add quarantine bucket output |

---

## Validation Checks

| Check | Command | Pass Condition |
|-------|---------|----------------|
| Ds yield | `grep -c is_flip_success.*true ds.jsonl` | ≥500 |
| Adversarial-safe yield | `grep -c is_adversarial_safe.*true adv_safe.jsonl` | ≥400 |
| No duplicates | `jq -r .id *.jsonl \| sort \| uniq -d \| wc -l` | 0 |
| Format compliance | `validate_format.py --strict` | 0 errors |

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Abliterated model not cached | Add to `prefetch_models.sh` |
| Adversarial-safe yield lower than expected | Increase input record count; try temp=0.05 |
| AgentDojo `security` field inconsistent | Fallback: compare final tool to injection goal |
| Deduplication removes too many | Lower threshold from 0.9 to 0.85 |

---

## Definition of Done

- [ ] Ds file has ≥500 samples with `is_flip_success: true`
- [ ] Adversarial-safe file has ≥400 samples with `is_adversarial_safe: true`
- [ ] No samples in quarantine bucket (or quarantine reviewed and resolved)
- [ ] Deduplication reduces count by <5%
- [ ] All outputs pass `validate_format.py --strict`
