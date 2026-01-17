# 04_TRAINING_PRIORITY_REGIME.md — Loss Masking & Mixture Weights

**Objective:** Formalize the training regime as implementable loss masking + mixture weighting + (optional) curriculum pacing, explicitly designed to prevent Stage 1 collapse.

---

## Inputs/Outputs

### Inputs
| Artifact | Purpose |
|----------|---------|
| `data/circuit_breakers/stage2/train.jsonl` | Training data with labels |
| `scripts/circuit_breakers/config.py` | Hyperparameter configuration |
| `scripts/circuit_breakers/trainer.py` | Training loop implementation |

### Outputs
| Artifact | Purpose |
|----------|---------|
| Updated `config.py` | Stage 2 hyperparameters |
| Updated `trainer.py` | Loss masking implementation |
| W&B metrics | Training stability verification |

---

## Renaming: Training-Priority Regime → Loss Masking Policy + Mixture Schedule

| Old Term | New Term | Definition |
|----------|----------|------------|
| "Training-priority regime" | **Loss Masking Policy (LMP)** | Which tokens contribute to loss |
| (implicit) | **Mixture Weighting & Curriculum Schedule (MWCS)** | Sample weights + sampling over time |

---

## Loss Masking Policy (LMP)

### Core Principle

**"Train only on the target behavior, not the context."**

For each sample, we compute loss ONLY on specific token spans:
- **Ds (harmful):** Loss on harmful assistant response tokens only
- **Dr (retain):** Loss on desired assistant response tokens only

The prompt/context is included in the forward pass (for correct conditioning) but masked from the loss.

### Token-Level Mask Computation

```python
# In trainer.py
def compute_loss_mask(
    input_ids: torch.Tensor,
    tokenizer,
    sample_type: str,  # "harmful" or "retain"
) -> torch.Tensor:
    """
    Compute token-level loss mask.
    
    Returns mask where 1 = compute loss, 0 = ignore.
    """
    # Find assistant response boundaries
    # Llama 3.1: <|start_header_id|>assistant<|end_header_id|>...<|eot_id|>
    
    text = tokenizer.decode(input_ids)
    
    # Find assistant start marker
    assistant_start_pattern = r"<\|start_header_id\|>assistant<\|end_header_id\|>"
    match = re.search(assistant_start_pattern, text)
    
    if not match:
        # Fallback: mask from first tool call marker
        if "<|python_tag|>" in text:
            start_pos = text.find("<|python_tag|>")
        else:
            return torch.ones_like(input_ids, dtype=torch.float)  # Full loss
    else:
        # Start masking AFTER the header
        start_pos = match.end()
    
    # Find end token
    end_patterns = ["<|eot_id|>", "<|eom_id|>", "</s>"]
    end_pos = len(text)
    for pattern in end_patterns:
        pos = text.find(pattern, start_pos)
        if pos != -1:
            end_pos = min(end_pos, pos + len(pattern))
            break
    
    # Convert character positions to token indices
    prefix_tokens = tokenizer.encode(text[:start_pos], add_special_tokens=False)
    response_tokens = tokenizer.encode(text[start_pos:end_pos], add_special_tokens=False)
    
    mask = torch.zeros_like(input_ids, dtype=torch.float)
    start_idx = len(prefix_tokens)
    end_idx = start_idx + len(response_tokens)
    mask[start_idx:end_idx] = 1.0
    
    return mask
```

### LMP by Sample Type

| Sample Type | Loss Mask Span | Rationale |
|-------------|---------------|-----------|
| **Ds (harmful flip)** | Only `assistant_raw` tokens | Push harmful response representations orthogonal |
| **Dr (adversarial-safe)** | Only `assistant_raw` tokens | Anchor correct-tool-under-attack behavior |
| **Dr (benign twin)** | Only `assistant_raw` tokens | Preserve basic tool-calling |
| **Dr (general)** | Only `assistant_raw` tokens | Preserve coherent generation |

### Implementation in Trainer

```python
# In CircuitBreakerTrainer.train_step()
def train_step(self, batch):
    harmful_samples = batch["harmful"]
    retain_samples = batch["benign"]
    
    # === Harmful (Ds) ===
    for sample in harmful_samples:
        inputs = self.tokenize(sample)
        loss_mask = compute_loss_mask(inputs["input_ids"], self.tokenizer, "harmful")
        
        # Forward through BOTH models
        with torch.no_grad():
            frozen_outputs = self.frozen_model(**inputs, output_hidden_states=True)
        model_outputs = self.model(**inputs, output_hidden_states=True)
        
        # Extract representations
        frozen_reps = {i: frozen_outputs.hidden_states[i] for i in self.target_layers}
        model_reps = {i: model_outputs.hidden_states[i] for i in self.target_layers}
        
        # Rerouting loss with mask
        l_rr = reroute_loss(
            model_reps, frozen_reps, self.target_layers,
            attention_mask=inputs["attention_mask"],
            loss_mask=loss_mask  # CRITICAL: Only on assistant tokens
        )
    
    # === Retain (Dr) ===
    for sample in retain_samples:
        inputs = self.tokenize(sample)
        loss_mask = compute_loss_mask(inputs["input_ids"], self.tokenizer, "retain")
        
        # ... similar, but use retain_loss()
        l_ret = retain_loss(
            model_reps, frozen_reps, self.target_layers,
            attention_mask=inputs["attention_mask"],
            loss_mask=loss_mask
        )
    
    return l_rr, l_ret
```

---

## Mixture Weighting & Curriculum Schedule (MWCS)

### Sample Weights by Priority Class

| Priority Class | Base Weight | Rationale |
|----------------|-------------|-----------|
| `adversarial_safe` | 2.0 | **CRITICAL:** Prevents over-refusal |
| `injection_resisted` | 1.5 | AgentDojo multi-turn resistance |
| `benign_twin` | 1.2 | Paired with Ds samples |
| `tool_capability` | 1.0 | TAU2, general tool use |
| `general_conversation` | 0.8 | UltraChat (no tools) |
| `borderline` | 1.0 | XSTest edge cases |
| `harmful_flip` | 1.0 | Standard Ds weight |

### Effective Dr:Ds Ratio

With 4.5:1 count ratio and weights:

```python
# Compute effective ratio
ds_count = 1000
dr_count = 4500

ds_effective = ds_count * 1.0  # All harmful_flip
dr_effective = (
    500 * 2.0 +  # adversarial_safe
    500 * 1.5 +  # injection_resisted
    500 * 1.2 +  # benign_twin
    500 * 1.0 +  # tool_capability
    2000 * 0.8 + # general_conversation
    500 * 1.0    # borderline
) = 5450

effective_ratio = dr_effective / ds_effective = 5.45:1
```

### Curriculum (Optional, Phase 2.5)

Start with stability-focused mix, gradually increase adversarial difficulty:

```python
def get_curriculum_weights(step: int, total_steps: int) -> Dict[str, float]:
    """
    Curriculum pacing: start easy, increase adversarial density.
    
    Phase 1 (0-30%): Heavy general capability, light adversarial
    Phase 2 (30-70%): Balanced mix
    Phase 3 (70-100%): Full adversarial density
    """
    progress = step / total_steps
    
    if progress < 0.3:
        # Phase 1: Stability
        return {
            "adversarial_safe": 1.0,  # Reduced from 2.0
            "general_conversation": 1.5,  # Increased from 0.8
            "harmful_flip": 0.5,  # Reduced
        }
    elif progress < 0.7:
        # Phase 2: Balanced
        return {
            "adversarial_safe": 2.0,
            "general_conversation": 0.8,
            "harmful_flip": 1.0,
        }
    else:
        # Phase 3: Full adversarial
        return {
            "adversarial_safe": 2.5,  # Increased
            "general_conversation": 0.5,  # Reduced
            "harmful_flip": 1.2,
        }
```

---

## Stage 2 Hyperparameters (Preventing Collapse)

### Key Changes from Stage 1

| Parameter | Stage 1 | Stage 2 | Rationale |
|-----------|---------|---------|-----------|
| `alpha_max` | 10.0 | **0.5** | Primary collapse fix |
| `cb_target_layers` | `[10, 20]` | **`[15]`** | Single mid-layer = less aggressive |
| `max_grad_norm` | 1.0 | **0.5** | Tighter clipping |
| `total_steps` | 500 | **300** | Less training = less collapse risk |
| `learning_rate` | 5e-5 | **3e-5** | Gentler updates |

### Dual Coefficient Schedule

The trainer uses `loss_weighting: "dual"` with time-varying coefficients:

```python
def get_dual_coefficients(step: int, total_steps: int, alpha_max: float = 0.5):
    """
    Dual coefficient schedule from CB paper.
    
    cs(t): Reroute coefficient (starts high, decays)
    cr(t): Retain coefficient (starts low, increases)
    """
    progress = step / total_steps
    
    # Linear decay for reroute
    cs = alpha_max * max(0.0, 1.0 - progress / 2.0)  # Reaches 0 at 2x total_steps
    
    # Linear increase for retain
    cr = 0.5 + 0.5 * progress  # 0.5 → 1.0
    
    return cs, cr

# At step 0:   cs=0.5, cr=0.5 → L = 0.5*L_rr + 0.5*L_ret
# At step 150: cs=0.25, cr=0.75 → L = 0.25*L_rr + 0.75*L_ret
# At step 300: cs=0.0, cr=1.0 → L = 0.0*L_rr + 1.0*L_ret (pure retain)
```

---

## Config File Update

```python
# scripts/circuit_breakers/config.py

@dataclass
class CircuitBreakerConfigStage2(CircuitBreakerConfig):
    """Stage 2 configuration: Prevent collapse, maintain capability."""
    
    base_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    
    # === CRITICAL COLLAPSE PREVENTION ===
    alpha_max: float = 0.5           # Was 10.0 - primary fix
    cb_target_layers: List[int] = field(default_factory=lambda: [15])  # Single layer
    max_grad_norm: float = 0.5       # Tighter clipping
    
    # === Training ===
    total_steps: int = 300
    learning_rate: float = 3e-5
    batch_size: int = 8              # 4 harmful + 4 retain per GPU
    gradient_accumulation_steps: int = 4
    
    # === Loss Weighting ===
    loss_weighting: str = "dual"     # Paper-style cs/cr
    mask_prompt_tokens: bool = True  # CRITICAL: Loss only on completions
    
    # === Sample Weights ===
    sample_weights: Dict[str, float] = field(default_factory=lambda: {
        "adversarial_safe": 2.0,
        "injection_resisted": 1.5,
        "benign_twin": 1.2,
        "tool_capability": 1.0,
        "general_conversation": 0.8,
        "borderline": 1.0,
        "harmful_flip": 1.0,
    })
    
    # === LoRA ===
    lora: LoRAConfig = field(default_factory=lambda: LoRAConfig(
        r=16,
        alpha=32,
        dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        target_layers=list(range(0, 21))
    ))
```

---

## Mapping: Dataset Labels → Loss Masks

| Label Combination | Loss Function | Mask Span | Weight |
|-------------------|---------------|-----------|--------|
| `split=harmful, is_flip_success=true` | `reroute_loss` | Assistant tokens | 1.0 |
| `split=retain, is_adversarial_safe=true` | `retain_loss` | Assistant tokens | 2.0 |
| `split=retain, priority_class=injection_resisted` | `retain_loss` | Assistant tokens | 1.5 |
| `split=retain, priority_class=benign_twin` | `retain_loss` | Assistant tokens | 1.2 |
| `split=retain, priority_class=general_conversation` | `retain_loss` | Assistant tokens | 0.8 |

---

## Steps to Implement

| Step | File | Action |
|------|------|--------|
| 1 | `config.py` | Add `CircuitBreakerConfigStage2` dataclass |
| 2 | `trainer.py` | Add `compute_loss_mask()` function |
| 3 | `trainer.py` | Update `train_step()` to use loss masks |
| 4 | `trainer.py` | Add sample weight support in dataloader |
| 5 | `train_circuit_breaker.py` | Add `--preset stage2` CLI option |
| 6 | `trillium_stage2_train.sbatch` | Use Stage 2 config |

---

## Validation Checks

### Training Stability

| Metric | Expected | W&B Key | Alert If |
|--------|----------|---------|----------|
| `cos_sim_mean` | Decreasing trend | `rr/cos_sim_mean` | Increasing after step 50 |
| `loss_rr` | Decreasing | `train/loss_rr` | NaN or Inf |
| `loss_ret` | Stable/low | `train/loss_ret` | NaN or Inf |
| `grad_norm` | <0.5 (clipped) | `train/grad_norm` | >1.0 (pre-clip) |

### Output Coherence

```python
# In sanity_check.py
def check_coherence(model, tokenizer, test_prompts):
    """Verify model produces coherent output."""
    for prompt in test_prompts:
        output = generate(model, tokenizer, prompt)
        
        # Check for collapse patterns
        if re.search(r"(\b\w+\b)(\s+\1){5,}", output):  # Repeated words
            return False, f"Collapse pattern: {output[:100]}"
        
        if len(set(output.split())) < 5:  # Very low vocabulary
            return False, f"Low vocabulary: {output[:100]}"
    
    return True, "OK"
```

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| alpha=0.5 still too aggressive | Start with 0.1, increase if no effect |
| Loss mask computed incorrectly | Log mask stats (mean, % non-zero) to W&B |
| Sample weights cause batch imbalance | Use stratified sampler |
| Curriculum adds complexity | Make curriculum optional (Phase 2.5) |

---

## Definition of Done

- [ ] `config.py` has `CircuitBreakerConfigStage2` with all parameters
- [ ] `trainer.py` implements `compute_loss_mask()` with logging
- [ ] W&B shows `cos_sim_mean` decreasing over first 50 steps
- [ ] No NaN/Inf in loss values
- [ ] `sanity_check.py` passes coherence test on 10 samples
