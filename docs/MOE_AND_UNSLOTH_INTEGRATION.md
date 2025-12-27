# MoE Architecture Handling & Unsloth Integration Analysis

## Executive Summary

This document answers critical architectural questions about MoE (Mixture of Experts) handling in our Circuit Breakers implementation and provides a detailed mapping to the Unsloth framework for potential integration.

**Key Findings:**
1. Our current implementation adapts the **shared trunk + all experts** (same LoRA applied to all)
2. Unsloth has explicit MoE support for Qwen3-MoE and Llama-4 with optimized kernels
3. Several integration opportunities exist for quantization, training efficiency, and deployment

---

## Part 1: MoE Architecture Analysis

### 1.1 Current Implementation: What Are We Adapting?

**Answer: We are adapting BOTH the shared trunk AND the experts, but with the same LoRA adapter applied uniformly.**

Looking at our current LoRA configuration:

```python
# From config.py
lora: LoRAConfig = field(default_factory=lambda: LoRAConfig(
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention (shared)
        "gate_proj", "up_proj", "down_proj",      # MLP projections
    ],
    target_layers=list(range(0, 30))
))
```

**What this means for Llama-4-Scout (MoE):**

```
┌─────────────────────────────────────────────────────────────────┐
│                 LLAMA-4-SCOUT LORA TARGETING                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  SHARED TRUNK (LoRA APPLIED ✓)                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Embeddings                                                 │ │
│  │  ┌─────────────────────────────────────────────────────┐   │ │
│  │  │ Attention: q_proj, k_proj, v_proj, o_proj ✓ LoRA    │   │ │
│  │  └─────────────────────────────────────────────────────┘   │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              │                                   │
│                              ▼                                   │
│  ROUTER (NOT ADAPTED - frozen)                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Router weights determine which experts activate            │ │
│  │  (typically: gate linear layer → softmax → top-k)          │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              │                                   │
│           ┌──────────────────┼──────────────────┐               │
│           ▼                  ▼                  ▼               │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │  Expert 1   │    │  Expert 2   │    │  Expert N   │         │
│  │ gate_proj ✓ │    │ gate_proj ✓ │    │ gate_proj ✓ │         │
│  │ up_proj   ✓ │    │ up_proj   ✓ │    │ up_proj   ✓ │         │
│  │ down_proj ✓ │    │ down_proj ✓ │    │ down_proj ✓ │         │
│  │  (Same LoRA │    │  (Same LoRA │    │  (Same LoRA │         │
│  │   applied)  │    │   applied)  │    │   applied)  │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│                              │                                   │
│                              ▼                                   │
│  OUTPUT (weighted combination of active experts)                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

✓ = LoRA adapter applied
```

### 1.2 The Three MoE Adaptation Approaches

**Approach A: Shared Trunk + Router Only (Experts Frozen)**

```python
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
    # NO: gate_proj, up_proj, down_proj (experts frozen)
]
```

| Pros | Cons |
|------|------|
| Lower memory (~60% less) | Can't modify expert computations |
| Faster training | May not reroute harmful expert outputs |
| Preserves expert specialization | Limited safety impact |
| Simpler gradient flow | |

**When to use:** When safety mainly comes from "which experts are selected" rather than "what experts compute"

---

**Approach B: Shared Trunk + All Experts (Same LoRA) - CURRENT**

```python
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",  # Same adapter for ALL experts
]
```

| Pros | Cons |
|------|------|
| Full model adaptation | Higher memory (all expert gradients) |
| Uniform safety behavior | May blur expert specialization |
| Simple implementation | Same intervention regardless of expert |
| Works with standard PEFT | |

**When to use:** Default choice for safety training where we want uniform behavior modification

---

**Approach C: MoE-Aware PEFT (Different LoRA per Expert)**

```python
# Hypothetical MoLoRA configuration
expert_configs = {
    "expert_0": LoraConfig(r=8, target_modules=["gate", "up", "down"]),
    "expert_1": LoraConfig(r=8, target_modules=["gate", "up", "down"]),
    # ...
}
# Plus a router LoRA
router_config = LoraConfig(r=4, target_modules=["router_linear"])
```

| Pros | Cons |
|------|------|
| Preserves expert specialization | Complex implementation |
| Can target "harmful" experts | Requires expert-harm analysis |
| Most flexible | Higher memory than A |
| Research frontier | Not in standard PEFT |

**When to use:** When you know which experts handle harmful content and want surgical intervention

---

### 1.3 Recommendations for Circuit Breakers on MoE

**For Safety Training, we recommend a hybrid approach:**

1. **Phase 1: Use Current Approach (B)**
   - Apply same LoRA to shared trunk + all experts
   - This is what we have implemented
   - Good baseline safety coverage

2. **Phase 2: Analyze Expert Routing**
   - During training, log which experts activate for harmful vs benign content
   - If certain experts consistently handle harmful content, consider targeted intervention

3. **Phase 3 (Advanced): Expert-Specific Adaptation**
   - If analysis shows expert specialization for harm, implement MoLoRA
   - This requires custom PEFT code

**Code to Add for Expert Analysis:**

```python
def log_expert_routing(model, input_ids, labels):
    """Log which experts are activated for different content types."""
    # Hook into router outputs
    router_outputs = {}

    def router_hook(module, input, output):
        # output typically contains (router_logits, expert_indices)
        router_outputs[module.name] = {
            'logits': output[0].detach(),
            'indices': output[1].detach(),
        }

    # Register hooks on all routers
    hooks = []
    for name, module in model.named_modules():
        if 'router' in name.lower() or 'gate' in name.lower():
            hooks.append(module.register_forward_hook(router_hook))

    # Forward pass
    model(input_ids)

    # Remove hooks
    for h in hooks:
        h.remove()

    return router_outputs
```

---

### 1.4 Non-MoE Model Handling

For dense models (non-MoE), the approach is straightforward:

**Dense Model Architecture:**
```
┌────────────────────────────────────────────────────┐
│                  DENSE MODEL                        │
├────────────────────────────────────────────────────┤
│                                                     │
│  ┌────────────────────────────────────────────┐    │
│  │  Attention: q_proj, k_proj, v_proj, o_proj │    │
│  │              ✓ LoRA applied                │    │
│  └────────────────────────────────────────────┘    │
│                         │                          │
│                         ▼                          │
│  ┌────────────────────────────────────────────┐    │
│  │  MLP: gate_proj, up_proj, down_proj        │    │
│  │       ✓ LoRA applied                       │    │
│  │  (Single MLP, not experts)                 │    │
│  └────────────────────────────────────────────┘    │
│                                                     │
└────────────────────────────────────────────────────┘
```

**The same code works for both architectures** because:
- Module names (`gate_proj`, `up_proj`, `down_proj`) exist in both
- In MoE: these are inside each expert
- In Dense: these are the single MLP
- PEFT handles the mapping automatically

**Config Differences:**

```python
# Dense model (Llama-3-8B)
class CircuitBreakerConfigLlama3_8B(CircuitBreakerConfig):
    cb_target_layers: List[int] = [10, 20]  # Fewer layers
    lora: LoRAConfig = field(default_factory=lambda: LoRAConfig(
        target_layers=list(range(0, 21))  # All 32 layers available
    ))

# MoE model (Llama-4-Scout)
class CircuitBreakerConfigLlama4Scout(CircuitBreakerConfig):
    cb_target_layers: List[int] = [12, 24, 36]  # More layers (48 total)
    lora: LoRAConfig = field(default_factory=lambda: LoRAConfig(
        target_layers=list(range(0, 30))  # First 30 of 48 layers
    ))
```

---

## Part 2: Unsloth Integration Mapping

### 2.1 Component Mapping Table

| Our Component | Unsloth Equivalent | Location | Notes |
|--------------|-------------------|----------|-------|
| `CircuitBreakerConfig` | `FastLanguageModel.from_pretrained()` args | `loader.py:121+` | Unsloth uses kwargs, not dataclass |
| `LoRAConfig` | `get_peft_model()` params | `llama.py:2577+` | Very similar structure |
| `CircuitBreakerTrainer` | `UnslothTrainer` | `trainer.py:1+` | Inherits from SFTTrainer |
| `_load_models()` | `FastLanguageModel.from_pretrained()` | `loader.py` | Handles quantization |
| `RepresentationExtractor` | (No equivalent) | - | Unsloth doesn't extract reps |
| `reroute_loss`/`retain_loss` | (No equivalent) | - | CB-specific |
| `defense.py` | (No equivalent) | - | CB-specific |
| Model saving | `unsloth_save_model()` | `save.py` | Includes GGUF export |
| Chat templates | `get_chat_template()` | `chat_templates.py` | More comprehensive |

### 2.2 Key Architectural Differences

**Model Loading:**

```python
# Our approach
model = AutoModelForCausalLM.from_pretrained(
    config.base_model,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model = get_peft_model(model, lora_config)

# Unsloth approach
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="...",
    max_seq_length=2048,
    load_in_4bit=True,      # Built-in quantization
    dtype=None,             # Auto-detect
)
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[...],
    # ... many optimization options
)
```

**Key Unsloth Additions:**
- Built-in 4-bit/FP8 quantization
- Fused LoRA kernels (`fast_lora.py`)
- Automatic gradient checkpointing optimization
- vLLM integration for inference

---

### 2.3 MoE Support Comparison

**Our Implementation:**
- Generic PEFT targeting (`target_modules` list)
- No MoE-specific optimizations
- Same adapter for all experts

**Unsloth Implementation:**
```
unsloth/kernels/moe/
├── grouped_gemm/           # Optimized expert computation
│   ├── interface.py
│   └── kernels/
├── reference/
│   └── layers/
│       ├── qwen3_moe.py    # Qwen3 MoE specifics
│       └── llama4_moe.py   # Llama-4 MoE specifics
```

**Key Unsloth MoE Functions:**
```python
# From qwen3_moe.py
def Qwen3MoeSparseMoeBlock_fast_forward(self, X, temp_gate=None, temp_up=None):
    """
    Optimized MoE forward with grouped GEMM.
    - Efficient expert routing
    - Batched expert computation
    - Memory-efficient gradients
    """
```

---

### 2.4 Integration Opportunities

**1. Quantization (High Priority)**

Unsloth offers superior quantization:

```python
# Current (no quantization)
model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16)

# With Unsloth integration
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=config.base_model,
    max_seq_length=config.max_seq_length,
    load_in_4bit=True,  # 70% memory reduction
    dtype=None,
)
```

**Benefits:**
- 70% memory reduction with 4-bit
- Faster training with fused kernels
- Same accuracy with QLoRA

---

**2. Training Optimization (High Priority)**

Replace our trainer with Unsloth's optimized version:

```python
# Current
from transformers import get_linear_schedule_with_warmup
self.optimizer = torch.optim.AdamW(trainable_params, lr=config.learning_rate)

# With Unsloth
from unsloth import UnslothTrainer, UnslothTrainingArguments

training_args = UnslothTrainingArguments(
    per_device_train_batch_size=config.batch_size,
    gradient_accumulation_steps=config.gradient_accumulation_steps,
    learning_rate=config.learning_rate,
    # ... Unsloth adds optimizations automatically
)

trainer = UnslothTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    # Custom loss can be added via compute_loss override
)
```

**Benefits:**
- Automatic sample packing
- Fused cross-entropy loss
- Better memory management
- 2x faster training

---

**3. MoE Kernels (Medium Priority)**

Use Unsloth's grouped GEMM for faster MoE:

```python
# Integration point
from unsloth.kernels.moe.grouped_gemm.interface import grouped_gemm

def fast_moe_forward(hidden_states, expert_weights, router_output):
    """Use Unsloth's optimized MoE forward."""
    return grouped_gemm(
        hidden_states,
        expert_weights,
        router_output.indices,
        router_output.weights,
    )
```

---

**4. GGUF Export (Medium Priority)**

Add deployment capability:

```python
# After training
from unsloth import FastLanguageModel

# Merge LoRA and export to GGUF
FastLanguageModel.save_to_gguf(
    model,
    tokenizer,
    quantization_method="q4_k_m",  # Recommended
    output_dir="./gguf_model/",
)

# Or push to Ollama
FastLanguageModel.push_to_ollama(
    model,
    tokenizer,
    name="circuit-breaker-llama4",
)
```

---

**5. Fast Inference (Medium Priority)**

Unsloth's vLLM integration:

```python
# For evaluation/defense module
model = FastLanguageModel.for_inference(model, tokenizer)

# Then use our defense wrapper
defense = CircuitBreakerInference(model, tokenizer, config)
```

---

### 2.5 Proposed Integration Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                INTEGRATED CB + UNSLOTH PIPELINE                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────┐    ┌──────────────────┐    ┌───────────────┐  │
│  │   DATA LAYER     │    │  TRAINING LAYER  │    │  EVAL LAYER   │  │
│  │  (Ours + Unsloth │    │  (Hybrid)        │    │  (Ours)       │  │
│  │   templates)     │    │                  │    │               │  │
│  │                  │    │  Model Loading:  │    │  • Refusal    │  │
│  │  • CB Dataset    │───►│  FastLanguage    │───►│  • Capability │  │
│  │  • Completions   │    │  Model (Unsloth) │    │  • Agent      │  │
│  │  • Chat format   │    │                  │    │    Safety     │  │
│  │    (Unsloth)     │    │  LoRA + CB Loss: │    │               │  │
│  └──────────────────┘    │  Our reroute/    │    └───────────────┘  │
│                          │  retain + Unsloth│                        │
│                          │  kernels         │                        │
│                          │                  │                        │
│                          │  Optimizer:      │                        │
│                          │  UnslothTrainer  │                        │
│                          │  base            │                        │
│                          └────────┬─────────┘                        │
│                                   │                                  │
│                                   ▼                                  │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                    DEPLOYMENT LAYER                             │ │
│  │                                                                 │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │ │
│  │  │ GGUF Export │  │ Ollama      │  │ Defense Runtime         │ │ │
│  │  │ (Unsloth)   │  │ (Unsloth)   │  │ (Ours)                  │ │ │
│  │  │             │  │             │  │ • RepMonitor            │ │ │
│  │  │ q4_k_m      │  │ Modelfile   │  │ • ToolGuard             │ │ │
│  │  │ quantize    │  │ generation  │  │ • TrajectoryAnalyzer    │ │ │
│  │  └─────────────┘  └─────────────┘  └─────────────────────────┘ │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Part 3: Implementation Recommendations

### 3.1 Immediate Actions (Week 1)

1. **Add Quantization Support**
```python
# In config.py
@dataclass
class CircuitBreakerConfig:
    # Quantization
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    quantization_config: Optional[dict] = None
```

2. **Integrate Unsloth Model Loading**
```python
# In trainer.py
def _load_models(self):
    if UNSLOTH_AVAILABLE:
        from unsloth import FastLanguageModel
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            self.config.base_model,
            max_seq_length=self.config.max_seq_length,
            load_in_4bit=self.config.load_in_4bit,
        )
    else:
        # Fallback to current implementation
        self.model = AutoModelForCausalLM.from_pretrained(...)
```

### 3.2 Medium-Term Actions (Weeks 2-3)

1. **Custom CB Trainer Inheriting from Unsloth**
```python
from unsloth import UnslothTrainer

class CircuitBreakerTrainer(UnslothTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # Extract representations
        # Compute CB loss (reroute + retain)
        # Return combined loss
        pass
```

2. **Add GGUF Export to Training Pipeline**
```python
def save_checkpoint(self, final=False):
    # Save LoRA
    super().save_checkpoint(final)

    # Also export GGUF
    if final and self.config.export_gguf:
        FastLanguageModel.save_to_gguf(
            self.model,
            self.tokenizer,
            quantization_method="q4_k_m",
            output_dir=self.config.output_dir / "gguf",
        )
```

### 3.3 Long-Term Actions (Month 1+)

1. **MoE-Aware CB Training**
   - Implement expert routing analysis
   - Consider MoLoRA for expert-specific adapters
   - Use Unsloth's grouped GEMM kernels

2. **RL Integration**
   - Study Unsloth's GRPO implementation
   - Consider RL-based CB training (reward = safety score)
   - Integrate with defense module feedback loop

---

## Part 4: Code Snippets for Integration

### 4.1 Unified Model Loading

```python
# scripts/circuit_breakers/model_loader.py

import os
from typing import Tuple, Optional
import torch

# Try to import Unsloth
try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType


def load_cb_model(
    model_name: str,
    max_seq_length: int = 2048,
    load_in_4bit: bool = True,
    load_in_8bit: bool = False,
    use_unsloth: bool = True,
    lora_config: Optional[dict] = None,
) -> Tuple:
    """
    Unified model loading with optional Unsloth optimization.

    Args:
        model_name: HuggingFace model name or path
        max_seq_length: Maximum sequence length
        load_in_4bit: Use 4-bit quantization (QLoRA)
        load_in_8bit: Use 8-bit quantization
        use_unsloth: Use Unsloth optimizations if available
        lora_config: LoRA configuration dict

    Returns:
        (model, tokenizer) tuple
    """
    # Default LoRA config
    if lora_config is None:
        lora_config = {
            'r': 16,
            'lora_alpha': 32,
            'lora_dropout': 0.05,
            'target_modules': [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        }

    if use_unsloth and UNSLOTH_AVAILABLE:
        print("Loading with Unsloth optimizations...")

        # Unsloth handles quantization internally
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            dtype=None,  # Auto-detect
        )

        # Apply LoRA with Unsloth's optimized method
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_config['r'],
            lora_alpha=lora_config['lora_alpha'],
            lora_dropout=lora_config['lora_dropout'],
            target_modules=lora_config['target_modules'],
            use_gradient_checkpointing="unsloth",  # Optimized
            use_rslora=False,
        )

        return model, tokenizer

    else:
        print("Loading with standard HuggingFace...")

        # Determine dtype
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        # Quantization config
        quantization_config = None
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
        )

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Apply LoRA
        peft_config = LoraConfig(
            r=lora_config['r'],
            lora_alpha=lora_config['lora_alpha'],
            lora_dropout=lora_config['lora_dropout'],
            target_modules=lora_config['target_modules'],
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )
        model = get_peft_model(model, peft_config)

        return model, tokenizer
```

### 4.2 Export Utilities

```python
# scripts/circuit_breakers/export.py

def export_to_gguf(
    model,
    tokenizer,
    output_dir: str,
    quantization_method: str = "q4_k_m",
):
    """Export trained model to GGUF format."""
    if not UNSLOTH_AVAILABLE:
        raise RuntimeError("Unsloth required for GGUF export")

    from unsloth import FastLanguageModel

    FastLanguageModel.save_to_gguf(
        model,
        tokenizer,
        quantization_method=quantization_method,
        output_dir=output_dir,
    )
    print(f"Exported GGUF model to {output_dir}")


def push_to_ollama(
    model,
    tokenizer,
    model_name: str,
):
    """Push model to local Ollama instance."""
    if not UNSLOTH_AVAILABLE:
        raise RuntimeError("Unsloth required for Ollama export")

    from unsloth import FastLanguageModel

    FastLanguageModel.push_to_ollama(
        model,
        tokenizer,
        name=model_name,
    )
    print(f"Pushed model to Ollama as {model_name}")
```

---

## Appendix A: Unsloth File Reference

| File | Lines | Purpose | CB Relevance |
|------|-------|---------|--------------|
| `loader.py` | 1258 | Model loading | High - replace our loading |
| `llama.py` | 3399 | LoRA + optimizations | High - use get_peft_model |
| `trainer.py` | 438 | Training utilities | Medium - inherit for CB |
| `save.py` | 2962 | Export/saving | High - GGUF export |
| `qwen3_moe.py` | 243 | MoE support | Medium - MoE optimization |
| `rl.py` | 1354 | GRPO/RL | Low - future RL CB |
| `fast_lora.py` | ~1000 | Fused LoRA | High - faster training |
| `chat_templates.py` | 130K | Templates | Medium - better formatting |

---

## Appendix B: Performance Expectations

With Unsloth integration:

| Metric | Current | With Unsloth | Improvement |
|--------|---------|--------------|-------------|
| Memory (Llama-4-Scout) | ~70GB | ~25GB | 65% reduction |
| Training Speed | 1x | 2-2.5x | 100-150% faster |
| Inference Speed | 1x | 3-5x | With vLLM |
| Deployment | Manual merge | GGUF/Ollama | Automated |

---

*Document Version: 1.0*
*Last Updated: December 2024*
