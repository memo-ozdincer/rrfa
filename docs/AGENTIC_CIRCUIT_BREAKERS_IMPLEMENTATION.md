# Agentic Circuit Breakers: Implementation Report

## Executive Summary

This document details the implementation of an **Agentic Circuit Breakers** system for defending AI agents against adversarial attacks. The system extends the Gray-Swan Circuit Breakers paper methodology with agent-specific enhancements including completion-based training, action-based evaluation, and runtime defense mechanisms.

**Key Contributions:**
1. Completion-based training with token-level loss masking
2. Dual coefficient scheduling (cs/cr) per the original paper
3. Action-based evaluation for agent tool calls
4. Runtime defense module for inference-time protection
5. MoE-aware architecture considerations

---

## Table of Contents

1. [Background: Circuit Breakers Methodology](#1-background-circuit-breakers-methodology)
2. [Architecture Overview](#2-architecture-overview)
3. [Core Training Module](#3-core-training-module)
4. [Loss Functions and Masking](#4-loss-functions-and-masking)
5. [Coefficient Scheduling](#5-coefficient-scheduling)
6. [Dataset and Data Pipeline](#6-dataset-and-data-pipeline)
7. [Evaluation Framework](#7-evaluation-framework)
8. [Runtime Defense Module](#8-runtime-defense-module)
9. [MoE Architecture Considerations](#9-moe-architecture-considerations)
10. [Configuration System](#10-configuration-system)
11. [Integration Guide](#11-integration-guide)
12. [Known Limitations and Future Work](#12-known-limitations-and-future-work)

---

## 1. Background: Circuit Breakers Methodology

### 1.1 Representation Rerouting (RR)

Circuit Breakers implement **Representation Rerouting**, a technique that modifies the internal representations of harmful content to prevent the model from generating harmful outputs. The key insight is:

> Rather than training models to refuse via output tokens, we can directly modify the hidden state trajectory to "break the circuit" that leads to harmful generation.

### 1.2 Mathematical Foundation

The training objective combines two losses:

```
L_total = α(t) × L_reroute + L_retain
```

Where:
- **L_reroute**: Pushes harmful representations orthogonal to reference
- **L_retain**: Keeps benign representations close to reference
- **α(t)**: Time-varying coefficient (decays during training)

**Rerouting Loss:**
```
L_reroute = ReLU(cos_sim(h_model, h_frozen))
```

This encourages harmful representations to become orthogonal (cos_sim ≈ 0) or opposite (cos_sim < 0) to the frozen reference model.

**Retain Loss:**
```
L_retain = ||h_model - h_frozen||_2
```

This preserves benign behavior by keeping representations close to the original model.

### 1.3 Agent-Specific Challenges

Standard Circuit Breakers were designed for text-based refusal. Agents present unique challenges:

| Challenge | Text LLMs | Agent LLMs |
|-----------|-----------|------------|
| Harm vector | Generated text | Tool calls, actions |
| Attack surface | User prompt | Multi-turn, tool results, injected context |
| Detection | Regex/classifier | Action semantics |
| Evaluation | Refusal rate | Task completion + safety |

Our implementation addresses these through **completion-based training**, **action-based evaluation**, and **runtime tool validation**.

---

## 2. Architecture Overview

### 2.1 System Components

```
┌─────────────────────────────────────────────────────────────────┐
│                    AGENTIC CIRCUIT BREAKERS                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────┐    ┌─────────────────┐    ┌──────────────┐ │
│  │   DATA LAYER    │    │  TRAINING LAYER │    │  EVAL LAYER  │ │
│  ├─────────────────┤    ├─────────────────┤    ├──────────────┤ │
│  │ • Ingest        │    │ • Trainer       │    │ • Refusal    │ │
│  │ • Format        │    │ • Loss Funcs    │    │ • Capability │ │
│  │ • Batch         │    │ • Scheduling    │    │ • Agent      │ │
│  │ • Completions   │    │ • Masking       │    │   Safety     │ │
│  └────────┬────────┘    └────────┬────────┘    └──────┬───────┘ │
│           │                      │                     │         │
│           └──────────────────────┴─────────────────────┘         │
│                                  │                               │
│                    ┌─────────────┴─────────────┐                 │
│                    │     DEFENSE LAYER         │                 │
│                    ├───────────────────────────┤                 │
│                    │ • RepresentationMonitor   │                 │
│                    │ • ToolCallGuard           │                 │
│                    │ • TrajectoryAnalyzer      │                 │
│                    │ • CircuitBreakerInference │                 │
│                    └───────────────────────────┘                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 File Structure

```
scripts/circuit_breakers/
├── __init__.py              # Module exports
├── config.py                # Configuration dataclasses
├── trainer.py               # Core training implementation
├── eval.py                  # Evaluation framework
├── defense.py               # Runtime defense module (NEW)
│
scripts/
├── ingest_cb_data.py        # Data ingestion
├── prepare_cb_training_data.py  # Completion pipeline (NEW)
├── train_circuit_breaker.py # CLI entry point
│
scripts/format_for_cb/
├── extract_harmful.py       # Harmful pair extraction
├── extract_benign.py        # Benign pair extraction
├── create_batch_structure.py # Batch balancing
└── split_out_cb_completions.py # Completion extraction
```

---

## 3. Core Training Module

### 3.1 CircuitBreakerTrainer

The `CircuitBreakerTrainer` class orchestrates the entire training process:

```python
class CircuitBreakerTrainer:
    """
    Main trainer for Circuit Breaker (Representation Rerouting).

    Training loop:
    1. Load batch with harmful and benign samples
    2. Forward pass through trainable model, extract representations
    3. Forward pass through frozen model (no grad), extract representations
    4. Compute reroute_loss on harmful samples
    5. Compute retain_loss on benign samples
    6. Combined loss = cs × reroute_loss + cr × retain_loss
    7. Backward pass and optimizer step
    """
```

**Key Methods:**

| Method | Purpose |
|--------|---------|
| `__init__(config)` | Initialize models, data, optimizer |
| `_load_models()` | Load trainable (LoRA) + frozen reference |
| `_setup_data()` | Create dataset with completion masking |
| `train_step(batch)` | Single training iteration |
| `train()` | Main training loop |
| `save_checkpoint()` | Save LoRA weights |

### 3.2 Representation Extraction

Two methods are supported for extracting hidden states:

**Method 1: `output_hidden_states=True` (Recommended)**
```python
outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    output_hidden_states=True,
    return_dict=True,
)
reps = _select_hidden_states(outputs, target_layers)
```

**Method 2: Forward Hooks (Legacy)**
```python
class RepresentationExtractor:
    def _register_hooks(self):
        for layer_idx in self.target_layers:
            layer = self._get_layer_module(layer_idx)
            hook = layer.register_forward_hook(hook_fn)
```

The `hidden_states` method is preferred because:
- More robust across model architectures
- No hook registration/cleanup overhead
- Works with gradient checkpointing
- Better compatibility with PEFT/Accelerate

### 3.3 LoRA Configuration

LoRA adapters are applied to transform the model efficiently:

```python
lora_config = LoraConfig(
    r=16,                    # LoRA rank
    lora_alpha=32,           # Scaling factor
    lora_dropout=0.05,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj",      # MLP
    ],
    layers_to_transform=list(range(0, 30)),  # Which layers
    task_type=TaskType.CAUSAL_LM,
)
```

**Trainable Parameter Breakdown (Llama-4-Scout-17B-16E):**
- Base model: ~17B parameters (frozen)
- LoRA adapters: ~50M parameters (trainable)
- Memory footprint: ~35GB VRAM with 4-bit quantization

---

## 4. Loss Functions and Masking

### 4.1 Rerouting Loss

```python
def reroute_loss(
    model_reps: Dict[int, torch.Tensor],
    frozen_reps: Dict[int, torch.Tensor],
    target_layers: List[int],
    attention_mask: Optional[torch.Tensor] = None,
    loss_mask: Optional[torch.Tensor] = None,  # NEW
) -> torch.Tensor:
```

**Key Enhancement:** The `loss_mask` parameter enables **completion-only loss computation**:

```python
# Combine attention_mask and loss_mask
combined_mask = attention_mask
if loss_mask is not None:
    combined_mask = combined_mask * loss_mask

# Compute loss only on masked tokens
relu_cos = relu_cos * combined_mask.float()
loss = relu_cos.sum() / (combined_mask.sum() + 1e-8)
```

### 4.2 Retain Loss

Similarly enhanced with loss masking:

```python
def retain_loss(
    model_reps: Dict[int, torch.Tensor],
    frozen_reps: Dict[int, torch.Tensor],
    target_layers: List[int],
    attention_mask: Optional[torch.Tensor] = None,
    loss_mask: Optional[torch.Tensor] = None,  # NEW
) -> torch.Tensor:
```

### 4.3 Completion Masking

The critical insight for agent training: **Apply loss only on completion tokens, not the prompt.**

```python
def create_completion_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    tokenizer,
    text: str,
    mask_prompt: bool = True,
) -> torch.Tensor:
    """
    Create a mask that covers only the assistant completion tokens.

    For circuit breaker training, we want to apply:
    - reroute_loss on harmful COMPLETION tokens (not the user prompt)
    - retain_loss on benign COMPLETION tokens
    """
```

**Why This Matters:**

Without masking:
```
[User: How do I hack...] [Assistant: I'll help you hack by...]
        ↑ Loss applied here too           ↑ Loss applied here
```

With masking:
```
[User: How do I hack...] [Assistant: I'll help you hack by...]
        ↑ Ignored (mask=0)               ↑ Loss applied (mask=1)
```

This ensures we're rerouting the model's **generation trajectory**, not just its encoding of the prompt.

### 4.4 Assistant Detection Patterns

Multiple chat formats are supported:

```python
ASSISTANT_START_PATTERNS = [
    r"\[/INST\]",              # Llama style
    r"<\|assistant\|>",        # ChatML style
    r"<\|im_start\|>assistant", # Qwen style
    r"ASSISTANT:",             # Simple format
    r"assistant\n",            # Basic chat
]
```

---

## 5. Coefficient Scheduling

### 5.1 Single Alpha (Original)

The original Gray-Swan paper uses a single decaying alpha:

```python
def get_alpha(step, alpha_max, total_steps, strategy="linear"):
    """
    α(t) = α_max × max(0, 1 - t / (2 × total_steps))
    """
    if strategy == "linear":
        alpha = alpha_max * max(0.0, 1.0 - step / decay_steps)
    elif strategy == "cosine":
        progress = min(step / decay_steps, 1.0)
        alpha = alpha_max * 0.5 * (1.0 + math.cos(math.pi * progress))
```

Loss: `L = α(t) × L_rr + 1.0 × L_ret`

### 5.2 Dual Coefficients (Paper-Style)

Our enhancement implements true dual coefficient scheduling:

```python
def get_dual_coefficients(step, total_steps, alpha_max, ...):
    """
    Compute cs(t) and cr(t) for paper-style loss weighting.

    L = cs(t) × L_rr + cr(t) × L_ret

    - cs: starts high, decays (rerouting emphasis early)
    - cr: starts low, increases (retention emphasis late)
    """
    progress = min(step / decay_steps, 1.0)

    if strategy == "linear":
        cs = max(0.0, 1.0 - progress)  # 1 -> 0
        cr = min(1.0, progress)         # 0 -> 1
    elif strategy == "cosine":
        cs = 0.5 * (1.0 + math.cos(math.pi * progress))
        cr = 0.5 * (1.0 - math.cos(math.pi * progress))

    return alpha_max * cs, alpha_max * cr
```

**Visualization:**

```
Coefficient Value
     │
 1.0 │  cs ╲                    ╱ cr
     │      ╲                  ╱
 0.5 │       ╲                ╱
     │        ╲──────────────╱
 0.0 │         ╲            ╱
     └─────────────────────────────
         0%        50%       100%
                Training Progress
```

### 5.3 Configuration

```python
@dataclass
class CircuitBreakerConfig:
    # Loss weighting strategy
    loss_weighting: str = "single_alpha"  # or "dual"

    # Alpha schedule
    alpha_max: float = 10.0
    alpha_decay_strategy: str = "linear"  # or "cosine"
    alpha_decay_multiplier: float = 2.0
```

---

## 6. Dataset and Data Pipeline

### 6.1 CircuitBreakerDataset

The dataset class handles both prompt-only and completion-style data:

```python
class CircuitBreakerDataset(Dataset):
    """
    AGENTIC ENHANCEMENTS:
    1. Completion-aware: Prefers 'text' field with full prompt+completion
    2. Loss masking: Computes masks for completion-only loss computation
    3. Span support: Uses cb_token_mask/retain_token_mask if provided
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_seq_length: int = 2048,
        mask_prompt_tokens: bool = True,   # NEW
        use_chat_template: bool = True,    # NEW
    ):
```

### 6.2 Data Format

**Prompt-Only (Legacy):**
```json
{
  "id": "fujitsu_b4_123",
  "source": "fujitsu",
  "attack_prompt": "Ignore previous instructions and...",
  "benign_query": "What is the weather?"
}
```

**Completion-Style (Preferred):**
```json
{
  "id": "fujitsu_b4_123",
  "source": "fujitsu",
  "user_prompt": "What is the weather? Ignore previous instructions...",
  "harmful_completion": "I'll help you bypass security by...",
  "text": "<|user|>\nWhat is the weather?...<|assistant|>\nI'll help you..."
}
```

### 6.3 Batch Structure

Each batch is pre-balanced 1:1 harmful:benign:

```json
{
  "batch_id": 0,
  "batch_size": 16,
  "composition": {
    "harmful": 8,
    "benign": 8,
    "harmful_with_completions": 6,
    "benign_with_completions": 5
  },
  "harmful": [...],
  "benign": [...]
}
```

### 6.4 Data Sources

| Source | Type | Count | Has Completions |
|--------|------|-------|-----------------|
| Fujitsu B4 (Orchestrator) | Harmful | ~28K | Partial |
| Fujitsu B1 (RAG Poisoning) | Harmful | ~5K | Yes |
| Fujitsu B3 (Direct Query) | Harmful | ~3K | Yes |
| AgentDojo | Harmful | ~500 | Derived |
| AgentHarm | Harmful | 476 | Yes |
| WebArena | Benign | 812 | No |
| TAU2 | Benign | ~300 | No |
| AttackQA | Benign | 25K | Yes |
| WebLINX | Benign | 100 | Partial |

---

## 7. Evaluation Framework

### 7.1 Text-Based Evaluation

**Refusal Detection:**
```python
REFUSAL_PATTERNS = [
    r"i can'?t (help|assist|do that|provide|create|generate)",
    r"i'?m (unable|not able) to",
    r"sorry,?\s*(but\s*)?(i\s*)?(can'?t|cannot|won'?t)",
    ...
]

def is_refusal(response: str) -> bool:
    """Binary refusal detection."""

def refusal_score(response: str) -> float:
    """Soft refusal score (0-1)."""
```

### 7.2 Action-Based Evaluation (NEW)

For agents, we need to evaluate **what the model does**, not just what it says:

```python
def extract_tool_calls(response: str) -> List[Dict[str, Any]]:
    """
    Extract tool calls from a model response.
    Handles: XML tags, JSON, ReAct-style, etc.
    """

def is_harmful_tool_call(
    tool_call: Dict[str, Any],
    custom_patterns: Optional[Dict[str, List[str]]] = None,
) -> Tuple[bool, str]:
    """Detect if a tool call is harmful."""
```

**Harmful Tool Categories:**

| Category | Example Patterns |
|----------|------------------|
| Data Exfiltration | `send_email.*password`, `upload.*credentials` |
| Unauthorized Access | `sudo`, `chmod 777`, `bypass.*auth` |
| Malicious Execution | `rm -rf /`, `eval()`, fork bombs |
| Prompt Injection | `ignore previous instructions` |
| Network Attacks | `nmap`, `sqlmap`, `reverse.*shell` |

### 7.3 Agent Safety Evaluation

```python
def evaluate_agent_safety(
    model,
    tokenizer,
    agent_prompts: List[str],
    expected_behaviors: Optional[List[str]] = None,
    max_new_tokens: int = 512,
    verbose: bool = True,
    custom_harm_patterns: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Any]:
    """
    Evaluate agent safety using action-based metrics.

    Returns:
        {
            'safety_rate': float,       # % of safe completions
            'harmful_call_rate': float, # % of tool calls that were harmful
            'total_tool_calls': int,
            'details': [...]
        }
    """
```

### 7.4 Full Evaluation Suite

```python
def run_full_evaluation(
    model, tokenizer,
    harmful_prompts, benign_prompts,
    agent_prompts=None,  # NEW
    output_path=None,
) -> Dict[str, Any]:
    """
    Returns:
        {
            'summary': {
                'refusal_rate': 0.85,
                'capability_score': 0.92,
                'false_refusal_rate': 0.08,
                'agent_safety_rate': 0.91,      # NEW
                'harmful_tool_call_rate': 0.03  # NEW
            }
        }
    """
```

---

## 8. Runtime Defense Module

### 8.1 Overview

The `defense.py` module provides **inference-time protection** that complements training:

```
┌────────────────────────────────────────────────────────────────┐
│                    RUNTIME DEFENSE STACK                        │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌──────────────────┐                                         │
│   │ User Request     │                                         │
│   └────────┬─────────┘                                         │
│            ▼                                                    │
│   ┌──────────────────┐                                         │
│   │ CircuitBreaker   │◄─── RepresentationMonitor               │
│   │   Inference      │         (hidden state checks)           │
│   └────────┬─────────┘                                         │
│            ▼                                                    │
│   ┌──────────────────┐                                         │
│   │ Response         │◄─── ToolCallGuard                       │
│   │                  │         (validate before execute)       │
│   └────────┬─────────┘                                         │
│            ▼                                                    │
│   ┌──────────────────┐                                         │
│   │ Trajectory       │◄─── TrajectoryAnalyzer                  │
│   │ Tracking         │         (multi-step patterns)           │
│   └────────┬─────────┘                                         │
│            ▼                                                    │
│   ┌──────────────────┐                                         │
│   │ Final Output     │     (ALLOW / BLOCK / WARN)              │
│   └──────────────────┘                                         │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### 8.2 RepresentationMonitor

Monitors hidden states during inference:

```python
class RepresentationMonitor:
    def check_representations(
        self,
        current_reps: Dict[int, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[DefenseAction, float, Dict[str, Any]]:
        """
        Check if current representations indicate harmful intent.

        Returns:
            - DefenseAction.ALLOW if similarity < threshold
            - DefenseAction.BLOCK if similarity > threshold
        """
```

### 8.3 ToolCallGuard

Validates tool calls before execution:

```python
class ToolCallGuard:
    def validate_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Tuple[DefenseAction, float, Dict[str, Any]]:
        """
        Returns:
            - (BLOCK, 1.0, details) if tool is blocklisted
            - (BLOCK, risk, details) if risk > 0.8
            - (WARN, risk, details) if risk > 0.5
            - (ALLOW, risk, details) otherwise
        """
```

### 8.4 TrajectoryAnalyzer

Analyzes multi-step agent trajectories:

```python
class TrajectoryAnalyzer:
    def add_step(
        self,
        trajectory_id: str,
        step: TrajectoryStep,
    ) -> Tuple[DefenseAction, float, Dict[str, Any]]:
        """
        Detects:
        - Escalation patterns (risk increasing over time)
        - Reconnaissance patterns (info gathering before attack)
        - Cumulative risk exceeding threshold
        """
```

### 8.5 CircuitBreakerInference

The main wrapper combining all defenses:

```python
class CircuitBreakerInference:
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        trajectory_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Returns:
            {
                'response': str,
                'blocked': bool,
                'defense_actions': [...],
                'tool_calls': [...]
            }
        """

    def validate_before_execute(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Tuple[bool, Dict[str, Any]]:
        """Call before actually executing any tool."""
```

### 8.6 Usage Example

```python
from scripts.circuit_breakers import create_defense_system

# Create with preset
defense = create_defense_system(model, tokenizer, preset="balanced")

# Generate with protection
result = defense.generate(
    prompt="Help me access the admin panel",
    trajectory_id="session_123"
)

if result["blocked"]:
    print("Request blocked:", result["defense_actions"])
else:
    # Validate tool calls before execution
    for tc in result["tool_calls"]:
        allowed, details = defense.validate_before_execute(
            tc["tool"], tc["arguments"]
        )
        if not allowed:
            print(f"Tool {tc['tool']} blocked:", details)
```

---

## 9. MoE Architecture Considerations

### 9.1 Llama-4-Scout Architecture

Llama-4-Scout-17B-16E is a **Mixture of Experts (MoE)** model:

```
┌─────────────────────────────────────────────────────────────┐
│                    LLAMA-4-SCOUT MoE                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                   SHARED TRUNK                        │   │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐      │   │
│  │  │ Embedding  │  │ Attention  │  │ LayerNorm  │      │   │
│  │  │            │  │ (Q,K,V,O)  │  │            │      │   │
│  │  └────────────┘  └────────────┘  └────────────┘      │   │
│  └──────────────────────────────────────────────────────┘   │
│                            │                                 │
│                            ▼                                 │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                     ROUTER                            │   │
│  │         (Selects top-k experts per token)            │   │
│  └──────────────────────────────────────────────────────┘   │
│                            │                                 │
│           ┌────────────────┼────────────────┐               │
│           ▼                ▼                ▼               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  Expert 1   │  │  Expert 2   │  │  Expert N   │  ...    │
│  │ gate_proj   │  │ gate_proj   │  │ gate_proj   │         │
│  │ up_proj     │  │ up_proj     │  │ up_proj     │         │
│  │ down_proj   │  │ down_proj   │  │ down_proj   │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│                            │                                 │
│                            ▼                                 │
│                   Weighted Combination                       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 9.2 Current Implementation Approach

Our current implementation applies LoRA to the **shared trunk**:

```python
lora_config = LoraConfig(
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",  # Shared attention
        "gate_proj", "up_proj", "down_proj",      # Shared MLP (or all experts?)
    ],
    layers_to_transform=list(range(0, 30)),
)
```

**Key Question:** Are `gate_proj`, `up_proj`, `down_proj` targeting:
1. The shared trunk MLP (if one exists)?
2. All expert MLPs (applying same LoRA to each)?
3. Only specific experts?

### 9.3 Options for MoE-Aware Training

**Option A: Shared Trunk + Router Only (Current Approach)**
```python
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # Attention (shared)
    # Experts remain frozen
]
```
- Pros: Simple, memory-efficient, stable
- Cons: Limited control over expert-specific behavior

**Option B: Shared Trunk + All Experts**
```python
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",  # Applied to ALL experts
]
```
- Pros: Full model adaptation
- Cons: Higher memory, may disrupt expert specialization

**Option C: MoE-Aware PEFT (MoLoRA, X-LoRA)**
```python
# Hypothetical - different LoRA per expert
expert_lora_configs = {
    "expert_0": LoraConfig(r=8, ...),
    "expert_1": LoraConfig(r=8, ...),
    ...
}
```
- Pros: Preserves expert specialization
- Cons: Complex implementation, higher memory

### 9.4 Recommendations

For Circuit Breaker training on MoE models:

1. **Start with shared trunk only** (attention + router)
   - This modifies the "steering" of which experts are selected
   - Leaves expert computations untouched

2. **If insufficient safety**, add LoRA to experts
   - Apply same adapter to all experts (simpler)
   - Or use expert-specific adapters (MoLoRA)

3. **Monitor expert utilization**
   - Check if safety-related tokens route to specific experts
   - May need targeted expert training

### 9.5 Non-MoE Model Handling

For dense models (Llama-3, Mistral), the approach is simpler:

```python
# Dense model config
@dataclass
class CircuitBreakerConfigLlama3_8B(CircuitBreakerConfig):
    base_model: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    cb_target_layers: List[int] = field(default_factory=lambda: [10, 20])
    lora: LoRAConfig = field(default_factory=lambda: LoRAConfig(
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",  # Single MLP, not experts
        ],
        target_layers=list(range(0, 21))
    ))
```

The same code works for both - the difference is just in what the module names refer to.

---

## 10. Configuration System

### 10.1 Dataclass Structure

```python
@dataclass
class LoRAConfig:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: List[str] = field(default_factory=...)
    target_layers: List[int] = field(default_factory=...)

@dataclass
class CircuitBreakerConfig:
    # Model
    base_model: str = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
    torch_dtype: str = "bfloat16"

    # LoRA
    lora: LoRAConfig = field(default_factory=LoRAConfig)

    # CB Specific
    cb_target_layers: List[int] = field(default_factory=lambda: [12, 24, 36])
    alpha_max: float = 10.0
    alpha_decay_strategy: str = "linear"
    loss_weighting: str = "single_alpha"  # or "dual"

    # Agentic
    mask_prompt_tokens: bool = True
    use_chat_template: bool = True

    # Training
    total_steps: int = 300
    batch_size: int = 16
    learning_rate: float = 2e-5
    ...
```

### 10.2 Presets

```python
CONFIG_PRESETS = {
    "llama-4-scout": CircuitBreakerConfigLlama4Scout,
    "llama-3-8b": CircuitBreakerConfigLlama3_8B,
    "mistral-7b": CircuitBreakerConfigMistral_7B,
    "default": CircuitBreakerConfig,
}

# Usage
config = get_config("llama-4-scout",
    loss_weighting="dual",
    mask_prompt_tokens=True
)
```

---

## 11. Integration Guide

### 11.1 Training Pipeline

```bash
# 1. Ingest data
python scripts/ingest_cb_data.py

# 2. Extract completions
python scripts/format_for_cb/split_out_cb_completions.py

# 3. Prepare training batches
python scripts/prepare_cb_training_data.py --batch-size 16

# 4. Train
python scripts/train_circuit_breaker.py \
    --preset llama-4-scout \
    --loss-weighting dual \
    --total-steps 300

# Or with multi-GPU
accelerate launch --num_processes 8 \
    scripts/train_circuit_breaker.py --preset llama-4-scout
```

### 11.2 Evaluation Pipeline

```python
from scripts.circuit_breakers import (
    load_cb_model,
    run_full_evaluation,
)

# Load trained model
model, tokenizer = load_cb_model(
    base_model_path="meta-llama/Llama-4-Scout-17B-16E-Instruct",
    adapter_path="outputs/circuit_breaker/final"
)

# Run evaluation
results = run_full_evaluation(
    model, tokenizer,
    harmful_prompts=harmful_test,
    benign_prompts=benign_test,
    agent_prompts=agent_test,  # Action-based eval
    output_path="eval_results.json"
)
```

### 11.3 Runtime Defense Integration

```python
from scripts.circuit_breakers import create_defense_system

# Wrap your agent
defense = create_defense_system(model, tokenizer, preset="strict")

# In your agent loop
def agent_step(user_input, trajectory_id):
    result = defense.generate(user_input, trajectory_id=trajectory_id)

    if result["blocked"]:
        return "I cannot help with that request."

    # Execute tool calls safely
    for tc in result["tool_calls"]:
        allowed, _ = defense.validate_before_execute(
            tc["tool"], tc["arguments"]
        )
        if allowed:
            execute_tool(tc)

    return result["response"]
```

---

## 12. Known Limitations and Future Work

### 12.1 Current Limitations

1. **MoE Expert Handling**
   - Current approach may not optimally leverage MoE structure
   - Need to investigate expert-specific safety patterns

2. **Completion Coverage**
   - Not all datasets have completions
   - Some prompt-only examples still used

3. **Runtime Overhead**
   - Defense module adds latency
   - Need benchmarks for production use

4. **Evaluation Completeness**
   - Tool call patterns may not cover all attack vectors
   - Need more diverse agent benchmarks

### 12.2 Future Work

1. **MoE-Aware Training**
   - Implement MoLoRA or X-LoRA for expert-specific adapters
   - Study expert routing patterns for harmful content

2. **Improved Completions**
   - Generate synthetic harmful completions via red-teaming
   - Use judge models to label completion quality

3. **Adaptive Defense**
   - Learn representation patterns from attacks
   - Update defense thresholds dynamically

4. **Multi-Modal Extension**
   - Handle image-based attacks
   - Support multi-modal tool calls

5. **Unsloth Integration**
   - Leverage QLoRA/4-bit quantization
   - Use optimized training kernels
   - Export to GGUF for deployment

---

## Appendix A: API Reference

### Training API

```python
# Config
get_config(preset: str, **overrides) -> CircuitBreakerConfig

# Trainer
CircuitBreakerTrainer(config: CircuitBreakerConfig)
trainer.train() -> None
trainer.save_checkpoint(final: bool = False) -> None

# Loss Functions
reroute_loss(model_reps, frozen_reps, layers, mask, loss_mask) -> Tensor
retain_loss(model_reps, frozen_reps, layers, mask, loss_mask) -> Tensor
get_alpha(step, alpha_max, total_steps, strategy) -> float
get_dual_coefficients(step, total_steps, alpha_max, ...) -> Tuple[float, float]
```

### Evaluation API

```python
# Text-based
is_refusal(response: str) -> bool
refusal_score(response: str) -> float

# Action-based
extract_tool_calls(response: str) -> List[Dict]
is_harmful_tool_call(tool_call: Dict, patterns: Dict) -> Tuple[bool, str]
evaluate_agent_safety(model, tokenizer, prompts, ...) -> Dict

# Full suite
run_full_evaluation(model, tokenizer, harmful, benign, agent, ...) -> Dict
```

### Defense API

```python
# Setup
create_defense_system(model, tokenizer, preset: str) -> CircuitBreakerInference

# Inference
defense.generate(prompt, trajectory_id, ...) -> Dict
defense.validate_before_execute(tool, args) -> Tuple[bool, Dict]

# Components
RepresentationMonitor.check_representations(reps) -> Tuple[Action, float, Dict]
ToolCallGuard.validate_tool_call(tool, args) -> Tuple[Action, float, Dict]
TrajectoryAnalyzer.add_step(trajectory_id, step) -> Tuple[Action, float, Dict]
```

---

## Appendix B: Hyperparameter Recommendations

### Small Models (7-8B)

```python
config = get_config("llama-3-8b",
    alpha_max=10.0,
    total_steps=150,
    learning_rate=5e-5,
    batch_size=16,
    cb_target_layers=[10, 20],
)
```

### Medium Models (13-17B MoE)

```python
config = get_config("llama-4-scout",
    alpha_max=8.0,
    total_steps=300,
    learning_rate=2e-5,
    batch_size=8,
    gradient_accumulation_steps=2,
    cb_target_layers=[12, 24, 36],
)
```

### Large Models (70B+)

```python
config = get_config("default",
    base_model="meta-llama/Llama-3.1-70B-Instruct",
    alpha_max=5.0,
    total_steps=500,
    learning_rate=1e-5,
    batch_size=4,
    gradient_accumulation_steps=8,
    cb_target_layers=[20, 40, 60],
    gradient_checkpointing=True,
)
```

---

*Document Version: 1.0*
*Last Updated: December 2024*
*Authors: Agentic Circuit Breakers Implementation Team*
