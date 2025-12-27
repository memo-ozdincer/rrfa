#!/usr/bin/env python3
"""
Unsloth Bridge for Circuit Breakers

Minimal bridge to use Unsloth's optimized model loading with CB training.
Preserves Unsloth's speedups while ensuring CB compatibility.

Usage:
    from scripts.circuit_breakers.unsloth_bridge import load_model_for_cb

    model, tokenizer = load_model_for_cb(
        "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        use_unsloth=True,
        lora_config=my_lora_config,
    )
"""

import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

# Check availability
try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False


@dataclass
class UnslothBridgeConfig:
    """Configuration for Unsloth bridge."""
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    dtype: Optional[str] = None  # None = auto-detect
    use_gradient_checkpointing: str = "unsloth"  # or True, False
    trust_remote_code: bool = True


def check_hidden_states_support(model, tokenizer) -> bool:
    """
    Quick check that model supports output_hidden_states.
    Critical for CB's representation extraction.
    """
    import torch

    try:
        dummy_input = tokenizer("test", return_tensors="pt")
        dummy_input = {k: v.to(model.device) for k, v in dummy_input.items()}

        with torch.no_grad():
            outputs = model(**dummy_input, output_hidden_states=True)

        if outputs.hidden_states is None:
            return False
        if len(outputs.hidden_states) == 0:
            return False
        return True

    except Exception as e:
        warnings.warn(f"Hidden states check failed: {e}")
        return False


def load_with_unsloth(
    model_name: str,
    config: UnslothBridgeConfig,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
) -> Tuple[Any, Any]:
    """
    Load model using Unsloth's FastLanguageModel.

    Returns (model, tokenizer) with LoRA applied if target_modules specified.
    """
    if not UNSLOTH_AVAILABLE:
        raise ImportError(
            "Unsloth not installed. Install with: pip install unsloth"
        )

    print(f"Loading with Unsloth: {model_name}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name,
        max_seq_length=config.max_seq_length,
        load_in_4bit=config.load_in_4bit,
        dtype=config.dtype,
        trust_remote_code=config.trust_remote_code,
    )

    if target_modules:
        print(f"Applying Unsloth LoRA to: {target_modules}")
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
            use_gradient_checkpointing=config.use_gradient_checkpointing,
        )

    return model, tokenizer


def load_with_hf(
    model_name: str,
    load_in_4bit: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    device_map: str = "auto",
) -> Tuple[Any, Any]:
    """
    Load model using vanilla HuggingFace + PEFT.

    Fallback when Unsloth is unavailable or incompatible.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"Loading with HuggingFace: {model_name}")

    # Quantization config
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = None

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        torch_dtype=torch.bfloat16 if not load_in_4bit else None,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if target_modules:
        from peft import LoraConfig, get_peft_model, TaskType

        print(f"Applying PEFT LoRA to: {target_modules}")
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )
        model = get_peft_model(model, lora_config)

    return model, tokenizer


def load_model_for_cb(
    model_name: str,
    use_unsloth: bool = True,
    unsloth_config: Optional[UnslothBridgeConfig] = None,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    verify_hidden_states: bool = True,
    fallback_to_hf: bool = True,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """
    Load model for Circuit Breaker training.

    Attempts Unsloth first if requested, with automatic fallback to HuggingFace
    if hidden_states support is missing.

    Args:
        model_name: HuggingFace model name or path
        use_unsloth: Whether to try Unsloth first
        unsloth_config: Unsloth-specific configuration
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        target_modules: Modules to apply LoRA to (None = no LoRA)
        verify_hidden_states: Check hidden_states support after loading
        fallback_to_hf: Fall back to HuggingFace if Unsloth fails

    Returns:
        (model, tokenizer, info_dict)
        info_dict contains: {"loader": "unsloth"|"hf", "hidden_states_ok": bool}
    """
    info = {
        "loader": None,
        "hidden_states_ok": False,
        "model_name": model_name,
        "lora_applied": target_modules is not None,
    }

    if unsloth_config is None:
        unsloth_config = UnslothBridgeConfig()

    # Default target modules for common architectures
    if target_modules is None:
        target_modules = []  # No LoRA unless explicitly requested

    model = None
    tokenizer = None

    # Try Unsloth first
    if use_unsloth and UNSLOTH_AVAILABLE:
        try:
            model, tokenizer = load_with_unsloth(
                model_name,
                unsloth_config,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules if target_modules else None,
            )
            info["loader"] = "unsloth"

            # Verify hidden states
            if verify_hidden_states:
                info["hidden_states_ok"] = check_hidden_states_support(model, tokenizer)
                if not info["hidden_states_ok"]:
                    warnings.warn(
                        "Unsloth model does not support output_hidden_states properly. "
                        "CB training requires hidden states for representation rerouting."
                    )
                    if fallback_to_hf:
                        print("Falling back to HuggingFace loader...")
                        model = None
                        tokenizer = None
                    else:
                        raise RuntimeError("Hidden states not supported and fallback disabled")
            else:
                info["hidden_states_ok"] = True  # Assume OK if not verified

        except Exception as e:
            warnings.warn(f"Unsloth loading failed: {e}")
            if not fallback_to_hf:
                raise

    # Fall back to HuggingFace
    if model is None:
        model, tokenizer = load_with_hf(
            model_name,
            load_in_4bit=unsloth_config.load_in_4bit,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules if target_modules else None,
        )
        info["loader"] = "hf"

        if verify_hidden_states:
            info["hidden_states_ok"] = check_hidden_states_support(model, tokenizer)
            if not info["hidden_states_ok"]:
                raise RuntimeError(
                    "Model does not support output_hidden_states. "
                    "CB training cannot proceed."
                )
        else:
            info["hidden_states_ok"] = True

    print(f"\nModel loaded via: {info['loader']}")
    print(f"Hidden states support: {'OK' if info['hidden_states_ok'] else 'FAILED'}")
    print(f"LoRA applied: {info['lora_applied']}")

    return model, tokenizer, info


def prepare_for_cb_training(model, tokenizer) -> Tuple[Any, Any]:
    """
    Final preparation for CB training.

    Ensures model is in training mode with proper settings.
    """
    model.train()

    # Ensure gradient checkpointing doesn't break hidden states
    if hasattr(model, 'gradient_checkpointing_enable'):
        # Check if already enabled
        if not getattr(model, 'gradient_checkpointing', False):
            model.gradient_checkpointing_enable()

    # Ensure hidden states will be returned
    if hasattr(model, 'config'):
        model.config.output_hidden_states = True

    return model, tokenizer


# Quick test function
def test_bridge(model_name: str = "unsloth/llama-3-8b-bnb-4bit"):
    """Quick test of the bridge."""
    print("Testing Unsloth bridge...")

    model, tokenizer, info = load_model_for_cb(
        model_name,
        use_unsloth=True,
        target_modules=["q_proj", "v_proj"],  # Minimal LoRA for testing
        verify_hidden_states=True,
        fallback_to_hf=True,
    )

    print(f"\nTest Results:")
    print(f"  Loader used: {info['loader']}")
    print(f"  Hidden states: {'OK' if info['hidden_states_ok'] else 'FAILED'}")

    # Quick forward pass
    import torch
    inputs = tokenizer("Hello world", return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    if outputs.hidden_states:
        print(f"  Hidden states count: {len(outputs.hidden_states)}")
        print(f"  Shape: {outputs.hidden_states[0].shape}")

    return model, tokenizer, info


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        test_bridge(sys.argv[1])
    else:
        test_bridge()
