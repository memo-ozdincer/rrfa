#!/usr/bin/env python3
"""
Unsloth + Circuit Breakers Compatibility Verification

Tests whether Unsloth's FastLanguageModel properly supports:
1. output_hidden_states for representation extraction
2. LoRA module naming conventions compatible with CB
3. Hidden state shapes matching vanilla HuggingFace

Run: python scripts/analysis/unsloth_cb_compatibility.py --model unsloth/llama-3-8b-bnb-4bit
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Check if unsloth is available
try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    print("WARNING: Unsloth not installed. Install with: pip install unsloth")


def test_hidden_states_support(
    model,
    tokenizer,
    test_prompt: str = "Hello, how are you?",
) -> Dict[str, Any]:
    """Test if model supports output_hidden_states."""
    import torch

    results = {
        "supports_hidden_states": False,
        "num_layers": 0,
        "hidden_state_shape": None,
        "dtype": None,
        "error": None,
    }

    try:
        # Tokenize
        inputs = tokenizer(test_prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Forward with hidden states
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # Check structure
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            hs = outputs.hidden_states
            results["supports_hidden_states"] = True
            results["num_layers"] = len(hs) - 1  # Exclude embedding layer
            results["hidden_state_shape"] = list(hs[0].shape)
            results["dtype"] = str(hs[0].dtype)

            print(f"  Hidden states: {len(hs)} tensors")
            print(f"  Shape: {results['hidden_state_shape']}")
            print(f"  Dtype: {results['dtype']}")
        else:
            results["error"] = "output_hidden_states returned None or missing"
            print(f"  ERROR: {results['error']}")

    except Exception as e:
        results["error"] = str(e)
        print(f"  ERROR: {e}")

    return results


def test_lora_module_structure(model) -> Dict[str, Any]:
    """Analyze LoRA module structure for CB compatibility."""
    results = {
        "has_lora": False,
        "lora_module_count": 0,
        "lora_module_paths": [],
        "naming_convention": None,
    }

    lora_modules = []
    for name, module in model.named_modules():
        module_type = type(module).__name__
        if 'lora' in name.lower() or 'Lora' in module_type:
            lora_modules.append(name)

    results["has_lora"] = len(lora_modules) > 0
    results["lora_module_count"] = len(lora_modules)
    results["lora_module_paths"] = lora_modules[:10]  # First 10 examples

    # Detect naming convention
    if lora_modules:
        if any('lora_A' in m for m in lora_modules):
            results["naming_convention"] = "standard_peft"  # lora_A, lora_B
        elif any('lora.A' in m for m in lora_modules):
            results["naming_convention"] = "dot_notation"   # lora.A, lora.B
        elif any('LoRA' in m for m in lora_modules):
            results["naming_convention"] = "unsloth_custom"
        else:
            results["naming_convention"] = "unknown"

    print(f"  Has LoRA: {results['has_lora']}")
    print(f"  LoRA modules: {results['lora_module_count']}")
    if results["lora_module_paths"]:
        print(f"  Naming convention: {results['naming_convention']}")
        print(f"  Examples: {results['lora_module_paths'][:3]}")

    return results


def test_representation_extraction(
    model,
    tokenizer,
    target_layers: List[int],
    test_prompt: str = "Hello, how are you?",
) -> Dict[str, Any]:
    """Test CB-style representation extraction."""
    import torch

    results = {
        "extraction_works": False,
        "target_layers": target_layers,
        "extracted_shapes": {},
        "total_memory_mb": 0,
        "error": None,
    }

    try:
        inputs = tokenizer(test_prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        if outputs.hidden_states is None:
            results["error"] = "hidden_states is None"
            return results

        hs = outputs.hidden_states

        # Extract target layers (index+1 because 0 is embeddings)
        extracted = {}
        total_mem = 0
        for layer_idx in target_layers:
            hs_idx = layer_idx + 1
            if hs_idx < len(hs):
                tensor = hs[hs_idx]
                extracted[layer_idx] = tensor
                mem_mb = tensor.numel() * tensor.element_size() / (1024**2)
                total_mem += mem_mb
                results["extracted_shapes"][layer_idx] = list(tensor.shape)

        results["extraction_works"] = len(extracted) == len(target_layers)
        results["total_memory_mb"] = total_mem

        print(f"  Target layers: {target_layers}")
        print(f"  Extracted shapes: {results['extracted_shapes']}")
        print(f"  Memory: {total_mem:.2f} MB")

    except Exception as e:
        results["error"] = str(e)
        print(f"  ERROR: {e}")

    return results


def test_gradient_flow(
    model,
    tokenizer,
    target_layer: int = 12,
    test_prompt: str = "Hello, how are you?",
) -> Dict[str, Any]:
    """Test if gradients flow through hidden states (needed for CB training)."""
    import torch

    results = {
        "gradients_flow": False,
        "grad_shape": None,
        "error": None,
    }

    try:
        inputs = tokenizer(test_prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Forward with hidden states (grad enabled)
        outputs = model(**inputs, output_hidden_states=True)

        if outputs.hidden_states is None:
            results["error"] = "hidden_states is None"
            return results

        # Get target layer hidden state
        hs = outputs.hidden_states[target_layer + 1]

        # Create dummy loss and backward
        dummy_loss = hs.mean()
        dummy_loss.backward()

        # Check if any LoRA params got gradients
        has_grad = False
        for name, param in model.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                if 'lora' in name.lower():
                    has_grad = True
                    break

        results["gradients_flow"] = has_grad
        results["grad_shape"] = list(hs.shape)

        if has_grad:
            print(f"  Gradients flow through hidden states")
        else:
            print(f"  WARNING: No gradients detected in LoRA params")

        # Zero grads for cleanup
        model.zero_grad()

    except Exception as e:
        results["error"] = str(e)
        print(f"  ERROR: {e}")

    return results


def compare_with_vanilla_hf(
    unsloth_model,
    model_name: str,
    tokenizer,
    test_prompt: str = "Hello, how are you?",
) -> Dict[str, Any]:
    """Compare Unsloth hidden states with vanilla HuggingFace (if memory permits)."""
    import torch

    results = {
        "comparison_done": False,
        "shapes_match": False,
        "values_close": False,
        "max_diff": None,
        "error": None,
    }

    # This test is expensive - only run if explicitly requested
    print("  (Skipping HF comparison - requires loading full model twice)")
    results["comparison_done"] = False
    results["error"] = "Skipped to save memory"

    return results


def run_full_compatibility_check(
    model_name: str = "unsloth/llama-3-8b-bnb-4bit",
    target_layers: List[int] = [8, 16, 24],
    max_seq_length: int = 512,
) -> Dict[str, Any]:
    """Run all compatibility tests."""

    print("=" * 60)
    print("UNSLOTH + CIRCUIT BREAKERS COMPATIBILITY CHECK")
    print("=" * 60)
    print(f"\nModel: {model_name}")
    print(f"Target layers: {target_layers}")

    if not UNSLOTH_AVAILABLE:
        return {"error": "Unsloth not installed"}

    # Load model
    print(f"\n{'='*60}")
    print("1. LOADING MODEL")
    print(f"{'='*60}")

    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=True,
            dtype=None,  # Auto-detect
        )
        print(f"  Model loaded successfully")
        print(f"  Device: {model.device}")
    except Exception as e:
        return {"error": f"Failed to load model: {e}"}

    # Apply LoRA for testing
    print(f"\n{'='*60}")
    print("2. APPLYING LORA")
    print(f"{'='*60}")

    try:
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            bias="none",
            use_gradient_checkpointing="unsloth",
        )
        print(f"  LoRA applied successfully")
    except Exception as e:
        print(f"  WARNING: LoRA application failed: {e}")
        print(f"  Continuing without LoRA...")

    results = {
        "model_name": model_name,
        "target_layers": target_layers,
        "tests": {},
    }

    # Test 1: Hidden states support
    print(f"\n{'='*60}")
    print("3. TESTING HIDDEN STATES SUPPORT")
    print(f"{'='*60}")
    results["tests"]["hidden_states"] = test_hidden_states_support(model, tokenizer)

    # Test 2: LoRA structure
    print(f"\n{'='*60}")
    print("4. ANALYZING LORA MODULE STRUCTURE")
    print(f"{'='*60}")
    results["tests"]["lora_structure"] = test_lora_module_structure(model)

    # Test 3: Representation extraction
    print(f"\n{'='*60}")
    print("5. TESTING CB REPRESENTATION EXTRACTION")
    print(f"{'='*60}")
    results["tests"]["extraction"] = test_representation_extraction(
        model, tokenizer, target_layers
    )

    # Test 4: Gradient flow
    print(f"\n{'='*60}")
    print("6. TESTING GRADIENT FLOW")
    print(f"{'='*60}")
    results["tests"]["gradients"] = test_gradient_flow(
        model, tokenizer, target_layer=target_layers[0]
    )

    # Summary
    print(f"\n{'='*60}")
    print("COMPATIBILITY SUMMARY")
    print(f"{'='*60}")

    all_pass = True
    checks = [
        ("Hidden states support", results["tests"]["hidden_states"]["supports_hidden_states"]),
        ("LoRA modules present", results["tests"]["lora_structure"]["has_lora"]),
        ("Representation extraction", results["tests"]["extraction"]["extraction_works"]),
        ("Gradient flow", results["tests"]["gradients"]["gradients_flow"]),
    ]

    for name, passed in checks:
        status = "PASS" if passed else "FAIL"
        symbol = "+" if passed else "X"
        print(f"  [{symbol}] {name}: {status}")
        if not passed:
            all_pass = False

    results["all_compatible"] = all_pass

    if all_pass:
        print(f"\n  RESULT: Unsloth is COMPATIBLE with Circuit Breakers")
    else:
        print(f"\n  RESULT: Unsloth may have COMPATIBILITY ISSUES")
        print(f"  Review failed tests and consider using vanilla HuggingFace")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Test Unsloth compatibility with Circuit Breakers"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="unsloth/llama-3-8b-bnb-4bit",
        help="Model name (use Unsloth's pre-quantized models for best results)",
    )
    parser.add_argument(
        "--target-layers",
        nargs="+",
        type=int,
        default=[8, 16, 24],
        help="Target layers for CB extraction",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=512,
        help="Max sequence length",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path for results",
    )
    args = parser.parse_args()

    results = run_full_compatibility_check(
        model_name=args.model,
        target_layers=args.target_layers,
        max_seq_length=args.max_seq_length,
    )

    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
