#!/usr/bin/env python3
"""
MoE LoRA Diff Report Generator

Produces empirical evidence about:
1. Module name matches between target_modules and actual model
2. How many LoRA modules are instantiated
3. Whether adapters are SHARED vs DISTINCT across experts
4. Memory deltas with output_hidden_states

Run: python scripts/analysis/moe_lora_diff_report.py --model meta-llama/Llama-4-Scout-17B-16E-Instruct
"""

import argparse
import gc
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


@dataclass
class ModuleAnalysis:
    """Analysis results for a single module type."""
    name: str
    total_count: int
    in_experts: int
    in_shared: int
    unique_shapes: List[Tuple[int, ...]]
    example_paths: List[str]


@dataclass
class LoRAAnalysis:
    """Analysis of LoRA adapter instantiation."""
    total_lora_modules: int
    lora_A_count: int
    lora_B_count: int
    shared_across_experts: bool
    distinct_adapter_count: int
    total_trainable_params: int
    memory_mb: float


def analyze_model_structure(model_name: str, device: str = "cpu") -> Dict:
    """Analyze model structure without loading full weights."""
    print(f"\n{'='*60}")
    print(f"ANALYZING MODEL STRUCTURE: {model_name}")
    print(f"{'='*60}\n")

    from transformers import AutoConfig

    try:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    except Exception as e:
        print(f"Could not load config: {e}")
        return {}

    analysis = {
        "model_name": model_name,
        "model_type": getattr(config, "model_type", "unknown"),
        "num_layers": getattr(config, "num_hidden_layers", 0),
        "hidden_size": getattr(config, "hidden_size", 0),
        "num_attention_heads": getattr(config, "num_attention_heads", 0),
    }

    # MoE-specific
    if hasattr(config, "num_experts") or hasattr(config, "num_local_experts"):
        analysis["is_moe"] = True
        analysis["num_experts"] = getattr(config, "num_experts",
                                          getattr(config, "num_local_experts", 0))
        analysis["num_experts_per_tok"] = getattr(config, "num_experts_per_tok",
                                                   getattr(config, "num_selected_experts", 2))
    else:
        analysis["is_moe"] = False

    print(f"Model Type: {analysis['model_type']}")
    print(f"Num Layers: {analysis['num_layers']}")
    print(f"Hidden Size: {analysis['hidden_size']}")
    print(f"Is MoE: {analysis['is_moe']}")
    if analysis["is_moe"]:
        print(f"Num Experts: {analysis.get('num_experts', 'N/A')}")
        print(f"Experts per Token: {analysis.get('num_experts_per_tok', 'N/A')}")

    return analysis


def find_target_modules(model, target_module_names: List[str]) -> Dict[str, ModuleAnalysis]:
    """Find all modules matching target names and analyze their structure."""
    print(f"\n{'='*60}")
    print("MODULE MATCHING ANALYSIS")
    print(f"{'='*60}\n")

    module_analysis = {}

    for target_name in target_module_names:
        matches = []
        for name, module in model.named_modules():
            if name.endswith(target_name) or f".{target_name}" in name:
                matches.append((name, module))

        if matches:
            # Categorize as expert vs shared
            in_experts = sum(1 for n, _ in matches if "expert" in n.lower() or "moe" in n.lower())
            in_shared = len(matches) - in_experts

            # Get unique shapes
            shapes = set()
            for _, m in matches:
                if hasattr(m, 'weight'):
                    shapes.add(tuple(m.weight.shape))

            module_analysis[target_name] = ModuleAnalysis(
                name=target_name,
                total_count=len(matches),
                in_experts=in_experts,
                in_shared=in_shared,
                unique_shapes=list(shapes),
                example_paths=[n for n, _ in matches[:5]],  # First 5 examples
            )

            print(f"\n{target_name}:")
            print(f"  Total matches: {len(matches)}")
            print(f"  In experts: {in_experts}")
            print(f"  In shared/attention: {in_shared}")
            print(f"  Unique shapes: {shapes}")
            print(f"  Example paths:")
            for path in module_analysis[target_name].example_paths[:3]:
                print(f"    - {path}")

    return module_analysis


def analyze_lora_modules(model) -> LoRAAnalysis:
    """Analyze LoRA adapter instantiation after get_peft_model."""
    print(f"\n{'='*60}")
    print("LORA ADAPTER ANALYSIS")
    print(f"{'='*60}\n")

    lora_A_modules = []
    lora_B_modules = []
    lora_module_paths = []

    for name, module in model.named_modules():
        module_type = type(module).__name__
        if "lora_A" in name or "lora_a" in name.lower():
            lora_A_modules.append((name, module))
            lora_module_paths.append(name)
        elif "lora_B" in name or "lora_b" in name.lower():
            lora_B_modules.append((name, module))

    # Check if adapters are shared (same object ID) or distinct
    if lora_A_modules:
        ids = [id(m) for _, m in lora_A_modules]
        unique_ids = len(set(ids))
        shared = unique_ids < len(ids)
    else:
        unique_ids = 0
        shared = False

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Estimate memory
    memory_mb = trainable_params * 4 / (1024 * 1024)  # Assuming float32

    analysis = LoRAAnalysis(
        total_lora_modules=len(lora_A_modules) + len(lora_B_modules),
        lora_A_count=len(lora_A_modules),
        lora_B_count=len(lora_B_modules),
        shared_across_experts=shared,
        distinct_adapter_count=unique_ids,
        total_trainable_params=trainable_params,
        memory_mb=memory_mb,
    )

    print(f"Total LoRA modules: {analysis.total_lora_modules}")
    print(f"  - lora_A: {analysis.lora_A_count}")
    print(f"  - lora_B: {analysis.lora_B_count}")
    print(f"\nAdapter sharing: {'SHARED' if shared else 'DISTINCT per module'}")
    print(f"Unique adapter instances: {unique_ids}")
    print(f"\nTrainable parameters: {trainable_params:,}")
    print(f"Estimated memory: {memory_mb:.2f} MB")

    # Show distribution across experts if MoE
    expert_adapters = defaultdict(int)
    shared_adapters = 0
    for path in lora_module_paths:
        if "expert" in path.lower():
            # Extract expert number
            import re
            match = re.search(r'expert[s]?[._]?(\d+)', path.lower())
            if match:
                expert_adapters[int(match.group(1))] += 1
        else:
            shared_adapters += 1

    if expert_adapters:
        print(f"\nLoRA distribution:")
        print(f"  Shared/Attention adapters: {shared_adapters}")
        print(f"  Expert adapters: {sum(expert_adapters.values())}")
        print(f"  Experts with adapters: {len(expert_adapters)}")

    return analysis


def measure_hidden_states_memory(
    model,
    tokenizer,
    target_layers: List[int],
    seq_length: int = 512,
    batch_size: int = 1,
) -> Dict[str, float]:
    """Measure memory impact of output_hidden_states."""
    print(f"\n{'='*60}")
    print("HIDDEN STATES MEMORY ANALYSIS")
    print(f"{'='*60}\n")

    import torch

    device = next(model.parameters()).device

    # Create dummy input
    dummy_ids = torch.randint(0, 1000, (batch_size, seq_length), device=device)
    attention_mask = torch.ones_like(dummy_ids)

    results = {}

    # Measure without hidden states
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    with torch.no_grad():
        _ = model(dummy_ids, attention_mask=attention_mask, use_cache=False)

    if torch.cuda.is_available():
        mem_no_hs = torch.cuda.max_memory_allocated() / (1024**2)
    else:
        mem_no_hs = 0
    results["without_hidden_states_mb"] = mem_no_hs

    # Measure with hidden states
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    with torch.no_grad():
        outputs = model(
            dummy_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

    if torch.cuda.is_available():
        mem_with_hs = torch.cuda.max_memory_allocated() / (1024**2)
    else:
        mem_with_hs = 0
    results["with_hidden_states_mb"] = mem_with_hs
    results["delta_mb"] = mem_with_hs - mem_no_hs

    # Check hidden states structure
    if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
        hs = outputs.hidden_states
        results["num_hidden_states"] = len(hs)
        results["hidden_state_shape"] = list(hs[0].shape)

        # Memory for just target layers
        target_hs_mem = sum(
            hs[i+1].numel() * 4 / (1024**2)  # +1 because index 0 is embeddings
            for i in target_layers
            if i+1 < len(hs)
        )
        results["target_layers_only_mb"] = target_hs_mem
    else:
        results["num_hidden_states"] = 0
        results["hidden_state_shape"] = []
        results["target_layers_only_mb"] = 0

    print(f"Sequence length: {seq_length}")
    print(f"Batch size: {batch_size}")
    print(f"\nMemory without hidden_states: {mem_no_hs:.2f} MB")
    print(f"Memory with hidden_states: {mem_with_hs:.2f} MB")
    print(f"Delta: {results['delta_mb']:.2f} MB")
    print(f"\nHidden states count: {results['num_hidden_states']}")
    print(f"Hidden state shape: {results['hidden_state_shape']}")
    print(f"Target layers ({target_layers}) memory: {results['target_layers_only_mb']:.2f} MB")

    return results


def generate_diff_report(
    model_name: str,
    target_modules: List[str],
    target_layers: List[int],
    output_path: Optional[str] = None,
    load_model: bool = False,
    device: str = "cpu",
):
    """Generate the complete diff report."""

    report = {
        "model_name": model_name,
        "target_modules": target_modules,
        "target_layers": target_layers,
    }

    # 1. Analyze model structure (config only)
    report["structure"] = analyze_model_structure(model_name)

    if not load_model:
        print("\n⚠️  Skipping full model analysis (--load-model not set)")
        print("   Run with --load-model for complete LoRA and memory analysis")

        # Save partial report
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"\nPartial report saved to: {output_path}")

        return report

    # Load model for deeper analysis
    print(f"\n{'='*60}")
    print("LOADING MODEL FOR DEEP ANALYSIS")
    print(f"{'='*60}\n")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType

    # Determine load parameters based on device
    if device == "cpu":
        load_kwargs = {"torch_dtype": torch.float32, "device_map": "cpu"}
    else:
        load_kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}

    print(f"Loading model to {device}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        **load_kwargs,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 2. Find target modules before LoRA
    report["module_matches"] = {
        k: {
            "total_count": v.total_count,
            "in_experts": v.in_experts,
            "in_shared": v.in_shared,
            "shapes": [list(s) for s in v.unique_shapes],
            "examples": v.example_paths,
        }
        for k, v in find_target_modules(model, target_modules).items()
    }

    # 3. Apply LoRA
    print(f"\n{'='*60}")
    print("APPLYING LORA")
    print(f"{'='*60}\n")

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=target_modules,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 4. Analyze LoRA modules
    lora_analysis = analyze_lora_modules(model)
    report["lora_analysis"] = {
        "total_lora_modules": lora_analysis.total_lora_modules,
        "lora_A_count": lora_analysis.lora_A_count,
        "lora_B_count": lora_analysis.lora_B_count,
        "shared_across_experts": lora_analysis.shared_across_experts,
        "distinct_adapter_count": lora_analysis.distinct_adapter_count,
        "trainable_params": lora_analysis.total_trainable_params,
        "memory_mb": lora_analysis.memory_mb,
    }

    # 5. Measure hidden states memory (only if on GPU)
    if device != "cpu" and torch.cuda.is_available():
        report["hidden_states_memory"] = measure_hidden_states_memory(
            model, tokenizer, target_layers,
            seq_length=512, batch_size=1,
        )
    else:
        print("\n⚠️  Skipping hidden states memory analysis (requires GPU)")
        report["hidden_states_memory"] = {"note": "Requires GPU"}

    # Generate summary
    print(f"\n{'='*60}")
    print("SUMMARY & RECOMMENDATIONS")
    print(f"{'='*60}\n")

    is_moe = report["structure"].get("is_moe", False)
    num_experts = report["structure"].get("num_experts", 0)

    print(f"Model: {model_name}")
    print(f"Architecture: {'MoE' if is_moe else 'Dense'}")
    if is_moe:
        print(f"Experts: {num_experts}")

    if "lora_analysis" in report:
        la = report["lora_analysis"]
        print(f"\nLoRA Configuration:")
        print(f"  Total adapters: {la['total_lora_modules']}")
        print(f"  Trainable params: {la['trainable_params']:,}")

        if is_moe:
            if la["shared_across_experts"]:
                print(f"\n⚠️  SHARED ADAPTERS: Same weights used across experts")
                print(f"   This means experts will learn identical modifications.")
            else:
                print(f"\n✓  DISTINCT ADAPTERS: {la['distinct_adapter_count']} unique instances")
                print(f"   Each expert can learn different safety modifications.")

    # Recommendations
    print(f"\nCB-Specific Recommendations:")
    if is_moe:
        print("  1. Current setup applies distinct LoRA per expert (good for CB)")
        print("  2. Consider logging router decisions for harmful vs benign inputs")
        print("  3. Extract hidden states only from cb_target_layers to save memory")
    else:
        print("  1. Standard dense model - LoRA applies uniformly")
        print("  2. No expert-specific considerations needed")

    # Save report
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nFull report saved to: {output_path}")

    return report


def main():
    parser = argparse.ArgumentParser(description="MoE LoRA Diff Report Generator")
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-4-Scout-17B-16E-Instruct",
        help="Model name or path",
    )
    parser.add_argument(
        "--target-modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        help="Target modules for LoRA",
    )
    parser.add_argument(
        "--target-layers",
        nargs="+",
        type=int,
        default=[12, 24, 36],
        help="Target layers for CB representation extraction",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="moe_lora_diff_report.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--load-model",
        action="store_true",
        help="Actually load the model for deep analysis (requires GPU/memory)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda"],
        help="Device for model loading",
    )
    args = parser.parse_args()

    generate_diff_report(
        model_name=args.model,
        target_modules=args.target_modules,
        target_layers=args.target_layers,
        output_path=args.output,
        load_model=args.load_model,
        device=args.device,
    )


if __name__ == "__main__":
    main()
