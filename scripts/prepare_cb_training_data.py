#!/usr/bin/env python3
"""
Prepare Circuit Breaker Training Data with Completions

This script orchestrates the data pipeline to create training data for
Circuit Breakers with completion-based training:

1. Extract harmful pairs from source datasets
2. Extract benign pairs from source datasets
3. Split out completion-style examples where available
4. Create balanced training batches with 'text' fields

Usage:
    python scripts/prepare_cb_training_data.py --output-dir data/circuit_breakers

The output includes:
    - harmful_pairs.jsonl: Harmful examples (with completions where available)
    - benign_pairs.jsonl: Benign examples (with completions where available)
    - cb_training_batches.jsonl: Pre-batched training data for CB trainer
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from format_for_cb.extract_harmful import extract_fujitsu_harmful, extract_agentdojo_harmful
from format_for_cb.extract_benign import extract_webarena_benign, extract_agentdojo_benign

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
CB_DIR = DATA_DIR / "circuit_breakers"


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read a JSONL file."""
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> int:
    """Write a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def add_text_field(sample: Dict[str, Any], is_harmful: bool = True) -> Dict[str, Any]:
    """
    Add a 'text' field to a sample for completion-based training.

    The 'text' field contains the full conversation in a simple format
    that can later be re-templated using the tokenizer's chat template.
    """
    if sample.get("text"):
        return sample  # Already has text field

    # Try to construct from user_prompt + completion
    user_prompt = sample.get("user_prompt", "")
    if not user_prompt:
        user_prompt = sample.get("attack_prompt", "") or sample.get("prompt", "")
        if sample.get("benign_query"):
            user_prompt = f"{sample['benign_query']}\n{user_prompt}"

    # Get completion
    if is_harmful:
        completion = sample.get("harmful_completion", "")
        if not completion:
            # Try to derive from metadata
            md = sample.get("metadata", {})
            completion = md.get("target_llm_output", "") or md.get("mta_output", "")
    else:
        completion = sample.get("benign_completion", "")
        if not completion:
            completion = sample.get("response", "") or sample.get("answer", "")

    if user_prompt and completion:
        # Create a simple chat format that can be re-templated
        sample["user_prompt"] = user_prompt
        if is_harmful:
            sample["harmful_completion"] = completion
        else:
            sample["benign_completion"] = completion
        # Simple format - will be properly templated by the trainer
        sample["text"] = f"User: {user_prompt}\n\nAssistant: {completion}"

    return sample


def load_harmful_data(include_completions: bool = True) -> List[Dict[str, Any]]:
    """Load harmful data from all sources."""
    harmful_pairs = []

    print("\n=== Loading Harmful Data ===")

    # Try to load pre-processed harmful pairs
    harmful_path = CB_DIR / "harmful" / "harmful_pairs.jsonl"
    if harmful_path.exists():
        pairs = read_jsonl(harmful_path)
        print(f"  Loaded {len(pairs)} from existing harmful_pairs.jsonl")
        harmful_pairs.extend(pairs)

    # Also load from split completions if available
    completions_path = CB_DIR / "harmful" / "harmful_pairs.completions.jsonl"
    if completions_path.exists():
        pairs = read_jsonl(completions_path)
        print(f"  Loaded {len(pairs)} with completions")
        # Merge with main pairs, preferring completion versions
        existing_ids = {p.get("id") for p in harmful_pairs}
        completion_ids = {p.get("id") for p in pairs}
        # Remove prompt-only versions where completion exists
        harmful_pairs = [p for p in harmful_pairs if p.get("id") not in completion_ids]
        harmful_pairs.extend(pairs)

    # Add text fields where missing
    if include_completions:
        for i, sample in enumerate(harmful_pairs):
            harmful_pairs[i] = add_text_field(sample, is_harmful=True)

    # Count samples with completions
    with_completions = sum(1 for p in harmful_pairs if p.get("text"))
    print(f"  Total harmful: {len(harmful_pairs)} ({with_completions} with completions)")

    return harmful_pairs


def load_benign_data(include_completions: bool = True) -> List[Dict[str, Any]]:
    """Load benign data from all sources."""
    benign_pairs = []

    print("\n=== Loading Benign Data ===")

    # Try to load pre-processed benign pairs
    benign_path = CB_DIR / "benign" / "benign_pairs.jsonl"
    if benign_path.exists():
        pairs = read_jsonl(benign_path)
        print(f"  Loaded {len(pairs)} from existing benign_pairs.jsonl")
        benign_pairs.extend(pairs)

    # Also load from split completions if available
    completions_path = CB_DIR / "benign" / "benign_pairs.completions.jsonl"
    if completions_path.exists():
        pairs = read_jsonl(completions_path)
        print(f"  Loaded {len(pairs)} with completions")
        existing_ids = {p.get("id") for p in benign_pairs}
        completion_ids = {p.get("id") for p in pairs}
        benign_pairs = [p for p in benign_pairs if p.get("id") not in completion_ids]
        benign_pairs.extend(pairs)

    # Add text fields where missing
    if include_completions:
        for i, sample in enumerate(benign_pairs):
            benign_pairs[i] = add_text_field(sample, is_harmful=False)

    with_completions = sum(1 for p in benign_pairs if p.get("text"))
    print(f"  Total benign: {len(benign_pairs)} ({with_completions} with completions)")

    return benign_pairs


def create_balanced_batches(
    harmful_pairs: List[Dict[str, Any]],
    benign_pairs: List[Dict[str, Any]],
    batch_size: int = 16,
    prioritize_completions: bool = True,
) -> List[Dict[str, Any]]:
    """
    Create balanced 1:1 harmful:benign batches for CB training.

    Each batch contains:
    - batch_size // 2 harmful samples
    - batch_size // 2 benign samples

    If prioritize_completions is True, samples with 'text' fields
    are preferred.
    """
    print("\n=== Creating Balanced Batches ===")

    # Sort by completion availability if prioritizing
    if prioritize_completions:
        harmful_pairs = sorted(
            harmful_pairs,
            key=lambda x: (1 if x.get("text") else 0),
            reverse=True
        )
        benign_pairs = sorted(
            benign_pairs,
            key=lambda x: (1 if x.get("text") else 0),
            reverse=True
        )

    # Shuffle within completion/non-completion groups
    random.shuffle(harmful_pairs)
    random.shuffle(benign_pairs)

    # Determine number of batches
    half_batch = batch_size // 2
    n_harmful = len(harmful_pairs)
    n_benign = len(benign_pairs)
    max_batches = min(n_harmful // half_batch, n_benign // half_batch)

    print(f"  Harmful samples: {n_harmful}")
    print(f"  Benign samples: {n_benign}")
    print(f"  Batch size: {batch_size} ({half_batch} harmful + {half_batch} benign)")
    print(f"  Max batches: {max_batches}")

    batches = []
    for i in range(max_batches):
        batch_harmful = harmful_pairs[i * half_batch : (i + 1) * half_batch]
        batch_benign = benign_pairs[i * half_batch : (i + 1) * half_batch]

        # Count completions in this batch
        harmful_with_text = sum(1 for s in batch_harmful if s.get("text"))
        benign_with_text = sum(1 for s in batch_benign if s.get("text"))

        batch = {
            "batch_id": i,
            "batch_size": batch_size,
            "composition": {
                "harmful": len(batch_harmful),
                "benign": len(batch_benign),
                "harmful_with_completions": harmful_with_text,
                "benign_with_completions": benign_with_text,
            },
            "harmful": batch_harmful,
            "benign": batch_benign,
        }
        batches.append(batch)

    # Summary statistics
    total_harmful_text = sum(b["composition"]["harmful_with_completions"] for b in batches)
    total_benign_text = sum(b["composition"]["benign_with_completions"] for b in batches)
    total_samples = max_batches * batch_size

    print(f"\n  Created {len(batches)} batches ({total_samples} total samples)")
    print(f"  Harmful with completions: {total_harmful_text}/{max_batches * half_batch}")
    print(f"  Benign with completions: {total_benign_text}/{max_batches * half_batch}")

    return batches


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Circuit Breaker training data with completions"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CB_DIR,
        help="Output directory for training data",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size (must be even, default: 16)",
    )
    parser.add_argument(
        "--no-completions",
        action="store_true",
        help="Skip adding completion text fields",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Circuit Breaker Training Data Preparation")
    print("=" * 60)

    # Load data
    include_completions = not args.no_completions
    harmful_pairs = load_harmful_data(include_completions)
    benign_pairs = load_benign_data(include_completions)

    if not harmful_pairs or not benign_pairs:
        print("\n❌ Error: No data found. Run ingest_cb_data.py first.")
        sys.exit(1)

    # Create batches
    batches = create_balanced_batches(
        harmful_pairs,
        benign_pairs,
        batch_size=args.batch_size,
        prioritize_completions=include_completions,
    )

    # Write output
    output_path = output_dir / "cb_training_batches.jsonl"
    n_written = write_jsonl(output_path, batches)

    print("\n" + "=" * 60)
    print(f"Output: {output_path}")
    print(f"Batches: {n_written}")
    print("=" * 60)

    # Also save the merged harmful/benign files with text fields
    if include_completions:
        harmful_out = output_dir / "harmful" / "harmful_pairs_with_text.jsonl"
        benign_out = output_dir / "benign" / "benign_pairs_with_text.jsonl"
        write_jsonl(harmful_out, harmful_pairs)
        write_jsonl(benign_out, benign_pairs)
        print(f"\nAlso saved:")
        print(f"  {harmful_out}")
        print(f"  {benign_out}")


if __name__ == "__main__":
    main()
