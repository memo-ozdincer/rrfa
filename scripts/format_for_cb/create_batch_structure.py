#!/usr/bin/env python3
"""
Create balanced 1:1 batching structure for Circuit Breaker training.

Each training batch needs BOTH harmful and benign samples (1:1 ratio).
Within each category, mix 50% agent-specific + 50% general.

Usage:
    python scripts/format_for_cb/create_batch_structure.py \
        --harmful-agent data/circuit_breakers/harmful/*_agent.jsonl \
        --harmful-general data/circuit_breakers/harmful/*_general.jsonl \
        --benign-agent data/circuit_breakers/benign/agent_tools.jsonl \
        --benign-general data/circuit_breakers/benign/general_capability.jsonl \
        --output data/circuit_breakers/cb_training_batches.jsonl \
        --batch-size 16
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict
import glob


def load_jsonl(path: Path) -> List[Dict]:
    """Load JSONL file into list of dicts."""
    pairs = []
    with open(path, 'r') as f:
        for line in f:
            pairs.append(json.loads(line))
    return pairs


def create_balanced_batches(
    harmful_agent: List[Dict],
    harmful_general: List[Dict],
    benign_agent: List[Dict],
    benign_general: List[Dict],
    batch_size: int = 16
) -> List[Dict]:
    """
    Create balanced batches for Circuit Breaker training.

    Each batch contains:
    - batch_size/2 harmful samples (50% agent, 50% general)
    - batch_size/2 benign samples (50% agent, 50% general)

    Args:
        harmful_agent: Agent-specific harmful pairs
        harmful_general: General harmful pairs
        benign_agent: Agent-specific benign pairs
        benign_general: General benign pairs
        batch_size: Total samples per batch (must be even)

    Returns:
        List of batch dicts
    """
    if batch_size % 2 != 0:
        raise ValueError("batch_size must be even for 1:1 harmful:benign ratio")

    half_batch = batch_size // 2
    quarter_batch = half_batch // 2

    # Shuffle all datasets
    random.shuffle(harmful_agent)
    random.shuffle(harmful_general)
    random.shuffle(benign_agent)
    random.shuffle(benign_general)

    # Calculate number of batches (limited by smallest category)
    max_batches = min(
        len(harmful_agent) // quarter_batch,
        len(harmful_general) // quarter_batch,
        len(benign_agent) // quarter_batch,
        len(benign_general) // quarter_batch
    )

    print(f"\n📊 Dataset sizes:")
    print(f"   Harmful agent: {len(harmful_agent)}")
    print(f"   Harmful general: {len(harmful_general)}")
    print(f"   Benign agent: {len(benign_agent)}")
    print(f"   Benign general: {len(benign_general)}")
    print(f"\n📦 Creating {max_batches} batches of size {batch_size}")
    print(f"   Per batch: {half_batch} harmful (🔴{quarter_batch} agent + 🔴{quarter_batch} general) + {half_batch} benign (🟢{quarter_batch} agent + 🟢{quarter_batch} general)")

    batches = []
    for i in range(max_batches):
        # Extract samples for this batch
        batch_harmful_agent = harmful_agent[i * quarter_batch:(i + 1) * quarter_batch]
        batch_harmful_general = harmful_general[i * quarter_batch:(i + 1) * quarter_batch]
        batch_benign_agent = benign_agent[i * quarter_batch:(i + 1) * quarter_batch]
        batch_benign_general = benign_general[i * quarter_batch:(i + 1) * quarter_batch]

        # Combine and shuffle within batch
        batch_samples = (
            batch_harmful_agent +
            batch_harmful_general +
            batch_benign_agent +
            batch_benign_general
        )
        random.shuffle(batch_samples)

        batch = {
            "batch_id": i,
            "batch_size": batch_size,
            "composition": {
                "harmful": half_batch,
                "benign": half_batch,
                "harmful_agent": quarter_batch,
                "harmful_general": quarter_batch,
                "benign_agent": quarter_batch,
                "benign_general": quarter_batch
            },
            "samples": batch_samples
        }
        batches.append(batch)

    return batches


def main():
    parser = argparse.ArgumentParser(
        description="Create balanced 1:1 batches for Circuit Breaker training"
    )
    parser.add_argument(
        '--harmful-agent',
        type=str,
        required=True,
        help='Glob pattern for harmful agent files (e.g., harmful/*_agent.jsonl)'
    )
    parser.add_argument(
        '--harmful-general',
        type=str,
        required=True,
        help='Glob pattern for harmful general files'
    )
    parser.add_argument(
        '--benign-agent',
        type=Path,
        required=True,
        help='Path to benign agent file'
    )
    parser.add_argument(
        '--benign-general',
        type=Path,
        required=True,
        help='Path to benign general file'
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output path for batched data'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Total samples per batch (default: 16)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    # Load data
    print("Loading datasets...")

    # Harmful agent (multiple files possible)
    harmful_agent = []
    for path in glob.glob(args.harmful_agent):
        harmful_agent.extend(load_jsonl(Path(path)))
        print(f"  ✓ Loaded {Path(path).name}")

    # Harmful general (multiple files possible)
    harmful_general = []
    for path in glob.glob(args.harmful_general):
        harmful_general.extend(load_jsonl(Path(path)))
        print(f"  ✓ Loaded {Path(path).name}")

    # Benign
    benign_agent = load_jsonl(args.benign_agent)
    print(f"  ✓ Loaded {args.benign_agent.name}")

    benign_general = load_jsonl(args.benign_general)
    print(f"  ✓ Loaded {args.benign_general.name}")

    # Create batches
    batches = create_balanced_batches(
        harmful_agent,
        harmful_general,
        benign_agent,
        benign_general,
        batch_size=args.batch_size
    )

    # Save
    with open(args.output, 'w') as f:
        for batch in batches:
            f.write(json.dumps(batch) + '\n')

    print(f"\n✅ Saved {len(batches)} batches to {args.output}")
    print(f"📊 Total samples: {len(batches) * args.batch_size}")
    print(f"\n🔑 Key: Each batch has 1:1 harmful:benign ratio")
    print(f"   - Balance controlled by α schedule (α=10→0), NOT data volume")
    print(f"   - Ready for Circuit Breaker training!")


if __name__ == '__main__':
    main()
