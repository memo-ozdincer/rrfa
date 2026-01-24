#!/usr/bin/env python3
"""
Test all LMP policies with visual output.

Runs each available LMP policy against a single trace and prints
side-by-side comparisons of which tokens get loss.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer

from src.schemas.trace import Trace
from src.schemas.lossmask import LossMask
from src.schemas.tools import ETL_B as etl_b


def _iter_jsonl(path: Path):
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


def _mask_spans(mask: List[float]) -> List[tuple]:
    spans = []
    in_span = False
    start = 0
    for i, m in enumerate(mask):
        if m > 0 and not in_span:
            start = i
            in_span = True
        elif m <= 0 and in_span:
            spans.append((start, i))
            in_span = False
    if in_span:
        spans.append((start, len(mask)))
    return spans


def _visualize_mask(mask: List[float], width: int = 100) -> str:
    """Create ASCII visualization of mask (█ = loss, · = masked)."""
    if len(mask) <= width:
        return "".join("█" if m > 0 else "·" for m in mask)
    
    # Downsample
    step = len(mask) / width
    chars = []
    for i in range(width):
        start = int(i * step)
        end = int((i + 1) * step)
        segment = mask[start:end]
        ratio = sum(segment) / len(segment) if segment else 0
        if ratio > 0.5:
            chars.append("█")
        elif ratio > 0:
            chars.append("▄")
        else:
            chars.append("·")
    return "".join(chars)


def main():
    parser = argparse.ArgumentParser(description="Test all LMP policies visually")
    parser.add_argument("--traces", type=Path, required=True, help="Input trace_v1 JSONL")
    parser.add_argument("--tokenizer", required=True, help="Tokenizer name or path")
    parser.add_argument("--chat-template", type=str, default=None, help="Chat template")
    parser.add_argument("--trace-index", type=int, default=0, help="Which trace to use (0-indexed)")
    parser.add_argument("--max-length", type=int, default=4096)
    
    args = parser.parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    chat_template = args.chat_template
    if chat_template and Path(chat_template).exists():
        chat_template = Path(chat_template).read_text()
    
    lmp_registry = etl_b._default_lmp_registry()
    
    # Load specific trace
    traces = list(_iter_jsonl(args.traces))
    if args.trace_index >= len(traces):
        print(f"Error: trace_index {args.trace_index} >= {len(traces)} traces")
        sys.exit(1)
    
    trace = Trace.from_dict(traces[args.trace_index])
    
    print("=" * 80)
    print(f"TRACE: {trace.id}")
    print(f"Messages: {len(trace.messages)} | Roles: {[m.role for m in trace.messages]}")
    print("=" * 80)
    
    # Render once
    render = etl_b.render_trace(
        trace,
        tokenizer,
        max_length=args.max_length,
        add_generation_prompt=False,
        include_rendered_text=False,
        chat_template=chat_template,
    )
    
    print(f"\nSequence length: {len(render.input_ids)} tokens")
    if render.alignment and render.alignment.message_spans:
        print("Message spans:")
        for span in render.alignment.message_spans:
            print(f"  {span.role}: [{span.token_start}, {span.token_end})")
    print()
    
    # Test each policy
    policies_to_test = [
        "assistant_only",
        "completion_only",
        "full_sequence",
        "cb_full_sequence",
        "tool_calls_only",
        "action_prefix_only",
        "action_commitment",
    ]
    
    print("=" * 80)
    print("LMP POLICY COMPARISON")
    print("=" * 80)
    print("Legend: █ = loss applied | · = masked (no loss) | ▄ = partial\n")
    
    for policy_id in policies_to_test:
        try:
            policy = lmp_registry.get_policy(policy_id)
        except KeyError:
            print(f"  {policy_id}: NOT FOUND IN REGISTRY")
            continue
        
        mask = etl_b._apply_lmp_policy(render, policy)
        spans = _mask_spans(mask)
        unmasked = sum(1 for m in mask if m > 0)
        ratio = unmasked / len(mask) if mask else 0
        
        viz = _visualize_mask(mask, width=70)
        print(f"{policy_id:20s} | {viz} | {unmasked:4d}/{len(mask):4d} ({ratio:.1%})")
        if spans:
            print(f"{'':20s} | spans: {spans[:5]}{'...' if len(spans) > 5 else ''}")
        print()
    
    # Show token breakdown at boundaries
    print("=" * 80)
    print("TOKEN BOUNDARY DETAILS")
    print("=" * 80)
    
    if render.alignment and render.alignment.message_spans:
        for span in render.alignment.message_spans:
            start_tokens = render.input_ids[span.token_start:min(span.token_start + 5, span.token_end)]
            end_tokens = render.input_ids[max(span.token_start, span.token_end - 5):span.token_end]
            start_text = tokenizer.decode(start_tokens, skip_special_tokens=False)[:50]
            end_text = tokenizer.decode(end_tokens, skip_special_tokens=False)[-50:]
            print(f"\n{span.role} [{span.token_start}, {span.token_end}):")
            print(f"  start: {repr(start_text)}")
            print(f"  end:   {repr(end_text)}")


if __name__ == "__main__":
    main()
