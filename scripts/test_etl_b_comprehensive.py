#!/usr/bin/env python3
"""
Comprehensive ETL_B test script.

Tests all LMP policies across different trace types:
- Fujitsu B4 (skeleton, tool-flip) - needs generation first
- AgentDojo (complete, tool-calling)

Usage:
  # Test with complete traces (AgentDojo)
  python scripts/test_etl_b_comprehensive.py \
      --traces data/traces/agentdojo_traces.jsonl \
      --tokenizer meta-llama/Llama-3.1-8B-Instruct

  # Test with skeleton traces (Fujitsu B4) - use --allow-skeleton
  python scripts/test_etl_b_comprehensive.py \
      --traces data/traces/fujitsu_b4_traces.jsonl \
      --tokenizer meta-llama/Llama-3.1-8B-Instruct \
      --allow-skeleton

  # Test post-generation traces
  python scripts/test_etl_b_comprehensive.py \
      --traces data/traces/fujitsu_b4_ds.jsonl \
      --tokenizer meta-llama/Llama-3.1-8B-Instruct
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer

from src.schemas.trace import Trace
from src.schemas.lossmask import LossMask
from src.schemas.tools import ETL_B as etl_b


# All LMP policies to test
ALL_POLICIES = [
    "assistant_only",
    "completion_only",
    "full_sequence",
    "cb_full_sequence",
    "tool_calls_only",
    "action_prefix_only",
    "action_commitment",
]


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


def _visualize_mask(mask: List[float], width: int = 80) -> str:
    """Create ASCII visualization of mask."""
    if len(mask) <= width:
        return "".join("█" if m > 0 else "·" for m in mask)
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


def _print_header(title: str, char: str = "=") -> None:
    print(f"\n{char * 80}")
    print(title)
    print(char * 80)


def _truncate(text: str, max_chars: int = 200) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _decode_token_range(tokenizer, input_ids: List[int], start: int, end: int, max_chars: int = 100) -> str:
    """Decode a range of tokens."""
    tokens = input_ids[start:end]
    text = tokenizer.decode(tokens, skip_special_tokens=False)
    return _truncate(text, max_chars)


def analyze_trace(
    trace: Trace,
    tokenizer,
    lmp_registry,
    max_length: int,
    chat_template: Optional[str],
    allow_skeleton: bool,
) -> None:
    """Analyze a single trace with all LMP policies."""
    
    is_skeleton = trace.tier == "B1" or trace.completeness == "skeleton"
    
    _print_header(f"TRACE: {trace.id}")
    print(f"Dataset: {trace.source.dataset}")
    print(f"Tier: {trace.tier} | Completeness: {trace.completeness}")
    print(f"Category: {trace.labels.category if trace.labels else 'N/A'}")
    print(f"Messages: {len(trace.messages)} | Roles: {[m.role for m in trace.messages]}")
    
    # Count tool calls
    tool_call_count = sum(
        len(m.tool_calls) if m.tool_calls else 0 
        for m in trace.messages
    )
    print(f"Tool calls: {tool_call_count}")
    
    if is_skeleton:
        print(f"\n⚠️  SKELETON TRACE (no assistant messages)")
        if not allow_skeleton:
            print("   Skipping. Use --allow-skeleton to test.")
            return
        print("   Testing with --allow-skeleton...")
    
    # Show messages
    _print_header("MESSAGES", "-")
    for i, msg in enumerate(trace.messages):
        tc_info = f" [tool_calls={len(msg.tool_calls)}]" if msg.tool_calls else ""
        print(f"[{i}] {msg.role}{tc_info}: {_truncate(msg.content, 150)}")
        if msg.tool_calls:
            for j, tc in enumerate(msg.tool_calls):
                print(f"     └─ tool_call[{j}]: {tc.function.name}({list(tc.function.arguments.keys()) if tc.function.arguments else []})")
    
    # Render
    render = etl_b.render_trace(
        trace,
        tokenizer,
        max_length=max_length,
        add_generation_prompt=False,
        include_rendered_text=True,
        chat_template=chat_template,
    )
    
    _print_header("RENDER SUMMARY", "-")
    print(f"Token count: {len(render.input_ids)}")
    
    # Show alignment
    if render.alignment:
        print("\nMessage spans:")
        for span in render.alignment.message_spans or []:
            print(f"  {span.role} [{span.token_start}, {span.token_end}) = {span.token_end - span.token_start} tokens")
        
        if render.alignment.tool_call_spans:
            print("\nTool call spans:")
            for span in render.alignment.tool_call_spans:
                print(f"  {span.tool_name} [{span.token_start}, {span.token_end}) name_end={span.name_token_end}")
    
    # Show signals
    if render.signals:
        print("\nSignals:")
        if render.signals.injection_spans:
            for span in render.signals.injection_spans:
                print(f"  injection: [{span.token_start}, {span.token_end})")
        if render.signals.action_commitments:
            for ac in render.signals.action_commitments:
                print(f"  commitment: idx={ac.commitment_token_idx} tool={ac.committed_tool}")
    
    # Show special tokens (Llama format check)
    if render.special_tokens:
        st = render.special_tokens
        print(f"\nSpecial tokens:")
        print(f"  BOS position: {st.bos_position}")
        print(f"  EOS positions: {st.eos_positions}")
        print(f"  Python tag positions: {st.python_tag_positions}")
    
    # Check for Llama tool-calling format
    if render.rendered_text:
        has_python_tag = "<|python_tag|>" in render.rendered_text
        has_eom = "<|eom_id|>" in render.rendered_text
        print(f"\nLlama tool format: python_tag={has_python_tag}, eom_id={has_eom}")
    
    # Test all LMP policies
    _print_header("LMP POLICY COMPARISON", "-")
    print("Legend: █ = loss | · = masked | ▄ = partial\n")
    
    for policy_id in ALL_POLICIES:
        try:
            policy = lmp_registry.get_policy(policy_id)
        except KeyError:
            print(f"{policy_id:20s} | NOT IN REGISTRY")
            continue
        
        mask = etl_b._apply_lmp_policy(render, policy)
        spans = _mask_spans(mask)
        unmasked = sum(1 for m in mask if m > 0)
        ratio = unmasked / len(mask) if mask else 0
        
        viz = _visualize_mask(mask, width=50)
        status = "✓" if unmasked > 0 else "⚠️ ZERO"
        print(f"{policy_id:20s} | {viz} | {unmasked:4d}/{len(mask):4d} ({ratio:5.1%}) {status}")
        
        # Show first span details for non-zero masks
        if spans and unmasked > 0:
            first_span = spans[0]
            span_text = _decode_token_range(tokenizer, render.input_ids, first_span[0], min(first_span[0] + 20, first_span[1]))
            print(f"{'':20s}   └─ first span [{first_span[0]}, {first_span[1]}): {repr(span_text)}")
    
    # Show rendered text sample
    if render.rendered_text:
        _print_header("RENDERED TEXT SAMPLE", "-")
        # Show system/user boundary
        print("First 500 chars:")
        print(render.rendered_text[:500])
        print("\n...")
        
        # Show assistant section if present
        if render.alignment and render.alignment.assistant_spans:
            first_asst = render.alignment.assistant_spans[0]
            asst_text = _decode_token_range(tokenizer, render.input_ids, first_asst.token_start, first_asst.token_end, max_chars=500)
            print(f"\nAssistant span [{first_asst.token_start}, {first_asst.token_end}):")
            print(asst_text)


def main():
    parser = argparse.ArgumentParser(description="Comprehensive ETL_B testing")
    parser.add_argument("--traces", type=Path, required=True, help="Input trace_v1 JSONL")
    parser.add_argument("--tokenizer", required=True, help="Tokenizer name or path")
    parser.add_argument("--chat-template", type=str, default=None, help="Chat template file or string")
    parser.add_argument("--max-length", type=int, default=4096, help="Max sequence length")
    parser.add_argument("--limit", type=int, default=1, help="Number of traces per policy to show")
    parser.add_argument("--allow-skeleton", action="store_true", help="Allow skeleton traces")
    parser.add_argument("--trace-index", type=int, default=None, help="Test specific trace index")
    parser.add_argument("--lmp-registry", type=Path, default=None, help="Path to LMP registry JSON")
    
    args = parser.parse_args()
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    
    chat_template = args.chat_template
    if chat_template and Path(chat_template).exists():
        chat_template = Path(chat_template).read_text()
    
    lmp_registry = etl_b._load_lmp_registry_safe(args.lmp_registry)
    print(f"LMP policies available: {lmp_registry.list_policies()}")
    
    # Load traces
    traces = list(_iter_jsonl(args.traces))
    print(f"\nLoaded {len(traces)} traces from {args.traces}")
    
    # Categorize traces
    skeleton_traces = []
    complete_traces = []
    tool_calling_traces = []
    
    for row in traces:
        trace = Trace.from_dict(row)
        if trace.tier == "B1" or trace.completeness == "skeleton":
            skeleton_traces.append(trace)
        else:
            complete_traces.append(trace)
            # Check for tool calls
            has_tools = any(m.tool_calls for m in trace.messages)
            if has_tools:
                tool_calling_traces.append(trace)
    
    print(f"\nTrace breakdown:")
    print(f"  Skeleton (B1): {len(skeleton_traces)}")
    print(f"  Complete (B2): {len(complete_traces)}")
    print(f"  With tool calls: {len(tool_calling_traces)}")
    
    # Select traces to test
    if args.trace_index is not None:
        if args.trace_index >= len(traces):
            print(f"Error: trace_index {args.trace_index} >= {len(traces)}")
            sys.exit(1)
        test_traces = [Trace.from_dict(traces[args.trace_index])]
    else:
        test_traces = []
        
        # Add one from each category if available
        if tool_calling_traces:
            test_traces.append(tool_calling_traces[0])
            print(f"\n✓ Testing tool-calling trace: {tool_calling_traces[0].id}")
        
        if complete_traces and (not tool_calling_traces or complete_traces[0].id != tool_calling_traces[0].id):
            # Add a non-tool-calling complete trace if different
            for t in complete_traces:
                has_tools = any(m.tool_calls for m in t.messages)
                if not has_tools:
                    test_traces.append(t)
                    print(f"✓ Testing text-only complete trace: {t.id}")
                    break
        
        if skeleton_traces and args.allow_skeleton:
            test_traces.append(skeleton_traces[0])
            print(f"✓ Testing skeleton trace: {skeleton_traces[0].id}")
        elif skeleton_traces:
            print(f"⚠️  Skipping {len(skeleton_traces)} skeleton traces (use --allow-skeleton)")
    
    # Analyze each trace
    for trace in test_traces[:args.limit * 3]:  # Up to 3 different trace types
        analyze_trace(
            trace,
            tokenizer,
            lmp_registry,
            args.max_length,
            chat_template,
            args.allow_skeleton,
        )
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
