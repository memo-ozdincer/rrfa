#!/usr/bin/env python3
"""
Pipeline Stage Inspector

This script helps inspect data at each stage of the pipeline to verify:
1. Data integrity at each transformation
2. Format-agnostic storage is working correctly
3. No information is being lost

Usage:
    # Inspect a specific trace file
    python scripts/test_pipeline_stages.py --traces data/traces/example.jsonl --limit 3
    
    # Compare before/after format-agnostic conversion
    python scripts/test_pipeline_stages.py --compare-legacy data/traces/legacy.jsonl data/traces/agnostic.jsonl
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.schemas.trace import Trace


def print_section(title: str, width: int = 80):
    """Print a section header."""
    print("\n" + "=" * width)
    print(title)
    print("=" * width)


def print_subsection(title: str, width: int = 80):
    """Print a subsection header."""
    print("\n" + "-" * width)
    print(title)
    print("-" * width)


def inspect_message(msg, indent: str = "  "):
    """Detailed inspection of a message."""
    print(f"{indent}Role: {msg.role}")
    print(f"{indent}Content length: {len(msg.content) if msg.content else 0} chars")
    
    if msg.content:
        content_preview = msg.content[:100].replace("\n", "\\n")
        if len(msg.content) > 100:
            content_preview += "..."
        print(f"{indent}Content preview: {repr(content_preview)}")
        
        # Check for embedded tool syntax
        has_llama = "<|python_tag|>" in msg.content
        has_claude = "<function_calls>" in msg.content or "<invoke>" in msg.content
        has_json = msg.content.strip().startswith("{") and '"name"' in msg.content
        
        if has_llama or has_claude or has_json:
            print(f"{indent}⚠️  WARNING: Tool syntax detected in content!")
            if has_llama:
                print(f"{indent}   - Contains <|python_tag|>")
            if has_claude:
                print(f"{indent}   - Contains Claude XML")
            if has_json:
                print(f"{indent}   - Contains JSON tool call")
        else:
            print(f"{indent}✅ Content is clean (no tool syntax)")
    
    if msg.tool_calls:
        print(f"{indent}Tool calls: {len(msg.tool_calls)}")
        for i, tc in enumerate(msg.tool_calls):
            print(f"{indent}  [{i+1}] {tc.function.name}")
            print(f"{indent}      Arguments (dict): {tc.function.arguments}")
            print(f"{indent}      arguments_json: {repr(tc.function.arguments_json)}")
            
            if tc.function.raw_content:
                raw_preview = tc.function.raw_content[:60].replace("\n", "\\n")
                if len(tc.function.raw_content) > 60:
                    raw_preview += "..."
                print(f"{indent}      raw_content: {repr(raw_preview)}")
            else:
                print(f"{indent}      ⚠️  raw_content: None")


def inspect_trace(trace: Trace, trace_idx: int = 1, verbose: bool = True):
    """Detailed inspection of a single trace."""
    print_section(f"TRACE #{trace_idx}: {trace.id[:60]}...")
    
    # Basic info
    print(f"Completeness: {trace.completeness}")
    print(f"Tier: {trace.tier}")
    print(f"Split: {trace.split}")
    
    if trace.source:
        print(f"\nSource:")
        print(f"  Dataset: {trace.source.dataset}")
        print(f"  Subset: {trace.source.subset}")
        if trace.source.model_id:
            print(f"  Model ID: {trace.source.model_id}")
        if trace.source.model_family:
            print(f"  Model Family: {trace.source.model_family}")
        if trace.source.format_family:
            print(f"  Format Family: {trace.source.format_family}")
    
    if trace.labels:
        print(f"\nLabels:")
        print(f"  Category: {trace.labels.category}")
        if trace.labels.security_outcome:
            print(f"  Security Outcome: {trace.labels.security_outcome}")
        if trace.labels.attack_succeeded is not None:
            print(f"  Attack Succeeded: {trace.labels.attack_succeeded}")
    
    # Messages
    print(f"\nMessages: {len(trace.messages)}")
    for i, msg in enumerate(trace.messages):
        print_subsection(f"Message {i+1}/{len(trace.messages)}")
        if verbose:
            inspect_message(msg)
        else:
            print(f"  Role: {msg.role}, Content: {len(msg.content) if msg.content else 0} chars, "
                  f"Tool calls: {len(msg.tool_calls) if msg.tool_calls else 0}")
    
    # Signal hints
    if trace.signal_hints:
        print_subsection("Signal Hints")
        if trace.signal_hints.raw_format:
            print(f"  Raw format: {trace.signal_hints.raw_format}")
        if trace.signal_hints.raw_assistant_content:
            raw_len = len(trace.signal_hints.raw_assistant_content)
            print(f"  Raw assistant content: {raw_len} chars")
            if verbose:
                preview = trace.signal_hints.raw_assistant_content[:100].replace("\n", "\\n")
                if len(trace.signal_hints.raw_assistant_content) > 100:
                    preview += "..."
                print(f"    Preview: {repr(preview)}")
        if trace.signal_hints.injection_char_span:
            span = trace.signal_hints.injection_char_span
            print(f"  Injection span: msg[{span.message_index}][{span.char_start}:{span.char_end}]")
    
    # Raw metadata
    if trace.raw_metadata:
        print_subsection("Raw Metadata")
        if trace.raw_metadata.generation_params:
            print(f"  Generation params: {trace.raw_metadata.generation_params}")
        if trace.raw_metadata.tool_schema_ref:
            print(f"  Tool schema: {trace.raw_metadata.tool_schema_ref}")
    
    # Data integrity checks
    print_subsection("Data Integrity Checks")
    
    checks = []
    
    # Check for assistant messages
    has_assistant = any(m.role == "assistant" for m in trace.messages)
    checks.append(("Has assistant message", has_assistant))
    
    # Check completeness matches
    completeness_matches = (trace.completeness == "complete") == has_assistant
    checks.append(("Completeness field matches content", completeness_matches))
    
    # Check tier matches completeness
    tier_matches = (trace.tier == "B2") == (trace.completeness == "complete")
    checks.append(("Tier matches completeness", tier_matches))
    
    # Check for tool calls
    has_tool_calls = any(m.tool_calls for m in trace.messages)
    if has_tool_calls:
        # Check tool call structure
        for msg in trace.messages:
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    checks.append((f"Tool '{tc.function.name}' has arguments dict", 
                                   tc.function.arguments is not None))
                    checks.append((f"Tool '{tc.function.name}' has arguments_json", 
                                   tc.function.arguments_json is not None))
                    # raw_content is optional but recommended
                    if tc.function.raw_content:
                        checks.append((f"Tool '{tc.function.name}' has raw_content", True))
                    
                    # Check content is clean (no embedded syntax)
                    if msg.content:
                        has_syntax = ("<|python_tag|>" in msg.content or 
                                      "<function_calls>" in msg.content or
                                      "<invoke>" in msg.content)
                        checks.append((f"Message content is clean (no tool syntax)", not has_syntax))
    
    # Print check results
    for check_name, check_result in checks:
        status = "✅" if check_result else "❌"
        print(f"  {status} {check_name}")


def inspect_traces(trace_path: Path, limit: int = None, verbose: bool = True):
    """Inspect traces from a JSONL file."""
    print_section(f"INSPECTING: {trace_path}")
    
    traces = []
    with open(trace_path, "r") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                traces.append(Trace.from_dict(data))
                if limit and len(traces) >= limit:
                    break
            except Exception as e:
                print(f"⚠️  Error loading trace at line {i+1}: {e}")
    
    print(f"\nLoaded {len(traces)} traces")
    
    # Summary statistics
    completeness_counts = {}
    tier_counts = {}
    dataset_counts = {}
    
    for trace in traces:
        completeness_counts[trace.completeness] = completeness_counts.get(trace.completeness, 0) + 1
        tier_counts[trace.tier] = tier_counts.get(trace.tier, 0) + 1
        if trace.source:
            dataset_counts[trace.source.dataset] = dataset_counts.get(trace.source.dataset, 0) + 1
    
    print("\nSummary Statistics:")
    print(f"  Completeness: {completeness_counts}")
    print(f"  Tiers: {tier_counts}")
    print(f"  Datasets: {dataset_counts}")
    
    # Inspect individual traces
    for i, trace in enumerate(traces):
        inspect_trace(trace, i + 1, verbose=verbose)
    
    return traces


def compare_traces(legacy_path: Path, agnostic_path: Path, limit: int = 3):
    """Compare legacy vs format-agnostic traces."""
    print_section("COMPARING LEGACY vs FORMAT-AGNOSTIC TRACES")
    
    print("\nLoading legacy traces...")
    legacy_traces = []
    with open(legacy_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            legacy_traces.append(Trace.from_dict(json.loads(line)))
            if len(legacy_traces) >= limit:
                break
    
    print(f"Loaded {len(legacy_traces)} legacy traces")
    
    print("\nLoading format-agnostic traces...")
    agnostic_traces = []
    with open(agnostic_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            agnostic_traces.append(Trace.from_dict(json.loads(line)))
            if len(agnostic_traces) >= limit:
                break
    
    print(f"Loaded {len(agnostic_traces)} format-agnostic traces")
    
    # Compare
    for i in range(min(len(legacy_traces), len(agnostic_traces))):
        legacy = legacy_traces[i]
        agnostic = agnostic_traces[i]
        
        print_section(f"COMPARISON #{i+1}")
        
        print("\nLegacy Trace:")
        for j, msg in enumerate(legacy.messages):
            if msg.role == "assistant" and msg.content:
                preview = msg.content[:80].replace("\n", "\\n")
                print(f"  Message {j+1} content: {repr(preview)}...")
                print(f"    Has <|python_tag|>: {'<|python_tag|>' in msg.content}")
                if msg.tool_calls:
                    print(f"    Tool calls: {[tc.function.name for tc in msg.tool_calls]}")
        
        print("\nFormat-Agnostic Trace:")
        for j, msg in enumerate(agnostic.messages):
            if msg.role == "assistant" and msg.content:
                preview = msg.content[:80].replace("\n", "\\n")
                print(f"  Message {j+1} content: {repr(preview)}...")
                print(f"    Has <|python_tag|>: {'<|python_tag|>' in msg.content}")
                if msg.tool_calls:
                    print(f"    Tool calls: {[tc.function.name for tc in msg.tool_calls]}")
                    print(f"    Raw content preserved: {msg.tool_calls[0].function.raw_content is not None}")
        
        print("\nRaw Assistant Content Preserved:")
        if agnostic.signal_hints and agnostic.signal_hints.raw_assistant_content:
            print(f"  ✅ Yes ({len(agnostic.signal_hints.raw_assistant_content)} chars)")
        else:
            print(f"  ❌ No")


def main():
    parser = argparse.ArgumentParser(description="Inspect traces at different pipeline stages")
    parser.add_argument("--traces", type=Path, help="Trace JSONL file to inspect")
    parser.add_argument("--limit", type=int, default=3, help="Number of traces to inspect")
    parser.add_argument("--verbose", action="store_true", help="Show detailed information")
    parser.add_argument("--compare-legacy", nargs=2, metavar=("LEGACY", "AGNOSTIC"),
                        type=Path, help="Compare legacy vs format-agnostic traces")
    
    args = parser.parse_args()
    
    if args.compare_legacy:
        compare_traces(args.compare_legacy[0], args.compare_legacy[1], limit=args.limit)
    elif args.traces:
        inspect_traces(args.traces, limit=args.limit, verbose=args.verbose)
    else:
        parser.error("Either --traces or --compare-legacy must be provided")


if __name__ == "__main__":
    main()
