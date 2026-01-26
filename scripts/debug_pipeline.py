#!/usr/bin/env python3
"""
Debug script to compare old vs new pipeline tool identification logic.

This script loads sample data and prints key field values to identify
where the pipelines diverge in identifying wrong tool calls.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.schemas.trace import Trace

def debug_etl_a_output(trace_path: Path, limit: int = 5):
    """Debug ETL_A output to verify field values."""
    print("=" * 80)
    print(f"DEBUGGING ETL_A OUTPUT: {trace_path}")
    print("=" * 80)

    with open(trace_path, "r") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break

            data = json.loads(line)
            trace = Trace.from_dict(data)

            print(f"\n--- Trace {i+1}: {trace.id[:50]}... ---")
            print(f"  completeness: {trace.completeness}")
            print(f"  tier: {trace.tier}")

            # Check tool_attack fields
            if trace.tool_attack:
                print(f"  tool_attack.expected_tool: {trace.tool_attack.expected_tool}")
                print(f"  tool_attack.observed_tool: {trace.tool_attack.observed_tool}")
                print(f"  tool_attack.injection_text: {trace.tool_attack.injection_text[:50] if trace.tool_attack.injection_text else None}...")
            else:
                print("  tool_attack: None")

            # Check signal_hints fields
            if trace.signal_hints:
                print(f"  signal_hints.expected_tool_name: {trace.signal_hints.expected_tool_name}")
                print(f"  signal_hints.observed_tool_name: {trace.signal_hints.observed_tool_name}")
            else:
                print("  signal_hints: None")

            # Check for assistant messages
            assistant_msgs = [m for m in trace.messages if m.role == "assistant"]
            print(f"  num_assistant_messages: {len(assistant_msgs)}")
            if assistant_msgs:
                for j, msg in enumerate(assistant_msgs):
                    if msg.tool_calls:
                        for tc in msg.tool_calls:
                            print(f"    assistant[{j}] tool_call: {tc.function.name}")
                    else:
                        print(f"    assistant[{j}] tool_call: None")


def debug_ds_trace(trace_path: Path, limit: int = 5):
    """Debug DS trace output after generate_completions."""
    print("=" * 80)
    print(f"DEBUGGING DS TRACES: {trace_path}")
    print("=" * 80)

    with open(trace_path, "r") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break

            data = json.loads(line)
            trace = Trace.from_dict(data)

            print(f"\n--- DS Trace {i+1}: {trace.id[:50]}... ---")

            # Get expected and simulated from metadata
            expected_tool = None
            simulated_tool = None

            if trace.tool_attack:
                expected_tool = trace.tool_attack.expected_tool
                # BUG CHECK: observed_tool in skeleton stores simulated_tool, not actual observed
                simulated_tool = trace.tool_attack.observed_tool
                print(f"  tool_attack.expected_tool: {expected_tool}")
                print(f"  tool_attack.observed_tool (should be simulated): {simulated_tool}")

            if trace.signal_hints:
                print(f"  signal_hints.expected_tool_name: {trace.signal_hints.expected_tool_name}")
                print(f"  signal_hints.observed_tool_name: {trace.signal_hints.observed_tool_name}")

            # Get ACTUAL observed tool from assistant message
            actual_observed = None
            for msg in trace.messages:
                if msg.role == "assistant" and msg.tool_calls:
                    actual_observed = msg.tool_calls[0].function.name
                    break

            print(f"  ACTUAL observed (from message): {actual_observed}")

            # Check consistency
            if actual_observed and simulated_tool:
                if actual_observed == simulated_tool:
                    print(f"  ✓ Flip successful: actual ({actual_observed}) == simulated ({simulated_tool})")
                else:
                    print(f"  ✗ MISMATCH: actual ({actual_observed}) != simulated ({simulated_tool})")
                    print(f"    This trace should NOT be in DS!")


def compare_old_vs_new_ds(old_ds_path: Path, new_ds_path: Path, limit: int = 20):
    """Compare old DS format vs new DS format."""
    print("=" * 80)
    print("COMPARING OLD VS NEW DS FORMAT")
    print("=" * 80)

    # Load old format samples
    old_samples = {}
    with open(old_ds_path, "r") as f:
        for line in f:
            data = json.loads(line)
            # Old format has "labels" with expected/observed/simulated
            sample_id = data.get("id", "")
            labels = data.get("labels", {})
            old_samples[sample_id] = {
                "expected": labels.get("expected_tool"),
                "observed": labels.get("observed_tool"),
                "simulated": labels.get("simulated_tool"),
            }

    print(f"Old DS samples: {len(old_samples)}")

    # Load new format traces
    new_traces = {}
    with open(new_ds_path, "r") as f:
        for line in f:
            data = json.loads(line)
            trace = Trace.from_dict(data)

            expected = None
            simulated = None
            if trace.tool_attack:
                expected = trace.tool_attack.expected_tool
                simulated = trace.tool_attack.observed_tool  # Note: stores simulated

            # Get actual observed from message
            actual_observed = None
            for msg in trace.messages:
                if msg.role == "assistant" and msg.tool_calls:
                    actual_observed = msg.tool_calls[0].function.name
                    break

            new_traces[trace.id] = {
                "expected": expected,
                "metadata_observed": simulated,  # What's in tool_attack.observed_tool
                "actual_observed": actual_observed,  # What model actually called
            }

    print(f"New DS traces: {len(new_traces)}")

    # Show first few new traces
    print("\nFirst few new DS traces:")
    for i, (tid, vals) in enumerate(list(new_traces.items())[:limit]):
        print(f"  {tid[:40]}...")
        print(f"    expected: {vals['expected']}")
        print(f"    metadata_observed (should be simulated): {vals['metadata_observed']}")
        print(f"    actual_observed (from message): {vals['actual_observed']}")

        # Check if actual matches metadata
        if vals['actual_observed'] != vals['metadata_observed']:
            print(f"    ⚠️  DISCREPANCY: actual != metadata_observed")


def analyze_dr_generation_logic(ds_trace_path: Path, limit: int = 10):
    """
    Analyze how DR generation would work with these DS traces.

    Key question: Where does expected_tool come from for DR comparison?
    """
    print("=" * 80)
    print("ANALYZING DR GENERATION LOGIC")
    print("=" * 80)

    with open(ds_trace_path, "r") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break

            data = json.loads(line)
            trace = Trace.from_dict(data)

            print(f"\n--- DS Trace for DR input {i+1} ---")

            # Simulate what build_dr_from_ds does:
            # Get expected_tool
            expected_tool = None
            if trace.tool_attack:
                expected_tool = trace.tool_attack.expected_tool
            if trace.signal_hints:
                expected_tool = expected_tool or trace.signal_hints.expected_tool_name

            print(f"  expected_tool (for DR comparison): {expected_tool}")

            # What was the simulated tool for DS?
            simulated_tool = None
            if trace.tool_attack:
                simulated_tool = trace.tool_attack.observed_tool
            print(f"  simulated_tool (from DS): {simulated_tool}")

            # What did the model actually call in DS?
            ds_actual = None
            for msg in trace.messages:
                if msg.role == "assistant" and msg.tool_calls:
                    ds_actual = msg.tool_calls[0].function.name
                    break
            print(f"  DS actual call: {ds_actual}")

            # For DR, if model calls expected_tool -> correct_behavior
            # If model calls something else -> wrong_tool
            print(f"  DR will be 'correct' if model calls: {expected_tool}")
            print(f"  DR will be 'wrong' if model calls anything else")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Debug pipeline differences")
    parser.add_argument("--etl-a-output", type=Path, help="ETL_A output (skeletons)")
    parser.add_argument("--new-ds", type=Path, help="New pipeline DS output")
    parser.add_argument("--old-ds", type=Path, help="Old pipeline DS output")
    parser.add_argument("--limit", type=int, default=5, help="Number of samples to show")
    args = parser.parse_args()

    if args.etl_a_output and args.etl_a_output.exists():
        debug_etl_a_output(args.etl_a_output, args.limit)

    if args.new_ds and args.new_ds.exists():
        debug_ds_trace(args.new_ds, args.limit)
        analyze_dr_generation_logic(args.new_ds, args.limit)

    if args.old_ds and args.new_ds and args.old_ds.exists() and args.new_ds.exists():
        compare_old_vs_new_ds(args.old_ds, args.new_ds, args.limit)


if __name__ == "__main__":
    main()
