#!/usr/bin/env python3
"""
Compare raw Fujitsu data with ETL_A trace output to verify field consistency.

This helps identify if there's a mismatch between how the old pipeline
reads fields directly vs how the new pipeline reads them from traces.

Run with: source .venv/bin/activate && python scripts/compare_raw_vs_trace.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.schemas.trace import Trace


def load_raw_records(path: Path, limit: int = 100) -> dict:
    """Load raw Fujitsu records indexed by record_id."""
    records = {}
    with open(path, "r") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                record_id = record.get("record_id", f"idx_{i}")
                records[record_id] = {
                    "expected_tool": record.get("expected_tool"),
                    "simulated_tool": record.get("simulated_tool"),
                    "combined_query": record.get("combined_query"),
                    "benign_query": record.get("benign_query"),
                    "malicious_injection": record.get("malicious_injection"),
                }
            except json.JSONDecodeError:
                continue
    return records


def load_trace_records(path: Path, limit: int = 100) -> dict:
    """Load ETL_A traces indexed by raw_id (links.raw_id)."""
    traces = {}
    with open(path, "r") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                trace = Trace.from_dict(data)

                # Get the raw_id from links
                raw_id = None
                if trace.links:
                    raw_id = trace.links.raw_id
                if not raw_id:
                    # Try to get from source.record_locator
                    if trace.source and trace.source.record_locator:
                        raw_id = trace.source.record_locator.get("value")

                if not raw_id:
                    raw_id = f"trace_idx_{i}"

                # Extract fields as generate_completions.py would
                expected_tool = None
                simulated_tool = None  # What ETL_A stores in observed_tool

                if trace.tool_attack:
                    expected_tool = trace.tool_attack.expected_tool
                    simulated_tool = trace.tool_attack.observed_tool  # ETL_A stores simulated here
                if trace.signal_hints:
                    expected_tool = expected_tool or trace.signal_hints.expected_tool_name
                    simulated_tool = simulated_tool or trace.signal_hints.observed_tool_name

                # Get user message content
                user_content = None
                for msg in trace.messages:
                    if msg.role == "user":
                        user_content = msg.content
                        break

                # Get injection text
                injection_text = None
                if trace.tool_attack:
                    injection_text = trace.tool_attack.injection_text

                traces[raw_id] = {
                    "trace_id": trace.id,
                    "expected_tool": expected_tool,
                    "simulated_tool_from_observed": simulated_tool,
                    "user_content": user_content,
                    "injection_text": injection_text,
                }
            except Exception as e:
                print(f"Error loading trace at line {i}: {e}")
                continue
    return traces


def compare_records(raw_path: Path, trace_path: Path, limit: int = 50):
    """Compare raw records with trace records."""
    print("Loading raw records...")
    raw_records = load_raw_records(raw_path, limit)
    print(f"  Loaded {len(raw_records)} raw records")

    print("Loading trace records...")
    trace_records = load_trace_records(trace_path, limit)
    print(f"  Loaded {len(trace_records)} traces")

    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)

    matches = 0
    mismatches = 0
    missing = 0

    for raw_id, raw in raw_records.items():
        if raw_id not in trace_records:
            missing += 1
            if missing <= 3:
                print(f"\n⚠ MISSING: raw_id={raw_id} not found in traces")
            continue

        trace = trace_records[raw_id]

        # Compare expected_tool
        expected_match = raw["expected_tool"] == trace["expected_tool"]

        # Compare simulated_tool (raw) with simulated_tool_from_observed (trace)
        simulated_match = raw["simulated_tool"] == trace["simulated_tool_from_observed"]

        # Compare query content
        query_match = raw["combined_query"] == trace["user_content"]

        if expected_match and simulated_match and query_match:
            matches += 1
        else:
            mismatches += 1
            if mismatches <= 5:
                print(f"\n✗ MISMATCH: raw_id={raw_id}")
                print(f"  Expected tool:")
                print(f"    Raw:   {raw['expected_tool']}")
                print(f"    Trace: {trace['expected_tool']}")
                print(f"    Match: {expected_match}")
                print(f"  Simulated tool:")
                print(f"    Raw:   {raw['simulated_tool']}")
                print(f"    Trace (from observed): {trace['simulated_tool_from_observed']}")
                print(f"    Match: {simulated_match}")
                print(f"  Query content match: {query_match}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Matches:    {matches}")
    print(f"  Mismatches: {mismatches}")
    print(f"  Missing:    {missing}")

    if mismatches > 0:
        print(f"\n⚠ Found {mismatches} mismatches between raw data and traces!")
        print("  This could cause the new pipeline to mis-identify tool calls.")
        return False
    else:
        print(f"\n✓ All {matches} compared records match!")
        return True


def test_field_extraction_priority():
    """Test what happens when tool_attack and signal_hints have different values."""
    print("\n" + "=" * 80)
    print("TEST: Field Extraction Priority")
    print("=" * 80)

    # Simulate a trace with different values in tool_attack vs signal_hints
    from src.schemas.trace import TraceToolAttack, SignalHints

    scenarios = [
        {
            "name": "Both set, same values",
            "tool_attack": TraceToolAttack(expected_tool="search_web", observed_tool="execute_code"),
            "signal_hints": SignalHints(expected_tool_name="search_web", observed_tool_name="execute_code"),
            "expected_result": ("search_web", "execute_code"),
        },
        {
            "name": "tool_attack set, signal_hints None",
            "tool_attack": TraceToolAttack(expected_tool="search_web", observed_tool="execute_code"),
            "signal_hints": None,
            "expected_result": ("search_web", "execute_code"),
        },
        {
            "name": "tool_attack None, signal_hints set",
            "tool_attack": None,
            "signal_hints": SignalHints(expected_tool_name="search_web", observed_tool_name="execute_code"),
            "expected_result": ("search_web", "execute_code"),
        },
        {
            "name": "tool_attack has None values, signal_hints has values",
            "tool_attack": TraceToolAttack(expected_tool=None, observed_tool=None),
            "signal_hints": SignalHints(expected_tool_name="search_web", observed_tool_name="execute_code"),
            "expected_result": ("search_web", "execute_code"),
        },
        {
            "name": "CONFLICT: tool_attack and signal_hints have different values",
            "tool_attack": TraceToolAttack(expected_tool="search_web", observed_tool="read_file"),
            "signal_hints": SignalHints(expected_tool_name="retrieve_docs", observed_tool_name="execute_code"),
            "expected_result": ("search_web", "read_file"),  # tool_attack takes priority
        },
    ]

    all_passed = True
    for scenario in scenarios:
        tool_attack = scenario["tool_attack"]
        signal_hints = scenario["signal_hints"]
        expected_expected, expected_simulated = scenario["expected_result"]

        # Reproduce the extraction logic from generate_completions.py
        expected_tool = None
        simulated_tool = None

        if tool_attack:
            expected_tool = tool_attack.expected_tool
            simulated_tool = tool_attack.observed_tool
        if signal_hints:
            expected_tool = expected_tool or signal_hints.expected_tool_name
            simulated_tool = simulated_tool or signal_hints.observed_tool_name

        passed = (expected_tool == expected_expected and simulated_tool == expected_simulated)
        status = "✓" if passed else "✗"

        print(f"\n{status} {scenario['name']}")
        print(f"    expected_tool: {expected_tool} (expected: {expected_expected})")
        print(f"    simulated_tool: {simulated_tool} (expected: {expected_simulated})")

        if not passed:
            all_passed = False

    return all_passed


def main():
    raw_path = Path("/Users/memoozdincer/Documents/Research/Jin/rrfa/data/fujitsu/orchestrator_attacks_combined_deduplicated.jsonl")
    trace_path = Path("/Users/memoozdincer/Documents/Research/Jin/rrfa/data/traces/fujitsu_b4_traces.jsonl")

    if not raw_path.exists():
        print(f"Raw data not found: {raw_path}")
        return 1

    if not trace_path.exists():
        print(f"Trace data not found: {trace_path}")
        return 1

    # Run comparison
    compare_ok = compare_records(raw_path, trace_path, limit=50)

    # Run priority test
    priority_ok = test_field_extraction_priority()

    if compare_ok and priority_ok:
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
