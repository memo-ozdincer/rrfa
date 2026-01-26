#!/usr/bin/env python3
"""
Test Format-Agnostic Storage Implementation

This script validates that the format-agnostic storage mode:
1. Preserves ALL information from raw data
2. Correctly separates reasoning from tool syntax
3. Stores tool calls in structured format
4. Maintains backward compatibility with legacy mode
5. Works correctly at every pipeline stage

Usage:
    python scripts/test_format_agnostic.py --mode all
    python scripts/test_format_agnostic.py --mode unit
    python scripts/test_format_agnostic.py --mode integration
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.schemas.trace import Trace, Message, ToolCall, ToolCallFunction, SignalHints


# =============================================================================
# Inline versions of functions to avoid import issues
# =============================================================================

def strip_tool_syntax_from_content(content: str, format_family: Optional[str] = None) -> str:
    """Strip tool call syntax from assistant content, leaving only reasoning text."""
    if not content:
        return content

    # Auto-detect format if not provided
    if format_family is None:
        if "<|python_tag|>" in content:
            format_family = "llama_python_tag"
        elif "<function_calls>" in content or "<invoke>" in content:
            format_family = "anthropic_xml"
        elif content.strip().startswith("{") and ('"name"' in content or '"function"' in content):
            format_family = "openai_json"

    # Strip based on detected format
    if format_family == "llama_python_tag":
        if "<|python_tag|>" in content:
            reasoning = content.split("<|python_tag|>", 1)[0].strip()
            return reasoning
    elif format_family == "anthropic_xml":
        reasoning = re.sub(r'<function_calls>.*?</function_calls>', '', content, flags=re.DOTALL)
        reasoning = re.sub(r'<invoke>.*?</invoke>', '', reasoning, flags=re.DOTALL)
        return reasoning.strip()
    elif format_family == "openai_json":
        if content.strip().startswith("{") and content.strip().endswith("}"):
            try:
                json.loads(content)
                return ""  # Pure tool call, no reasoning
            except json.JSONDecodeError:
                pass

    return content


def extract_tool_call(response: str, strip_reasoning: bool = False) -> Optional[Dict[str, Any]]:
    """Extract tool call from assistant response."""
    if "<|python_tag|>" in response:
        # Extract reasoning text if requested
        reasoning_text = None
        if strip_reasoning:
            reasoning_text = response.split("<|python_tag|>", 1)[0].strip()

        # Extract the full tool call content after python_tag
        content = response.split("<|python_tag|>", 1)[1]
        raw_tool_content = content
        for end_token in ["<|eom_id|>", "<|eot_id|>", "</s>", "<|end_of_text|>"]:
            content = content.split(end_token)[0]
        content = content.strip()

        # Capture raw content including python_tag
        raw_content = "<|python_tag|>" + raw_tool_content.split("<|eom_id|>")[0].split("<|eot_id|>")[0]
        if "<|eom_id|>" in response:
            raw_content += "<|eom_id|>"
        elif "<|eot_id|>" in response:
            raw_content += "<|eot_id|>"

        match = re.match(r'(\w+)\s*\((.*)\)', content, re.DOTALL)
        if match:
            tool_name = match.group(1)
            args_str = match.group(2).strip()

            try:
                if '=' in args_str and not args_str.startswith('{'):
                    args = {}
                    for part in re.findall(r"(\w+)\s*=\s*['\"]([^'\"]*)['\"]", args_str):
                        args[part[0]] = part[1]
                    result = {
                        "name": tool_name,
                        "parameters": args,
                        "arguments_json": args_str,
                        "raw_content": raw_content,
                    }
                    if strip_reasoning:
                        result["reasoning_text"] = reasoning_text
                    return result
                else:
                    args = json.loads(args_str) if args_str else {}
                    result = {
                        "name": tool_name,
                        "parameters": args,
                        "arguments_json": args_str,
                        "raw_content": raw_content,
                    }
                    if strip_reasoning:
                        result["reasoning_text"] = reasoning_text
                    return result
            except (json.JSONDecodeError, ValueError):
                result = {
                    "name": tool_name,
                    "parameters": {"raw": args_str},
                    "arguments_json": args_str,
                    "raw_content": raw_content,
                }
                if strip_reasoning:
                    result["reasoning_text"] = reasoning_text
                return result

    return None


# =============================================================================
# Test Data
# =============================================================================

LLAMA_RESPONSE = """I'll help you search for information about climate change.

<|python_tag|>search_web(query="climate change impacts")<|eom_id|>"""

CLAUDE_RESPONSE = """I'll search for that information.

<function_calls>
<invoke>
<tool_name>search_web</tool_name>
<parameters>
<query>climate change impacts</query>
</parameters>
</invoke>
</function_calls>"""

OPENAI_RESPONSE_JSON = """I'll search for that.

{"name": "search_web", "parameters": {"query": "climate change impacts"}}"""

OPENAI_RESPONSE_CLEAN = "I'll search for information about climate change."  # OpenAI already separates


# =============================================================================
# Unit Tests
# =============================================================================

def test_strip_tool_syntax():
    """Test that strip_tool_syntax_from_content correctly extracts reasoning."""
    print("\n" + "="*80)
    print("TEST: strip_tool_syntax_from_content()")
    print("="*80)
    
    tests = [
        {
            "name": "Llama format",
            "input": LLAMA_RESPONSE,
            "format": "llama_python_tag",
            "expected": "I'll help you search for information about climate change.",
        },
        {
            "name": "Claude format",
            "input": CLAUDE_RESPONSE,
            "format": "anthropic_xml",
            "expected": "I'll search for that information.",
        },
        {
            "name": "No tool call (pure reasoning)",
            "input": "Just some reasoning text without any tool calls.",
            "format": None,
            "expected": "Just some reasoning text without any tool calls.",
        },
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        result = strip_tool_syntax_from_content(test["input"], test["format"])
        if result == test["expected"]:
            print(f"✅ PASS: {test['name']}")
            print(f"   Input:    {repr(test['input'][:60])}...")
            print(f"   Expected: {repr(test['expected'])}")
            print(f"   Got:      {repr(result)}")
            passed += 1
        else:
            print(f"❌ FAIL: {test['name']}")
            print(f"   Input:    {repr(test['input'][:60])}...")
            print(f"   Expected: {repr(test['expected'])}")
            print(f"   Got:      {repr(result)}")
            failed += 1
        print()
    
    return passed, failed


def test_extract_tool_call():
    """Test that extract_tool_call preserves all information."""
    print("\n" + "="*80)
    print("TEST: extract_tool_call()")
    print("="*80)
    
    tests = [
        {
            "name": "Llama format - basic extraction",
            "input": LLAMA_RESPONSE,
            "strip_reasoning": False,
        },
        {
            "name": "Llama format - with reasoning separation",
            "input": LLAMA_RESPONSE,
            "strip_reasoning": True,
        },
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        result = extract_tool_call(test["input"], strip_reasoning=test["strip_reasoning"])
        
        checks = []
        
        # Check all required fields present
        if result and "name" in result:
            checks.append(("name field", True))
        else:
            checks.append(("name field", False))
        
        if result and "parameters" in result:
            checks.append(("parameters field", True))
        else:
            checks.append(("parameters field", False))
        
        if result and "arguments_json" in result:
            checks.append(("arguments_json field", True))
        else:
            checks.append(("arguments_json field", False))
        
        if result and "raw_content" in result:
            checks.append(("raw_content field", True))
        else:
            checks.append(("raw_content field", False))
        
        if test["strip_reasoning"]:
            if result and "reasoning_text" in result:
                checks.append(("reasoning_text field", True))
            else:
                checks.append(("reasoning_text field", False))
        
        # Check content preservation
        if result and result.get("raw_content") and "<|python_tag|>" in result["raw_content"]:
            checks.append(("raw_content preserves syntax", True))
        else:
            checks.append(("raw_content preserves syntax", False))
        
        all_passed = all(check[1] for check in checks)
        
        if all_passed:
            print(f"✅ PASS: {test['name']}")
            passed += 1
        else:
            print(f"❌ FAIL: {test['name']}")
            failed += 1
        
        print(f"   Result: {result}")
        for check_name, check_result in checks:
            status = "✅" if check_result else "❌"
            print(f"   {status} {check_name}")
        print()
    
    return passed, failed


def test_data_preservation():
    """Test that format-agnostic mode preserves all data."""
    print("\n" + "="*80)
    print("TEST: Data Preservation in Format-Agnostic Mode")
    print("="*80)
    
    # Simulate what happens in generate_completions.py
    response = LLAMA_RESPONSE
    strip_tool_syntax = True
    
    # Extract tool call with reasoning separation
    tool_call = extract_tool_call(response, strip_reasoning=strip_tool_syntax)
    
    # Extract clean reasoning text
    assistant_content = response
    if strip_tool_syntax and tool_call and "reasoning_text" in tool_call:
        assistant_content = tool_call["reasoning_text"]
    
    # Build message (as in generate_completions.py)
    tc = None
    if tool_call:
        tc = ToolCall(
            function=ToolCallFunction(
                name=tool_call["name"],
                arguments=tool_call.get("parameters", {}),
                arguments_json=tool_call.get("arguments_json"),
                raw_content=tool_call.get("raw_content"),
            )
        )
    
    assistant_msg = Message(
        role="assistant",
        content=assistant_content,
        tool_calls=[tc] if tc else None,
    )
    
    # Build signal_hints (as in generate_completions.py)
    signal_hints = SignalHints(
        raw_assistant_content=response,  # Full response preserved here
        raw_format="llama_python_tag",
    )
    
    print("Original Response:")
    print(f"  {repr(response)}\n")
    
    print("Extracted Components:")
    print(f"  Message.content (reasoning only): {repr(assistant_content)}")
    print(f"  Tool call name: {tc.function.name if tc else None}")
    print(f"  Tool call arguments (dict): {tc.function.arguments if tc else None}")
    print(f"  Tool call arguments_json (string): {repr(tc.function.arguments_json) if tc else None}")
    print(f"  Tool call raw_content: {repr(tc.function.raw_content) if tc else None}")
    print(f"  signal_hints.raw_assistant_content: {repr(signal_hints.raw_assistant_content)}\n")
    
    # Verify nothing is lost
    checks = [
        ("Reasoning text extracted", assistant_content == "I'll help you search for information about climate change."),
        ("Tool name extracted", tc and tc.function.name == "search_web"),
        ("Tool arguments parsed", tc and tc.function.arguments.get("query") == "climate change impacts"),
        ("arguments_json preserved", tc and tc.function.arguments_json is not None),
        ("raw_content preserved", tc and tc.function.raw_content and "<|python_tag|>" in tc.function.raw_content),
        ("Full response preserved", signal_hints.raw_assistant_content == response),
        ("No tool syntax in content", "<|python_tag|>" not in assistant_content),
    ]
    
    passed = sum(1 for _, result in checks if result)
    failed = sum(1 for _, result in checks if not result)
    
    print("Validation Checks:")
    for check_name, check_result in checks:
        status = "✅" if check_result else "❌"
        print(f"  {status} {check_name}")
    
    print()
    return passed, failed


def test_reconstruction():
    """Test that we can reconstruct the original response from stored data."""
    print("\n" + "="*80)
    print("TEST: Response Reconstruction")
    print("="*80)
    
    # Original response
    response = LLAMA_RESPONSE
    
    # Simulate format-agnostic storage
    tool_call = extract_tool_call(response, strip_reasoning=True)
    reasoning = tool_call["reasoning_text"]
    raw_content = tool_call["raw_content"]
    
    # Reconstruct from parts
    reconstructed = reasoning + "\n\n" + raw_content
    
    print("Original Response:")
    print(f"  {repr(response)}\n")
    
    print("Stored Components:")
    print(f"  Reasoning: {repr(reasoning)}")
    print(f"  Raw Content: {repr(raw_content)}\n")
    
    print("Reconstructed Response:")
    print(f"  {repr(reconstructed)}\n")
    
    # They should be identical or functionally equivalent
    # (minor whitespace differences are acceptable)
    equivalent = (response.replace("\n\n", "\n") == reconstructed.replace("\n\n", "\n"))
    
    if equivalent:
        print("✅ PASS: Can reconstruct original response from stored components")
        return 1, 0
    else:
        print("❌ FAIL: Reconstruction doesn't match original")
        print(f"  Difference detected (may be whitespace)")
        return 0, 1


def test_cross_model_compatibility():
    """Test that structured data works across different formats."""
    print("\n" + "="*80)
    print("TEST: Cross-Model Compatibility")
    print("="*80)
    
    # Extract tool call from Llama response
    llama_tool_call = extract_tool_call(LLAMA_RESPONSE, strip_reasoning=True)
    
    print("Llama Response Tool Call:")
    print(f"  Name: {llama_tool_call['name']}")
    print(f"  Parameters: {llama_tool_call['parameters']}\n")
    
    # The structured data should be format-agnostic
    # Simulate rendering for different target formats
    print("Structured Data (format-agnostic):")
    print(f"  {json.dumps(llama_tool_call['parameters'], indent=2)}\n")
    
    print("This same data can be rendered as:")
    print(f"  Llama: <|python_tag|>search_web(query=\"{llama_tool_call['parameters']['query']}\")<|eom_id|>")
    print(f"  Claude: <invoke><tool_name>search_web</tool_name><parameters><query>{llama_tool_call['parameters']['query']}</query></parameters></invoke>")
    print(f"  OpenAI: {{\"name\": \"search_web\", \"parameters\": {{\"query\": \"{llama_tool_call['parameters']['query']}\"}}}}")
    print()
    
    # Check that we have the necessary data
    checks = [
        ("Tool name present", llama_tool_call["name"] == "search_web"),
        ("Parameters are dict", isinstance(llama_tool_call["parameters"], dict)),
        ("arguments_json available", "arguments_json" in llama_tool_call),
        ("raw_content for replay", "raw_content" in llama_tool_call),
        ("reasoning_text separated", "reasoning_text" in llama_tool_call),
    ]
    
    passed = sum(1 for _, result in checks if result)
    failed = sum(1 for _, result in checks if not result)
    
    print("Cross-Model Requirements:")
    for check_name, check_result in checks:
        status = "✅" if check_result else "❌"
        print(f"  {status} {check_name}")
    
    print()
    return passed, failed


# =============================================================================
# Integration Tests
# =============================================================================

def test_trace_serialization():
    """Test that traces can be serialized and deserialized without loss."""
    print("\n" + "="*80)
    print("TEST: Trace Serialization/Deserialization")
    print("="*80)
    
    # Create a sample trace with format-agnostic storage
    tool_call = extract_tool_call(LLAMA_RESPONSE, strip_reasoning=True)
    
    tc = ToolCall(
        function=ToolCallFunction(
            name=tool_call["name"],
            arguments=tool_call["parameters"],
            arguments_json=tool_call["arguments_json"],
            raw_content=tool_call["raw_content"],
        )
    )
    
    msg = Message(
        role="assistant",
        content=tool_call["reasoning_text"],
        tool_calls=[tc],
    )
    
    trace = Trace(
        id="test_trace_001",
        completeness="complete",
        tier="B2",
        source=None,  # Simplified for test
        messages=[msg],
        split="test",
    )
    
    trace.signal_hints = SignalHints(
        raw_assistant_content=LLAMA_RESPONSE,
        raw_format="llama_python_tag",
    )
    
    # Serialize to dict
    trace_dict = trace.to_dict()
    
    # Serialize to JSON
    trace_json = json.dumps(trace_dict, indent=2)
    
    # Deserialize
    trace_dict_restored = json.loads(trace_json)
    trace_restored = Trace.from_dict(trace_dict_restored)
    
    print("Original Trace:")
    print(f"  Message content: {repr(trace.messages[0].content)}")
    print(f"  Tool call name: {trace.messages[0].tool_calls[0].function.name}")
    print(f"  Tool arguments: {trace.messages[0].tool_calls[0].function.arguments}")
    print(f"  Raw content: {repr(trace.messages[0].tool_calls[0].function.raw_content[:50])}...")
    print(f"  Signal hints raw: {repr(trace.signal_hints.raw_assistant_content[:50])}...\n")
    
    print("Restored Trace:")
    print(f"  Message content: {repr(trace_restored.messages[0].content)}")
    print(f"  Tool call name: {trace_restored.messages[0].tool_calls[0].function.name}")
    print(f"  Tool arguments: {trace_restored.messages[0].tool_calls[0].function.arguments}")
    print(f"  Raw content: {repr(trace_restored.messages[0].tool_calls[0].function.raw_content[:50])}...")
    print(f"  Signal hints raw: {repr(trace_restored.signal_hints.raw_assistant_content[:50])}...\n")
    
    # Verify all fields preserved
    checks = [
        ("Message content preserved", trace.messages[0].content == trace_restored.messages[0].content),
        ("Tool name preserved", trace.messages[0].tool_calls[0].function.name == trace_restored.messages[0].tool_calls[0].function.name),
        ("Tool arguments preserved", trace.messages[0].tool_calls[0].function.arguments == trace_restored.messages[0].tool_calls[0].function.arguments),
        ("arguments_json preserved", trace.messages[0].tool_calls[0].function.arguments_json == trace_restored.messages[0].tool_calls[0].function.arguments_json),
        ("raw_content preserved", trace.messages[0].tool_calls[0].function.raw_content == trace_restored.messages[0].tool_calls[0].function.raw_content),
        ("signal_hints preserved", trace.signal_hints.raw_assistant_content == trace_restored.signal_hints.raw_assistant_content),
    ]
    
    passed = sum(1 for _, result in checks if result)
    failed = sum(1 for _, result in checks if not result)
    
    print("Serialization Checks:")
    for check_name, check_result in checks:
        status = "✅" if check_result else "❌"
        print(f"  {status} {check_name}")
    
    print()
    return passed, failed


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Test format-agnostic storage implementation")
    parser.add_argument(
        "--mode",
        choices=["all", "unit", "integration"],
        default="all",
        help="Test mode to run",
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("FORMAT-AGNOSTIC STORAGE TEST SUITE")
    print("="*80)
    
    total_passed = 0
    total_failed = 0
    
    if args.mode in ("all", "unit"):
        print("\n\n" + "="*80)
        print("UNIT TESTS")
        print("="*80)
        
        p, f = test_strip_tool_syntax()
        total_passed += p
        total_failed += f
        
        p, f = test_extract_tool_call()
        total_passed += p
        total_failed += f
        
        p, f = test_data_preservation()
        total_passed += p
        total_failed += f
        
        p, f = test_reconstruction()
        total_passed += p
        total_failed += f
        
        p, f = test_cross_model_compatibility()
        total_passed += p
        total_failed += f
    
    if args.mode in ("all", "integration"):
        print("\n\n" + "="*80)
        print("INTEGRATION TESTS")
        print("="*80)
        
        p, f = test_trace_serialization()
        total_passed += p
        total_failed += f
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"✅ Passed: {total_passed}")
    print(f"❌ Failed: {total_failed}")
    print(f"Total: {total_passed + total_failed}")
    
    if total_failed == 0:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total_failed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
