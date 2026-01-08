"""
Circuit Breaker Data Generation Package

This package provides tools for generating high-quality training data for
Circuit Breakers (Representation Rerouting). The key insight from the CB paper
is that training quality "largely depends on how precisely the data can elicit
the targeted representation."

Key principle: Ds (Circuit Breaker Set) must contain ACTUAL model outputs that
trigger harmful representations, not human-authored templates or labels.

Modules:
    - tool_format: Canonical tool-calling format specification
    - llm_harness: LLM-based generation for real harmful completions
    - ds_generator: Circuit Breaker Set (harmful-state elicitors) generation
    - dr_generator: Retain Set (capability + refusal) generation
    - eval_set: Held-out evaluation set generation
    - quality_gates: Data validation and quality checks
    - schema: JSONL schema definitions for all datasets
"""

__version__ = "1.0.0"
__author__ = "CB Data Team"
