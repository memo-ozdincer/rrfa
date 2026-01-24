#!/usr/bin/env python3
"""
Test MWCS schedule YAML loading and phase-based LMP overrides.

Creates a sample MWCS schedule YAML, then tests weight interpolation
and LMP override resolution at different training steps.
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml

from src.schemas.trace import Trace, TraceTraining, TraceMixture, TraceSource
from src.schemas.tools import ETL_B as etl_b


SAMPLE_MWCS_SCHEDULE = """
# Sample MWCS schedule for testing curriculum learning
name: test_curriculum
description: Test schedule with warmup -> finetune phases

class_weights:
  fujitsu_b4/tool_flip: 1.0
  fujitsu_b1/poisoning: 0.5
  agentdojo/injection: 0.8

curriculum:
  type: step
  interpolation: linear
  
  phases:
    - name: warmup
      start_step: 0
      end_step: 1000
      class_weights:
        fujitsu_b4/tool_flip: 2.0
        fujitsu_b1/poisoning: 0.5
        agentdojo/injection: 1.0
      lmp_overrides:
        fujitsu_b4/tool_flip: cb_full_sequence
        fujitsu_b1/poisoning: full_sequence
        
    - name: finetune
      start_step: 1000
      end_step: 5000
      class_weights:
        fujitsu_b4/tool_flip: 1.0
        fujitsu_b1/poisoning: 1.0
        agentdojo/injection: 0.5
      lmp_overrides:
        fujitsu_b4/tool_flip: action_prefix_only
        fujitsu_b1/poisoning: assistant_only
        
    - name: polish
      start_step: 5000
      end_step: 10000
      class_weights:
        fujitsu_b4/tool_flip: 0.5
        fujitsu_b1/poisoning: 1.5
        agentdojo/injection: 0.3
      lmp_overrides:
        fujitsu_b4/tool_flip: action_commitment
"""


def _make_dummy_trace(class_id: str, base_weight: float = 1.0) -> Trace:
    """Create a minimal trace for testing MWCS."""
    return Trace(
        id=f"test_trace_{class_id.replace('/', '_')}",
        source=TraceSource(dataset="fujitsu_b4"),
        messages=[],
        split="train",
        training=TraceTraining(
            sample_weight=base_weight,
            mixture=TraceMixture(class_id=class_id),
        ),
    )


def main():
    parser = argparse.ArgumentParser(description="Test MWCS schedule loading and interpolation")
    parser.add_argument("--schedule", type=Path, default=None, help="Path to MWCS schedule YAML (uses built-in sample if not provided)")
    parser.add_argument("--show-schedule", action="store_true", help="Print the sample schedule YAML")
    
    args = parser.parse_args()
    
    if args.show_schedule:
        print("Sample MWCS Schedule YAML:")
        print("=" * 60)
        print(SAMPLE_MWCS_SCHEDULE)
        print("=" * 60)
        return
    
    # Use provided schedule or create temp file with sample
    if args.schedule:
        schedule_path = args.schedule
    else:
        # Write sample to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(SAMPLE_MWCS_SCHEDULE)
            schedule_path = Path(f.name)
        print(f"Using sample schedule (written to {schedule_path})")
    
    print("\n" + "=" * 80)
    print("MWCS SCHEDULE TEST")
    print("=" * 80)
    
    # Load schedule
    schedule_data = etl_b._load_mwcs_schedule_yaml(schedule_path)
    if not schedule_data:
        print("ERROR: Failed to load schedule")
        sys.exit(1)
    
    print(f"\nSchedule name: {schedule_data.get('name')}")
    print(f"Description: {schedule_data.get('description')}")
    
    curriculum = schedule_data.get('curriculum', {})
    phases = curriculum.get('phases', [])
    print(f"Phases: {[p['name'] for p in phases]}")
    print(f"Interpolation: {curriculum.get('interpolation', 'none')}")
    
    # Test different steps
    test_steps = [0, 500, 999, 1000, 2500, 5000, 7500, 10000]
    test_classes = ["fujitsu_b4/tool_flip", "fujitsu_b1/poisoning", "agentdojo/injection"]
    
    print("\n" + "=" * 80)
    print("WEIGHT & LMP OVERRIDE BY STEP")
    print("=" * 80)
    
    for step in test_steps:
        print(f"\n--- Step {step} ---")
        weights, lmp_overrides = etl_b._resolve_yaml_schedule(schedule_data, step)
        
        # Find current phase
        current_phase = None
        for phase in phases:
            if phase['start_step'] <= step < phase['end_step']:
                current_phase = phase['name']
                break
        if current_phase is None and phases:
            if step < phases[0]['start_step']:
                current_phase = phases[0]['name']
            else:
                current_phase = phases[-1]['name']
        
        print(f"Phase: {current_phase}")
        
        for class_id in test_classes:
            weight = weights.get(class_id, 1.0)
            lmp = lmp_overrides.get(class_id) if lmp_overrides else None
            print(f"  {class_id:25s} weight={weight:.3f}  lmp_override={lmp}")
    
    print("\n" + "=" * 80)
    print("END-TO-END TRACE WEIGHT TEST")
    print("=" * 80)
    
    for class_id in test_classes:
        trace = _make_dummy_trace(class_id, base_weight=1.0)
        
        print(f"\nTrace class_id={class_id}, base_weight=1.0")
        for step in [0, 500, 1000, 5000]:
            final_weight, lmp_override = etl_b._apply_mwcs_weight_with_yaml(
                trace, schedule_path, step
            )
            print(f"  step={step:5d} -> final_weight={final_weight:.3f}, lmp_override={lmp_override}")
    
    # Cleanup temp file
    if not args.schedule:
        schedule_path.unlink()
        print(f"\n(Cleaned up temp file)")


if __name__ == "__main__":
    main()
