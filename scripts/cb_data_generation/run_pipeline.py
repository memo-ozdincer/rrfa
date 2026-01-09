#!/usr/bin/env python3
"""
Circuit Breaker Data Generation - Main Entry Point

Complete pipeline for generating CB training data:
1. Generate Ds (Circuit Breaker Set) with real LLM completions
2. Generate Dr (Retain Set) from benign traces + refusals
3. Generate Eval Set for testing
4. Run quality gates to validate

Usage:
    # Full pipeline
    python scripts/cb_data_generation/run_pipeline.py --all

    # Just generate (no training data exists yet)
    python scripts/cb_data_generation/run_pipeline.py --generate

    # Just validate existing data
    python scripts/cb_data_generation/run_pipeline.py --validate

    # Generate with specific model
    python scripts/cb_data_generation/run_pipeline.py --all \
        --model meta-llama/Llama-3.1-70B-Instruct \
        --backend vllm \
        --tensor-parallel 4
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]


def run_command(cmd: list, description: str) -> bool:
    """Run a command and return success status."""
    logger.info(f"Running: {description}")
    logger.info(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        logger.error(f"FAILED: {description}")
        return False
    
    logger.info(f"SUCCESS: {description}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Circuit Breaker Data Generation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # Pipeline stages
    parser.add_argument("--all", action="store_true", help="Run full pipeline")
    parser.add_argument("--generate", action="store_true", help="Run generation only")
    parser.add_argument("--validate", action="store_true", help="Run validation only")
    parser.add_argument("--generate-ds", action="store_true", help="Generate Ds only")
    parser.add_argument("--generate-dr", action="store_true", help="Generate Dr only")
    parser.add_argument("--generate-eval", action="store_true", help="Generate eval only")
    
    # Model settings
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--backend", choices=["vllm", "transformers"], default="vllm")
    parser.add_argument("--tensor-parallel", type=int, default=1)
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    
    # Generation settings
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    
    # Paths
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE_DIR / "data" / "circuit_breakers",
    )
    
    # Options
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-generation", action="store_true", 
                       help="Skip LLM generation, use existing completions only")
    parser.add_argument("--strict", action="store_true",
                       help="Fail validation on any warning")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from existing output, skip already-generated IDs")
    parser.add_argument("--batch-size", type=int, default=64,
                       help="Batch size for LLM generation (default: 64)")
    
    args = parser.parse_args()
    
    # Determine what to run
    run_ds = args.all or args.generate or args.generate_ds
    run_dr = args.all or args.generate or args.generate_dr
    run_eval = args.all or args.generate or args.generate_eval
    run_validation = args.all or args.validate
    
    if not any([run_ds, run_dr, run_eval, run_validation]):
        parser.print_help()
        return 1
    
    scripts_dir = BASE_DIR / "scripts" / "cb_data_generation"
    results = []
    
    # Generate Ds
    if run_ds:
        cmd = [
            sys.executable,
            str(scripts_dir / "generate_ds.py"),
            "--backend", args.backend,
            "--model", args.model,
            "--temperature", str(args.temperature),
            "--num-samples", str(args.num_samples),
            "--output", str(args.output_dir / "ds" / "circuit_breaker_set.jsonl"),
        ]
        
        if args.backend == "vllm":
            cmd.extend(["--tensor-parallel", str(args.tensor_parallel)])
        else:
            if args.load_in_8bit:
                cmd.append("--load-in-8bit")
            if args.load_in_4bit:
                cmd.append("--load-in-4bit")
        
        if args.limit:
            cmd.extend(["--limit", str(args.limit)])
        if args.dry_run:
            cmd.append("--dry-run")
        if args.skip_generation:
            cmd.append("--skip-generation")
        if args.resume:
            cmd.append("--resume")
        if args.batch_size:
            cmd.extend(["--batch-size", str(args.batch_size)])
        
        success = run_command(cmd, "Generate Circuit Breaker Set (Ds)")
        results.append(("Ds Generation", success))
    
    # Generate Dr
    if run_dr:
        cmd = [
            sys.executable,
            str(scripts_dir / "generate_dr.py"),
            "--output", str(args.output_dir / "dr" / "retain_set.jsonl"),
        ]
        
        if args.limit:
            cmd.extend(["--limit", str(args.limit)])
        if args.dry_run:
            cmd.append("--dry-run")
        
        success = run_command(cmd, "Generate Retain Set (Dr)")
        results.append(("Dr Generation", success))
    
    # Generate Eval
    if run_eval:
        cmd = [
            sys.executable,
            str(scripts_dir / "generate_eval.py"),
            "--output", str(args.output_dir / "eval" / "eval_set.jsonl"),
            "--include-forced-calls",
        ]
        
        if args.limit:
            cmd.extend(["--limit", str(args.limit)])
        if args.dry_run:
            cmd.append("--dry-run")
        
        success = run_command(cmd, "Generate Evaluation Set")
        results.append(("Eval Generation", success))
    
    # Validate
    if run_validation and not args.dry_run:
        cmd = [
            sys.executable,
            str(scripts_dir / "quality_gates.py"),
            "--ds", str(args.output_dir / "ds" / "circuit_breaker_set.jsonl"),
            "--dr", str(args.output_dir / "dr" / "retain_set.jsonl"),
            "--report", str(args.output_dir / "quality_report.json"),
        ]
        
        if args.strict:
            cmd.append("--strict")
        
        success = run_command(cmd, "Quality Gates Validation")
        results.append(("Quality Gates", success))
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("PIPELINE SUMMARY")
    logger.info("="*60)
    
    all_passed = True
    for name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        logger.info(f"  {name}: {status}")
        if not success:
            all_passed = False
    
    if all_passed:
        logger.info("\n✓ All pipeline stages completed successfully")
        logger.info(f"\nOutput directory: {args.output_dir}")
        logger.info("\nNext steps:")
        logger.info("  1. Review data/circuit_breakers/quality_report.json")
        logger.info("  2. Run: python scripts/train_circuit_breaker.py --data-path <path_to_ds>")
        return 0
    else:
        logger.error("\n✗ Some pipeline stages failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
