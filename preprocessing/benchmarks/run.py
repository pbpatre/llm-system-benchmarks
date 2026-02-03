#!/usr/bin/env python3
"""
===================================================================================
LLM Pre-Processing Bottleneck Benchmark Suite - CLI Runner
===================================================================================

A production-grade benchmarking suite to analyze the performance difference between
Jinja2 Templating (Python-bound, GIL-constrained) and BPE Tokenization (Rust-bound,
GIL-released) in LLM inference pipelines.

This suite proves that Python's Global Interpreter Lock (GIL) and Jinja2 templating
are major bottlenecks in high-throughput inference servers.

Experiments:
    1. The Anatomy of a Request (Baseline) - Breakdown latency for a single request
    2. The "Chat History" Tax (Scaling) - Show Jinja slowdown with conversation turns
    3. The Concurrency Test (GPU Wait Time) - Show Jinja % increasing with thread count
    4. Threading vs The GIL (Detailed Profiling) - Prove Rust releases GIL, Python doesn't

Usage:
    # Run full suite
    python -m preprocessing.benchmarks.run
    
    # Quick mode (~5 min)
    python -m preprocessing.benchmarks.run --quick
    
    # Run specific experiments
    python -m preprocessing.benchmarks.run --exp 1 3 4
    
    # Run single experiment directly
    python -m preprocessing.benchmarks.experiments.exp1_baseline
    
    # Or from the benchmarks directory:
    python run.py --exp 3 --quick

Environment Variables:
    HF_TOKEN: HuggingFace authentication token (required for some models)
"""

import os
import argparse
from typing import List

from .common import CONFIG, get_quick_config
from .suite import BenchmarkSuite


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="LLM Pre-Processing Bottleneck Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                      # Run all experiments (full mode)
  python run.py --quick              # Quick mode (~5 min)
  python run.py --exp 3              # Run only experiment 3
  python run.py --exp 1 3 4          # Run experiments 1, 3, and 4
  python run.py --exp 4 --no-plots   # Run experiment 4, skip plotting
  
Experiments:
  1 - Baseline: Single request latency breakdown
  2 - Scaling: Jinja overhead vs conversation turns
  3 - Concurrency: GPU wait time vs thread count
  4 - Threading: GIL blocking proof
        """
    )
    
    parser.add_argument(
        "--quick", 
        action="store_true", 
        help="Quick mode with reduced iterations (~5 min total)"
    )
    
    parser.add_argument(
        "--exp", 
        nargs="+", 
        type=int, 
        choices=[1, 2, 3, 4],
        help="Specific experiment(s) to run (1-4). Default: run all"
    )
    
    parser.add_argument(
        "--output", 
        type=str, 
        default="output",
        help="Output directory for plots and CSV (default: output/)"
    )
    
    parser.add_argument(
        "--no-plots", 
        action="store_true",
        help="Skip generating visualization plots"
    )
    
    parser.add_argument(
        "--list", 
        action="store_true",
        help="List available experiments and exit"
    )
    
    return parser.parse_args()


def list_experiments():
    """Print list of available experiments."""
    print("\n" + "="*70)
    print("AVAILABLE EXPERIMENTS")
    print("="*70)
    print("""
Experiment 1: The Anatomy of a Request (Baseline)
    - Breaks down preprocessing latency into Jinja, Tokenization, and Collation
    - Use: python -m preprocessing.benchmarks.experiments.exp1_baseline

Experiment 2: The "Chat History" Tax (Scaling)
    - Shows Jinja overhead increases with conversation turns (not tokens)
    - Use: python -m preprocessing.benchmarks.experiments.exp2_scaling

Experiment 3: The Concurrency Test (GPU Wait Time)
    - Simulates production throughput, measures GPU idle time
    - Use: python -m preprocessing.benchmarks.experiments.exp3_concurrency

Experiment 4: Threading vs The GIL (Detailed Profiling)
    - Proves Rust tokenizers release GIL, Jinja does not
    - Use: python -m preprocessing.benchmarks.experiments.exp4_threading

Run all: python -m preprocessing.benchmarks.run
""")


def main():
    """Main entry point for the benchmark suite CLI."""
    args = parse_args()
    
    # Handle --list
    if args.list:
        list_experiments()
        return
    
    # Select configuration
    if args.quick:
        config = get_quick_config()
        print("🚀 Running in QUICK mode (~5 min)")
    else:
        config = CONFIG.copy()
        print("🚀 Running in FULL mode")
    
    # Create output directory if needed
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    # Initialize suite
    suite = BenchmarkSuite(config)
    
    # Run experiments
    if args.exp:
        # Run specific experiments
        print(f"   Running experiments: {args.exp}")
        suite.run_experiments(args.exp)
        
        # Generate plots (unless disabled)
        if not args.no_plots:
            suite.generate_plots(args.output)
        
        # Save results
        csv_path = os.path.join(args.output, config.get("output_csv", "results.csv"))
        suite.save_results(csv_path)
        
        print("\n" + "="*70)
        print("📊 EXPERIMENTS COMPLETE")
        print("="*70)
    else:
        # Run all experiments
        suite.run_all(args.output)


if __name__ == "__main__":
    main()
