"""
===================================================================================
LLM Pre-Processing Bottleneck Benchmark Suite
===================================================================================

A production-grade benchmarking suite to analyze the performance difference between
Jinja2 Templating (Python-bound, GIL-constrained) and BPE Tokenization (Rust-bound,
GIL-released) in LLM inference pipelines.

This suite proves that Python's Global Interpreter Lock (GIL) and Jinja2 templating
are major bottlenecks in high-throughput inference servers.

Key Metric: GPU Wait Time
- The time GPU waits idle while CPU preprocessing completes
- This is the end-to-end wall time from first request to batch-ready-for-GPU

Experiments:
    1. The Anatomy of a Request (Baseline)
    2. The "Chat History" Tax (Scaling)
    3. The Concurrency Test (GPU Wait Time)
    4. Threading vs The GIL (Detailed Profiling)

Quick Start:
    from preprocessing.benchmarks import BenchmarkSuite
    
    # Run all experiments
    suite = BenchmarkSuite()
    suite.run_all()
    
    # Run specific experiments
    suite = BenchmarkSuite()
    suite.run_experiment(1)  # Baseline
    suite.run_experiment(3)  # Concurrency
    suite.generate_plots()

CLI Usage:
    python -m preprocessing.benchmarks.run              # Full suite
    python -m preprocessing.benchmarks.run --quick      # Quick mode
    python -m preprocessing.benchmarks.run --exp 1 3 4  # Specific experiments
    python -m preprocessing.benchmarks.run --list       # List experiments

Individual Experiments (for blog posts):
    python -m preprocessing.benchmarks.experiments.exp1_baseline
    python -m preprocessing.benchmarks.experiments.exp2_scaling
    python -m preprocessing.benchmarks.experiments.exp3_concurrency
    python -m preprocessing.benchmarks.experiments.exp4_threading
"""

from .common import CONFIG, get_quick_config, ExperimentResult
from .suite import BenchmarkSuite
from .experiments import (
    Experiment1Baseline,
    Experiment2Scaling,
    Experiment3Concurrency,
    Experiment4Threading,
)

__version__ = "1.0.0"

__all__ = [
    # Configuration
    "CONFIG",
    "get_quick_config",
    # Suite
    "BenchmarkSuite",
    # Experiments
    "Experiment1Baseline",
    "Experiment2Scaling",
    "Experiment3Concurrency",
    "Experiment4Threading",
    # Data classes
    "ExperimentResult",
]
