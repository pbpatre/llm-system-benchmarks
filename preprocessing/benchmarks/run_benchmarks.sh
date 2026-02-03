#!/bin/bash
#
# LLM Pre-Processing Bottleneck Benchmark Suite Runner
#
# Usage:
#   ./run_benchmarks.sh                    # Run full suite
#   ./run_benchmarks.sh --quick            # Quick mode (~5 min)
#   ./run_benchmarks.sh --exp 1            # Run experiment 1 only
#   ./run_benchmarks.sh --exp 1 3 4        # Run experiments 1, 3, and 4
#   ./run_benchmarks.sh --list             # List available experiments
#
# Individual experiments (for blog posts):
#   ./run_benchmarks.sh exp1               # Run baseline experiment
#   ./run_benchmarks.sh exp2               # Run scaling experiment
#   ./run_benchmarks.sh exp3               # Run concurrency experiment
#   ./run_benchmarks.sh exp4               # Run threading experiment
#

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Add project root to PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Check for shortcut experiment names
case "${1:-}" in
    exp1)
        echo "Running Experiment 1: Baseline..."
        python -m preprocessing.benchmarks.experiments.exp1_baseline "${@:2}"
        ;;
    exp2)
        echo "Running Experiment 2: Scaling..."
        python -m preprocessing.benchmarks.experiments.exp2_scaling "${@:2}"
        ;;
    exp3)
        echo "Running Experiment 3: Concurrency..."
        python -m preprocessing.benchmarks.experiments.exp3_concurrency "${@:2}"
        ;;
    exp4)
        echo "Running Experiment 4: Threading..."
        python -m preprocessing.benchmarks.experiments.exp4_threading "${@:2}"
        ;;
    *)
        # Default: run the main suite with all arguments
        python -m preprocessing.benchmarks.run "$@"
        ;;
esac
