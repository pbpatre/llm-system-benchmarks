"""
LLM System Benchmarks - Main entry point.

Run individual benchmark suites or execute all benchmarks.
"""


def main():
    """Main entry point for the benchmarking suite."""
    print("LLM System Benchmarks")
    print("=" * 50)
    print("\nAvailable benchmark suites:")
    print("  - preprocessing: Analyze Jinja2 templating vs BPE tokenization bottlenecks")
    print("\nRun a specific suite:")
    print("  python -m preprocessing.benchmarks.run --quick")
    print("\nSee preprocessing/benchmarks/README.md for detailed usage.")


if __name__ == "__main__":
    main()
