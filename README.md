# LLM System Benchmarks

A benchmarking suite for analyzing performance characteristics of LLM systems and internals. This repository provides production-grade benchmarks to understand bottlenecks, optimization opportunities, and system behavior in large language model pipelines.

## Overview

LLM inference systems involve multiple stages—preprocessing, model execution, and postprocessing—each with unique performance characteristics. This repository provides rigorous benchmarks to:

- Identify bottlenecks in LLM pipelines
- Quantify the impact of Python's GIL on throughput
- Compare different parallelization strategies
- Guide architectural decisions for production systems

## Benchmark Suites

| Suite | Description | Documentation |
|-------|-------------|---------------|
| [Preprocessing](preprocessing/) | Analyzes preprocessing bottlenecks (Jinja2 templating vs BPE tokenization) | [README](preprocessing/benchmarks/README.md) |

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/yourusername/llm-system-benchmarks.git
cd llm-system-benchmarks

# Install dependencies
uv sync

# For development (includes pytest, ruff)
uv sync --dev
```

## Quick Start

```bash
# Run preprocessing benchmarks (quick mode, ~5 min)
uv run python -m preprocessing.benchmarks.run --quick

# Run full preprocessing suite (~30 min)
uv run python -m preprocessing.benchmarks.run

# Run specific experiments
uv run python -m preprocessing.benchmarks.run --exp 1 3
```

## Requirements

- Python >= 3.10
- CUDA-capable GPU (recommended for realistic benchmarks)
- HuggingFace account with access to gated models (e.g., Llama)

Set your HuggingFace token:
```bash
export HF_TOKEN="your_token_here"
```

## Project Structure

```
llm-system-benchmarks/
├── README.md                 # This file
├── pyproject.toml            # Project configuration and dependencies
├── main.py                   # Main entry point
│
├── preprocessing/            # Preprocessing benchmarks
│   └── benchmarks/
│       ├── common/           # Shared utilities
│       ├── experiments/      # Individual experiments
│       ├── visualization/    # Plotting functions
│       ├── run.py            # CLI runner
│       └── suite.py          # Benchmark orchestrator
│
└── results.csv               # Example benchmark results
```

## Adding New Benchmark Suites

To add a new benchmark suite (e.g., `inference/`, `quantization/`):

1. Create a new directory under the project root
2. Follow the structure of `preprocessing/benchmarks/`
3. Include a `README.md` documenting the suite
4. Add any new dependencies to `pyproject.toml`

## Contributing

Contributions are welcome! Please ensure:

1. Code follows the existing style (run `uv run ruff check`)
2. New experiments include docstrings and documentation
3. Benchmarks are reproducible with clear methodology

## License

See [LICENSE](LICENSE) for details.
