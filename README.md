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
| [Training Profiling](training/) | Communication profiling for distributed training (FSDP/DDP, NVLink vs PCIe) | [PROFILING_GUIDE](training/PROFILING_GUIDE.md) |

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

### Preprocessing Benchmarks

```bash
# Run preprocessing benchmarks (quick mode, ~5 min)
uv run python -m preprocessing.benchmarks.run --quick

# Run full preprocessing suite (~30 min)
uv run python -m preprocessing.benchmarks.run

# Run specific experiments
uv run python -m preprocessing.benchmarks.run --exp 1 3
```

### Communication Profiling (Multi-GPU)

```bash
# Run both NVLink and PCIe experiments
bash training/distributed/run_profiling_experiments.sh fsdp

# Analyze results
python training/analysis/profile_trace_analyzer.py

# View traces in TensorBoard
tensorboard --logdir=./output/profiling_traces
```

See [PROFILING_GUIDE.md](training/PROFILING_GUIDE.md) for detailed instructions.

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
├── README.md                    # This file
├── pyproject.toml               # Project configuration and dependencies
├── main.py                      # Main entry point
│
├── preprocessing/               # Preprocessing benchmarks
│   └── benchmarks/
│       ├── common/              # Shared utilities
│       ├── experiments/         # Individual experiments
│       ├── visualization/       # Plotting functions
│       ├── run.py               # CLI runner
│       └── suite.py             # Benchmark orchestrator
│
├── training/                    # Distributed training profiling
│   ├── README.md                # Training suite documentation
│   ├── distributed/             # Main distributed training scripts
│   │   ├── profile_distributed.py       # FSDP/DDP profiling script
│   │   ├── train_distributed_node.py    # Multi-node training runner
│   │   ├── train_single_node_baseline.py # Single-node baseline
│   │   ├── ddp_fsdp_oom_demo.py         # OOM demo with checkpointing
│   │   └── run_profiling_experiments.sh # Automated experiment runner
│   ├── analysis/                # Trace analysis tools
│   │   ├── profile_trace_analyzer.py    # PyTorch profiler analyzer
│   │   ├── dataloader_bottleneck_simulation.py
│   │   ├── padding_tax_simulation.py
│   │   ├── random_vs_seq_dataloader_simulation.py
│   │   └── ...                  # Other analysis scripts
│   └── helper/                  # Utility modules
│       ├── profiler_analyzer.py # FSDP overlap metrics
│       ├── check_gpu_info.py    # GPU information
│       └── fsdp_ddp_comparison_demo.py
│
└── output/                      # All generated outputs (gitignored)
    ├── README.md                # Output directory structure
    ├── profiling_traces/        # PyTorch profiler traces
    └── communication_analysis.md # Generated comparison tables
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
