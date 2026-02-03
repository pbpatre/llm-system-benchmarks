# LLM Preprocessing Benchmarks: Quantifying the "Python Tax"

**A production-grade analysis suite to characterize CPU bottlenecks in High-Throughput LLM Inference.**

## Abstract

In modern LLM inference architectures, the GPU is often assumed to be the bottleneck. However, high-throughput serving engines must perform significant CPU-bound preprocessing—specifically **Jinja2 templating** and **BPE tokenization**—before a batch can be submitted to the GPU.

This repository contains a rigorous benchmarking suite designed to quantify this "Preprocessing Tax." It isolates the impact of the Python Global Interpreter Lock (GIL) on chat templating and demonstrates why standard threading models fail to scale in production RAG/Agentic workloads.

## Key Findings

Running these benchmarks on a standard 12-core CPU (Apple M3 Pro) reveals:

1. **The "Chat History Tax":** Jinja2 templating cost scales linearly with the **number of messages**, not token count. A 100-turn conversation incurs **~5x higher** templating latency than a single-turn request of the same length.
2. **The GIL Bottleneck:** Rust-based tokenizers (HuggingFace) release the GIL and scale linearly with threads. Python-based Jinja templating **does not scale**, capping throughput even on high-core-count servers.
3. **Concurrency Saturation:** At 64 concurrent requests, Jinja templating accounts for **>16% of total GPU Wait Time**, effectively throttling H100 utilization.

---

## Quick Start

### Installation

Clone the repo and install dependencies using [uv](https://github.com/astral-sh/uv):

```bash
git clone https://github.com/yourusername/llm-system-benchmarks.git
cd llm-system-benchmarks
uv sync
```

Set your HuggingFace token (required for gated models like Llama 3):

```bash
export HF_TOKEN="your_hf_token_here"
```

### Reproduce the Findings

Run the full suite in "Quick Mode" (approx. 5 minutes) to verify the pipeline:

```bash
# Using uv
uv run python -m preprocessing.benchmarks.run --quick

# Or using shell script
./preprocessing/benchmarks/run_benchmarks.sh --quick
```

This will:
1. Download the tokenizer (Llama-3.1-8B-Instruct).
2. Run all 4 experiments.
3. Generate CSVs and High-Res Plots in the `output/` directory.

### Run Specific Experiments

```bash
# Run experiments 1 and 3 only
uv run python -m preprocessing.benchmarks.run --exp 1 3

# Or use the shell script shortcuts
./preprocessing/benchmarks/run_benchmarks.sh exp1 --quick
./preprocessing/benchmarks/run_benchmarks.sh exp3
```

### Run Individual Experiments Standalone

Each experiment can run independently—perfect for sharing or blog posts:

```bash
# Experiment 1: The Anatomy of a Request
uv run python -m preprocessing.benchmarks.experiments.exp1_baseline --quick

# Experiment 2: The "Chat History" Tax
uv run python -m preprocessing.benchmarks.experiments.exp2_scaling --quick

# Experiment 3: The Concurrency Test
uv run python -m preprocessing.benchmarks.experiments.exp3_concurrency --quick

# Experiment 4: Threading vs The GIL
uv run python -m preprocessing.benchmarks.experiments.exp4_threading --quick
```

---

## Experimental Methodology

### Experiment 1: The Anatomy of a Request

**Goal:** Establish a baseline latency for a single, large request (~1k tokens).

* **Metrics:** Wall-clock time for Template vs. Tokenize vs. Collate.
* **Result:** For single requests, overhead is negligible (~1.2ms). This often lulls engineers into a false sense of security.

### Experiment 2: The "Chat History" Tax

**Goal:** Isolate the cost of *structure* vs. *content*.

* **Method:** We keep total tokens constant (~100k) but vary the conversation structure from 1 turn to 100 turns.
* **Finding:** Templating latency grows linearly with turns. Complex Agent/RAG prompts pay a higher tax per token.

### Experiment 3: Production Concurrency

**Goal:** Simulate a vLLM-style serving queue.

* **Method:** Process 100k requests in batches of 64 using a `ThreadPoolExecutor`.
* **Metric:** **"GPU Wait Time"** — the time the GPU sits idle while a batch is prepared.
* **Finding:** As thread count increases, Tokenization (Rust) speeds up (3.5x). Templating (Python) stalls.

### Experiment 4: Threading vs The GIL

**Goal:** Prove the GIL is the culprit.

* **Method:** Shows Jinja speedup factor flatlining at ~0.8x-1.0x regardless of threads, while Rust tokenizers achieve near-linear scaling.

---

## Programmatic Usage

```python
from preprocessing.benchmarks import BenchmarkSuite

# Run all experiments
suite = BenchmarkSuite()
suite.run_all()

# Or run specific experiments
suite = BenchmarkSuite()
suite.run_experiment(1)  # Baseline
suite.run_experiment(3)  # Concurrency
suite.generate_plots("./output")
suite.save_results("results.csv")

# Access results
df = suite.get_results_dataframe()
exp3_result = suite.get_experiment_result(3)
```

### Custom Configuration

Modify `common/config.py` or pass a custom config dict:

```python
from preprocessing.benchmarks import BenchmarkSuite, CONFIG

# Custom configuration
my_config = CONFIG.copy()
my_config["target_tokens"] = 5000
my_config["exp3_thread_counts"] = [1, 2, 4, 8]

suite = BenchmarkSuite(my_config)
suite.run_all()
```

---

## Visualizations

The suite automatically generates publication-ready plots using `matplotlib`/`seaborn` in the `output/` directory.

### Output Files

All outputs are written to `output/` (configurable via `--output`):

| File | Description |
|------|-------------|
| `output/exp1_baseline_breakdown.png` | Latency breakdown bar chart |
| `output/exp2_scaling_turns.png` | Scaling analysis dual plot |
| `output/exp3_gpu_wait_time.png` | GPU wait time vs threads — demonstrates the "Jinja Wall" |
| `output/exp3_jinja_overhead.png` | Jinja % of total time |
| `output/exp4_threading_speedup.png` | Visual proof of Rust vs. Python scaling |
| `output/results.csv` | Combined results from all experiments |

---

## Repository Structure

```text
preprocessing/benchmarks/
├── __init__.py              # Package exports and documentation
├── run.py                   # Main CLI entry point
├── suite.py                 # BenchmarkSuite orchestrator
├── run_benchmarks.sh        # Shell convenience script
├── README.md                # This file
│
├── common/                  # Shared utilities
│   ├── config.py            # Configuration parameters
│   ├── data_classes.py      # Result containers
│   ├── data_generators.py   # Synthetic data generation
│   ├── system_monitor.py    # Non-blocking CPU/RAM profiling
│   ├── workers.py           # Thread/Process worker logic
│   └── utils.py             # Batch collation utilities
│
├── experiments/             # The core scientific experiments
│   ├── exp1_baseline.py     # Single-request latency breakdown
│   ├── exp2_scaling.py      # Message count vs. Latency analysis
│   ├── exp3_concurrency.py  # High-throughput production simulation
│   └── exp4_threading.py    # GIL contention profiling
│
└── visualization/           # Plotting functions
    └── plots.py             # Publication-quality visualizations
```

---

## Requirements

* **Python:** 3.10+
* **Memory:** ~4GB RAM (Tests use memory-efficient chunking)
* **CPU:** Multi-core recommended (Tested on Apple Silicon M3 Pro, 12-core)

---

## For Blog Posts

Each experiment file in `experiments/` is self-contained and well-documented:

1. **Clear docstrings** explaining the experiment's purpose and methodology
2. **Standalone execution** with its own `main()` function
3. **CLI arguments** for customization (`--quick`, `--tokens`, etc.)
4. **Detailed output** with key insights highlighted

Simply copy the individual experiment file along with the `common/` module for a complete, runnable example.

---

## License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.

---

*Built for the AI Systems Engineering community.*
