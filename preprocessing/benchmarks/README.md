# LLM Preprocessing Benchmarks: Quantifying the "Python Tax"

**A production-grade analysis suite to characterize CPU bottlenecks in High-Throughput LLM Inference.**

> **Blog Series:** This benchmark suite accompanies the research blog series "The Preprocessing Barrier" published on [Deep Systems](https://deepsystems.substack.com).
> - [Part 1: The Preprocessing Barrier in LLMs](https://deepsystems.substack.com/p/the-preprocessing-barrier-in-llms)
> - [Part 2: How CPU Preprocessing Can Starve Your GPUs](https://deepsystems.substack.com/p/the-preprocessing-barrier-part-2)
>
> **For the full research narrative, see [BLOG.md](./BLOG.md)** — a comprehensive deep dive into the findings.

## Abstract

In modern LLM inference architectures, the GPU is often assumed to be the bottleneck. However, high-throughput serving engines must perform significant CPU-bound preprocessing—specifically **Jinja2 templating** and **BPE tokenization**—before a batch can be submitted to the GPU.

This repository contains a rigorous benchmarking suite designed to quantify this "Preprocessing Tax." It isolates the impact of the Python Global Interpreter Lock (GIL) on chat templating and demonstrates why standard threading models fail to scale in production RAG/Agentic workloads.

**Part 2** extends this analysis to production inference engines (vLLM, SGLang), demonstrating how coupled preprocessing doesn't just consume CPU cycles—it **starves the GPU**, creating a "Utilization Illusion" where `nvidia-smi` reports high utilization while the GPU idles between requests.

## Key Findings

### Part 1: The Preprocessing Barrier (Experiments 1-4)

Running these benchmarks on a standard 12-core CPU (Apple M3 Pro) reveals:

1. **The "Chat History Tax":** Jinja2 templating cost scales linearly with the **number of messages**, not token count. A 100-turn conversation incurs **~5x higher** templating latency than a single-turn request of the same length.
2. **The GIL Bottleneck:** Rust-based tokenizers (HuggingFace) release the GIL and scale linearly with threads. Python-based Jinja templating **does not scale**, capping throughput even on high-core-count servers.
3. **Concurrency Saturation:** At 64 concurrent requests, Jinja templating accounts for **>16% of total GPU Wait Time**, effectively throttling H100 utilization.

### Part 2: GPU Starvation Under Load (Experiments 5-7)

Stress tests against production inference engines reveal:

4. **The Utilization Illusion:** Despite 95% GPU utilization reported by `nvidia-smi`, throughput remains flat at ~133 req/s while latency degrades by **18x** under load. The GPU processes small batches and waits for the CPU.
5. **The Radix Paradox:** SGLang's Radix Attention provides **60% latency reduction** at low concurrency, but the benefit **vanishes at high load**. Even with 100% cache hits, requests spend **1.78 seconds** in the Python tokenization queue.
6. **Decoupled Preprocessing Wins:** Moving preprocessing to a sidecar increases throughput by **70%** and reduces latency by **~1 second**. The remaining ~1000ms floor is the "Uvicorn Limit"—pure Python HTTP overhead.

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
2. Run all 7 experiments.
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
# Part 1: The Preprocessing Barrier (CPU-focused)
# Experiment 1: The Anatomy of a Request
uv run python -m preprocessing.benchmarks.experiments.exp1_baseline --quick

# Experiment 2: The "Chat History" Tax
uv run python -m preprocessing.benchmarks.experiments.exp2_scaling --quick

# Experiment 3: The Concurrency Test
uv run python -m preprocessing.benchmarks.experiments.exp3_concurrency --quick

# Experiment 4: Threading vs The GIL
uv run python -m preprocessing.benchmarks.experiments.exp4_threading --quick

# Part 2: GPU Starvation (Requires GPU + inference server)
# Experiment 5: vLLM Latency Wall
uv run python -m preprocessing.benchmarks.experiments.exp5_vllm_latency_wall

# Experiment 6: SGLang Radix Cache Paradox (requires SGLang server on port 30000)
uv run python -m preprocessing.benchmarks.experiments.exp6_sglang_radix_latency

# Experiment 7: Sidecar Pre-tokenization (requires SGLang server on port 30000)
uv run python -m preprocessing.benchmarks.experiments.exp7_sidecar_latency
```

---

## Experimental Methodology

### Part 1: The Preprocessing Barrier (CPU-focused)

#### Experiment 1: The Anatomy of a Request

**Goal:** Establish a baseline latency for a single, large request (~1k tokens).

* **Metrics:** Wall-clock time for Template vs. Tokenize vs. Collate.
* **Result:** For single requests, overhead is negligible (~1.2ms). This often lulls engineers into a false sense of security.

#### Experiment 2: The "Chat History" Tax

**Goal:** Isolate the cost of *structure* vs. *content*.

* **Method:** We keep total tokens constant (~100k) but vary the conversation structure from 1 turn to 100 turns.
* **Finding:** Templating latency grows linearly with turns. Complex Agent/RAG prompts pay a higher tax per token.

#### Experiment 3: Production Concurrency

**Goal:** Simulate a vLLM-style serving queue.

* **Method:** Process 100k requests in batches of 64 using a `ThreadPoolExecutor`.
* **Metric:** **"GPU Wait Time"** — the time the GPU sits idle while a batch is prepared.
* **Finding:** As thread count increases, Tokenization (Rust) speeds up (3.5x). Templating (Python) stalls.

#### Experiment 4: Threading vs The GIL

**Goal:** Prove the GIL is the culprit.

* **Method:** Shows Jinja speedup factor flatlining at ~0.8x-1.0x regardless of threads, while Rust tokenizers achieve near-linear scaling.

---

### Part 2: GPU Starvation Under Load (Requires GPU)

These experiments require a GPU and demonstrate how CPU preprocessing bottlenecks affect real inference engines.

#### Experiment 5: vLLM Baseline – The Latency Wall

**Goal:** Characterize the "Throughput vs. Latency" trade-off in vLLM's architecture.

* **Method:** Run 1000 requests with 100-turn chat histories at two concurrency levels: 20 (Interactive) and 400 (Saturation).
* **Finding:** Despite 95% GPU utilization, throughput remains flat (~133 req/s) while latency degrades by **18x**. The system is scheduler-bound, not GPU-bound.

| Scenario    | Concurrency | Throughput | GPU Util | P50 Latency (ms) |
|-------------|-------------|------------|----------|------------------|
| Interactive | 20          | 133.75     | 92.61%   | 150.17           |
| Saturation  | 400         | 133.99     | 95.42%   | 2745.41          |

#### Experiment 6: SGLang Radix Cache – The Paradox

**Goal:** Test whether SGLang's Radix Attention (GPU-side KV cache) can eliminate the latency wall.

* **Method:** Compare unique requests (cache miss) vs. shared prefix requests (cache hit) at low and high concurrency.
* **Finding:** At low load, Radix Cache drops latency by **60%**. At high load, the benefit vanishes—requests spend 1.78 seconds in the Python tokenization queue before the cache can help.

| Load | Concurrency | Type      | Throughput | P50 Latency (ms) |
|------|-------------|-----------|------------|------------------|
| Low  | 20          | Cache Miss| 115.10     | 164.19           |
| Low  | 20          | Cache Hit | 245.84     | 67.21            |
| High | 400         | Cache Miss| 219.81     | 1834.97          |
| High | 400         | Cache Hit | 227.52     | 1780.42          |

**Conclusion:** GPU-side optimizations cannot fix CPU-side bottlenecks.

#### Experiment 7: Sidecar Pre-tokenization – Breaking the Barrier

**Goal:** Evaluate the impact of decoupled preprocessing (Sidecar pattern) on latency and throughput.

* **Method:** Move Jinja2 templating and tokenization out of the inference server. Compare "Monolith" (server does preprocessing) vs. "Sidecar" (pre-tokenized input).
* **Finding:** Sidecar increases throughput by **70%** (178 → 302 req/s) and reduces P50 latency by **~1 second**.

| Scenario  | Concurrency | Throughput | P50 Latency (ms) | Speedup |
|-----------|-------------|------------|------------------|---------|
| Monolith  | 400         | 178.58     | 1997.18          | -       |
| Sidecar   | 400         | 302.60     | 1016.68          | 1.96x   |

**The "Uvicorn Floor":** The remaining ~1000ms latency is the limit of Python's HTTP handling (JSON deserialization, TCP connection management). Further optimization requires moving networking to Rust/C++.

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
| `output/exp5_vllm_results.png` | vLLM latency wall visualization |
| `output/exp6_sglang_radix_results.png` | SGLang Radix cache paradox |
| `output/exp7_sidecar_results.png` | Sidecar vs. Monolith comparison |
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
│   │
│   │ # Part 1: CPU Preprocessing Analysis
│   ├── exp1_baseline.py           # Single-request latency breakdown
│   ├── exp2_scaling.py            # Message count vs. Latency analysis
│   ├── exp3_concurrency.py        # High-throughput production simulation
│   ├── exp4_threading.py          # GIL contention profiling
│   │
│   │ # Part 2: GPU Starvation Analysis (requires GPU)
│   ├── exp5_vllm_latency_wall.py  # vLLM baseline - proves latency wall
│   ├── exp6_sglang_radix_latency.py # SGLang Radix - proves CPU bottleneck
│   └── exp7_sidecar_latency.py    # Sidecar pattern - proves decoupling works
│
└── visualization/           # Plotting functions
    └── plots.py             # Publication-quality visualizations
```

---

## Requirements

### Part 1 (Experiments 1-4): CPU Analysis

* **Python:** 3.10+
* **Memory:** ~4GB RAM (Tests use memory-efficient chunking)
* **CPU:** Multi-core recommended (Tested on Apple Silicon M3 Pro, 12-core)

### Part 2 (Experiments 5-7): GPU Starvation Analysis

* **GPU:** NVIDIA GPU with CUDA support (Tested on RTX 4090, H100)
* **vLLM:** For Experiment 5 (`pip install vllm`)
* **SGLang:** For Experiments 6-7 (`pip install sglang[all]`)
* **Additional deps:** `aiohttp`, `pynvml`, `minijinja`

**Starting SGLang server for exp6/exp7:**
```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-0.5B-Instruct \
    --port 30000 \
    --attention-backend triton
```

---

## For Blog Posts

Each experiment file in `experiments/` is self-contained and well-documented:

1. **Clear docstrings** explaining the experiment's purpose and methodology
2. **Standalone execution** with its own `main()` function
3. **CLI arguments** for customization (`--quick`, `--tokens`, etc.)
4. **Detailed output** with key insights highlighted

Simply copy the individual experiment file along with the `common/` module for a complete, runnable example.

---

## Conclusions: Preprocessing is Infrastructure

Our benchmarks demonstrate that monolithic LLM serving reaches limits before full GPU utilization. You cannot build high-throughput RAG systems by coupling serial Python text processing with parallel GPU inference.

**Key Takeaways:**

1. **Utilization is Misleading:** High GPU utilization metrics can mask starvation if the scheduler cannot feed the GPU fast enough. Always measure throughput and latency together.

2. **The Barrier is Structural:** At high concurrency, the cost of tokenization compounds via queuing theory (Kingman's Formula), turning milliseconds of compute into seconds of wait time.

3. **Decoupling is Mandatory:** To maximize the ROI of your H100s, preprocessing must be treated as a separate infrastructure layer that feeds clean tensors to your inference engine.

4. **The Uvicorn Floor:** Even with perfect preprocessing, Python HTTP servers hit a ~1000ms latency floor at 400 concurrent connections. Breaking below this requires moving networking to Rust or C++.

---

## Blog Series

This benchmark suite accompanies the research blog series **"The Preprocessing Barrier"**:

| Part | Title | Focus |
|------|-------|-------|
| [Part 1](https://deepsystems.substack.com/p/the-preprocessing-barrier-in-llms) | The Preprocessing Barrier in LLMs | CPU cost of Jinja2 + Tokenization |
| [Part 2](https://deepsystems.substack.com/p/the-preprocessing-barrier-part-2) | How CPU Preprocessing Can Starve Your GPUs | GPU starvation, Radix paradox, Sidecar pattern |

**Want the full story?** Read [BLOG.md](./BLOG.md) for a comprehensive narrative covering all experiments, findings, and architectural recommendations.

---

## License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.

---

*Built for the AI Systems Engineering community. Follow [Deep Systems](https://deepsystems.substack.com) for more research on LLM infrastructure.*
