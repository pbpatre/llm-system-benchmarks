# The Preprocessing Barrier: How CPU Bottlenecks Starve Your GPUs

*A deep dive into the hidden cost of Python-based preprocessing in high-throughput LLM inference*

> **This document accompanies the benchmark experiments in this repository.** Each finding can be reproduced by running the corresponding experiment. See the [README](./README.md) for setup instructions.

---

## Table of Contents

1. [Introduction: The Hidden Bottleneck](#introduction-the-hidden-bottleneck)
2. [Part 1: The Preprocessing Tax](#part-1-the-preprocessing-tax)
   - [The Preprocessing Pipeline](#the-preprocessing-pipeline)
   - [Experiment 1: The Baseline Illusion](#experiment-1-the-baseline-illusion)
   - [Experiment 2: The Chat History Tax](#experiment-2-the-chat-history-tax)
   - [Experiment 3: Production Concurrency](#experiment-3-production-concurrency)
   - [Experiment 4: Threading vs The GIL](#experiment-4-threading-vs-the-gil)
3. [Part 2: GPU Starvation Under Load](#part-2-gpu-starvation-under-load)
   - [The Architecture Problem](#the-architecture-problem)
   - [Experiment 5: The Utilization Illusion](#experiment-5-the-utilization-illusion)
   - [Experiment 6: The Radix Paradox](#experiment-6-the-radix-paradox)
   - [Experiment 7: Breaking the Barrier](#experiment-7-breaking-the-barrier)
4. [Conclusions](#conclusions)
5. [Reproduce the Results](#reproduce-the-results)

---

## Introduction: The Hidden Bottleneck

In the modern AI stack, the GPU is the protagonist. We obsess over H100 utilization, optimize CUDA kernels, and celebrate every percentage point of Model Flops Utilization (MFU). But there's a silent antagonist lurking in the wings: **CPU-bound preprocessing**.

Before a single FLOP is computed on the GPU, every LLM request must pass through a gauntlet of CPU operations:

1. **Chat Templating** — Converting structured messages into model-specific prompt strings (Jinja2)
2. **Tokenization** — Encoding strings into integer token IDs (BPE)
3. **Collation** — Batching and padding tensors for GPU transfer

This "Preprocessing Tax" is often dismissed as negligible. After all, what's a few milliseconds compared to GPU inference time?

**The answer: everything.**

Our benchmarks reveal that at production scale, this CPU bottleneck doesn't just add latency—it **starves the GPU**, creating a "Utilization Illusion" where `nvidia-smi` reports 95% utilization while your expensive hardware sits idle between requests.

---

## Part 1: The Preprocessing Tax

### The Preprocessing Pipeline

Every LLM request follows the same journey before reaching the GPU:

```
User Request → Jinja2 Template → Tokenizer → Tensor Collation → GPU
```

**Step 1: Chat Templating (Jinja2)**

Modern chat models require specific prompt formats. A simple request like:

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"}
  ]
}
```

Must be transformed into a model-specific string:

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful assistant.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
What is 2+2?<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
```

This transformation uses **Jinja2**, a Python templating engine. Jinja2 is elegant and flexible, but it's also **single-threaded** and bound by Python's Global Interpreter Lock (GIL).

**Step 2: Tokenization (HuggingFace/tiktoken)**

The prompt string is encoded into integer token IDs:

```python
[128000, 128006, 9125, 128007, 271, 2675, 527, 264, 11190, 18328, ...]
```

Modern tokenizers like HuggingFace Tokenizers are implemented in **Rust** and release the GIL during computation. This is critical—it means tokenization can scale with threads.

**Step 3: Collation (PyTorch)**

Token IDs are converted to tensors, padded to matching lengths, and prepared for GPU transfer. This is relatively fast and not the bottleneck.

---

### Experiment 1: The Baseline Illusion

> **Run it yourself:** `uv run python -m preprocessing.benchmarks.experiments.exp1_baseline --quick`

**Goal:** Establish a baseline for single-request latency.

For a single 1,000-token request, the breakdown looks harmless:

| Stage | Latency |
|-------|---------|
| Jinja2 Template | ~0.5ms |
| Tokenization | ~0.7ms |
| Collation | ~0.03ms |
| **Total** | **~1.2ms** |

**The Illusion:** Engineers see this and conclude preprocessing is negligible. After all, GPU inference takes 50-200ms. What's 1.2ms?

This baseline lulls teams into a false sense of security. The problems emerge at scale.

---

### Experiment 2: The Chat History Tax

> **Run it yourself:** `uv run python -m preprocessing.benchmarks.experiments.exp2_scaling --quick`

**Goal:** Isolate the cost of *structure* vs. *content*.

We keep total token count constant (~100k tokens) but vary the conversation structure:

| Configuration | Tokens | Jinja2 Time | Observation |
|---------------|--------|-------------|-------------|
| 1 Turn (monologue) | ~100k | 11.8ms | Baseline |
| 10 Turns | ~100k | 21.4ms | 1.8x slower |
| 50 Turns | ~100k | 41.2ms | 3.5x slower |
| **100 Turns** | ~100k | **58.7ms** | **~5x slower** |

**The Finding:** Jinja2 templating cost scales with **message count**, not token count.

Why? Because Jinja2 must:
- Parse the template for each message
- Perform string concatenation operations
- Handle control flow (loops, conditionals)

For complex RAG/Agent workloads with long conversation histories, this "Chat History Tax" compounds into significant overhead.

**Implication:** A 100-turn conversation incurs 5x higher templating cost than a single-turn request of equivalent length. Your RAG pipelines pay a hidden tax proportional to conversation complexity.

---

### Experiment 3: Production Concurrency

> **Run it yourself:** `uv run python -m preprocessing.benchmarks.experiments.exp3_concurrency --quick`

**Goal:** Simulate a vLLM-style serving queue under load.

We process batches of 64 requests using a ThreadPoolExecutor, measuring "GPU Wait Time"—the time the GPU sits idle while a batch is prepared.

| Threads | Jinja2 Time | Tokenize Time | Collate Time |
|---------|-------------|---------------|--------------|
| 1 | 542ms | 286ms | 22.1ms |
| 4 | 507ms | 98ms | 22.3ms |
| 16 | 481ms | 51ms | 22.2ms |
| 64 | **134ms** | **83ms** | 22.3ms |

**The Finding:** As threads increase:
- **Tokenization speeds up 3.5x** (286ms → 83ms) — Rust releases the GIL
- **Jinja2 barely improves** (542ms → 134ms) — Python GIL blocks parallelism

At 64 concurrent requests, Jinja2 templating accounts for **>16% of total GPU wait time**. The GPU is literally waiting for Python to finish string processing.

---

### Experiment 4: Threading vs The GIL

> **Run it yourself:** `uv run python -m preprocessing.benchmarks.experiments.exp4_threading --quick`

**Goal:** Prove the GIL is the culprit.

We measure the "speedup factor"—how much faster each operation gets as we add threads.

| Threads | Jinja2 Speedup | Tokenizer Speedup |
|---------|----------------|-------------------|
| 1 | 1.0x | 1.0x |
| 2 | 0.9x | 1.8x |
| 4 | 0.85x | 3.2x |
| 8 | 0.82x | 5.1x |
| 16 | 0.80x | 6.8x |

**The Finding:**
- **Tokenizers** achieve near-linear scaling (Rust, GIL-free)
- **Jinja2** flatlines at ~0.8x regardless of threads (Python, GIL-bound)

The Jinja2 speedup factor is actually **less than 1.0x** because threading adds overhead without enabling true parallelism. This is the GIL in action.

---

## Part 2: GPU Starvation Under Load

Part 1 established that preprocessing is expensive. Part 2 asks: **What happens to the GPU when the CPU can't keep up?**

The answer is counterintuitive and alarming.

### The Architecture Problem

Modern inference engines like **vLLM** and **SGLang** are designed for throughput. They use continuous batching and PagedAttention to maximize GPU utilization. But there's a critical dependency:

```
Request → [Python Event Loop] → Tokenize → Schedule → GPU
                    ↑
            Single-threaded bottleneck
```

Before the scheduler can allocate GPU memory, it needs the **exact token count**. This forces tokenization onto the critical path. Under load, the Python event loop becomes a queue, and requests back up waiting for preprocessing.

### vLLM: The Centralized Scheduler

vLLM orchestrates requests through a centralized Python scheduler:

1. FastAPI/Uvicorn accepts the HTTP request
2. The scheduler tokenizes the prompt (blocking)
3. PagedAttention allocates memory based on token count
4. GPU processes the batch

At low concurrency, this works fine. At high concurrency, the scheduler becomes the bottleneck.

### SGLang: Radix Attention

SGLang introduces **Radix Attention**, an optimization for multi-turn chat:

- The KV cache is modeled as a Radix Tree
- Shared prefixes (system prompts, conversation history) are cached
- Subsequent requests skip attention computation for cached portions

**The Theory:** Since attention is O(N²), caching the prefix should yield massive speedups.

**The Reality:** To traverse the Radix Tree, the system needs the **token sequence**. This means preprocessing still happens before the cache lookup. The GPU optimization is blocked by CPU preprocessing.

---

### Experiment 5: The Utilization Illusion

> **Run it yourself:** `uv run python -m preprocessing.benchmarks.experiments.exp5_vllm_latency_wall`
> 
> **Requires:** GPU + vLLM installed

**Goal:** Characterize the throughput vs. latency trade-off in vLLM.

We send 1,000 requests with 100-turn chat histories at two concurrency levels:

| Scenario | Concurrency | Throughput | GPU Util | P50 Latency |
|----------|-------------|------------|----------|-------------|
| Interactive | 20 | 133.75 req/s | 92.61% | 150ms |
| Saturation | 400 | 133.99 req/s | 95.42% | **2,745ms** |

**The Anomaly:** Despite a 20x increase in load:
- Throughput remained **flat** (133 req/s)
- Latency degraded by **18x** (150ms → 2,745ms)
- GPU utilization was **high** (95%)

**The Diagnosis:** The system was not GPU-bound—it was **scheduler-bound**. The Python process was so overwhelmed by Jinja2 and tokenization that it could only feed the GPU at 133 req/s.

The GPU spent its time:
1. Processing a small batch quickly
2. **Waiting** for the next batch to be prepared
3. Repeat

This creates the "Utilization Illusion": `nvidia-smi` shows high utilization because the GPU *is* being used, but the actual throughput is capped by CPU preprocessing. The GPU is idling between bursts, masked by the utilization metric.

---

### Experiment 6: The Radix Paradox

> **Run it yourself:** `uv run python -m preprocessing.benchmarks.experiments.exp6_sglang_radix_latency`
> 
> **Requires:** SGLang server running on port 30000

**Goal:** Test whether Radix Attention can eliminate the latency wall.

We compare:
- **Unique requests** (cache miss) — Every request has unique history
- **Shared requests** (cache hit) — All requests share the same prefix

| Load | Concurrency | Cache | Throughput | P50 Latency |
|------|-------------|-------|------------|-------------|
| Low | 20 | Miss | 115 req/s | 164ms |
| Low | 20 | **Hit** | 246 req/s | **67ms** |
| High | 400 | Miss | 220 req/s | 1,835ms |
| High | 400 | **Hit** | 228 req/s | **1,780ms** |

**At Low Concurrency (The Happy Path):**
Radix Cache is highly effective. Cache hits drop latency by **60%** (164ms → 67ms) and double throughput. The CPU isn't saturated, so GPU savings shine through.

**At High Concurrency (The Paradox):**
The benefit **vanishes**. Despite 100% cache hits, latency is still 1,780ms—nearly identical to cache misses (1,835ms).

**Why?** Even with a 100% cache hit, the request spent **1.78 seconds in the Python queue** waiting to be tokenized. The SGLang server cannot check the Radix Tree until it has the token sequence. The upstream CPU bottleneck completely masks the downstream GPU optimization.

**Conclusion:** GPU-side optimizations cannot fix CPU-side bottlenecks. A 0ms GPU computation doesn't help if the CPU takes 2 seconds to prepare the data.

---

### Experiment 7: Breaking the Barrier

> **Run it yourself:** `uv run python -m preprocessing.benchmarks.experiments.exp7_sidecar_latency`
> 
> **Requires:** SGLang server running on port 30000

**Goal:** Test the impact of decoupled preprocessing (Sidecar pattern).

We move Jinja2 templating and tokenization out of the inference server into a separate process. The server receives pre-computed tensors, skipping CPU-heavy text processing.

**Phase 1: Pure CPU Savings (Moderate Load)**

At 50 concurrent users with ~2,500 token prompts:

| Mode | Throughput | P50 Latency |
|------|------------|-------------|
| Monolith | 118 req/s | 379ms |
| Sidecar | 179 req/s | **212ms** |

**Finding:** Even without massive queuing, offloading preprocessing saves **~167ms per request**. This is the "Base Tax" that every monolithic request pays.

**Phase 2: The Multiplier Effect (High Load)**

At 400 concurrent users:

| Mode | Throughput | P50 Latency | Speedup |
|------|------------|-------------|---------|
| Monolith | 179 req/s | 1,997ms | - |
| Sidecar | **303 req/s** | **1,017ms** | **1.96x** |

**Key Findings:**

1. **Throughput increased 70%** (179 → 303 req/s)
   
   By removing the preprocessing bottleneck, the GPU was finally saturated. The monolith was leaving **40% of GPU capacity** on the table.

2. **Latency dropped by 1 second** (1,997ms → 1,017ms)
   
   This is **Kingman's Formula** in action. By shaving off the 167ms "Base Tax," we prevented the queue from exploding at high load. Small reductions in service time yield **exponential reductions** in wait time.

3. **The "Uvicorn Floor"** (~1,000ms)
   
   The remaining latency represents the limit of Python's HTTP handling. Even without tokenization, the server must:
   - Accept 400 TCP connections
   - Parse 400 HTTP headers
   - Deserialize 400 massive JSON bodies
   - Run Pydantic validation on 400 objects
   
   Breaking below 1 second requires moving networking to Rust or C++.

---

## Conclusions

Our benchmarks demonstrate a fundamental architectural constraint: **you cannot build high-throughput RAG systems by coupling serial Python text processing with parallel GPU inference**.

### Key Takeaways

1. **Utilization is Misleading**
   
   High GPU utilization metrics can mask starvation if the scheduler cannot feed the GPU fast enough. Always measure throughput and latency together. A GPU at 95% utilization processing 133 req/s is leaving money on the table.

2. **The Barrier is Structural**
   
   The cost of preprocessing compounds via queuing theory. Kingman's Formula tells us that as utilization approaches 100%, wait time approaches infinity. Milliseconds of compute become seconds of wait time.

3. **GPU Optimizations Can't Fix CPU Bottlenecks**
   
   SGLang's Radix Attention is brilliant engineering—it can eliminate GPU computation entirely for cached prefixes. But it cannot help requests stuck in the tokenization queue. The optimization is gated by preprocessing.

4. **Decoupling is Mandatory**
   
   To maximize ROI on your H100s, preprocessing must be treated as a **separate infrastructure layer** that feeds clean tensors to your inference engine. The Sidecar pattern unlocks 70% more throughput.

5. **The Python Ceiling Exists**
   
   Even with perfect preprocessing, Python HTTP servers hit a ~1,000ms latency floor at 400 concurrent connections. Breaking below this requires moving the networking layer to Rust or C++.

### The Path Forward

For production RAG/Agent workloads:

- **Short term:** Move templating and tokenization to a Sidecar process
- **Medium term:** Implement Rust-based preprocessing (minijinja + tokenizers)
- **Long term:** Full Rust/C++ gateway that feeds tensors directly to the inference engine

The preprocessing barrier is real, measurable, and solvable. The experiments in this repository provide the evidence; the architecture changes provide the solution.

---

## Reproduce the Results

All findings in this document can be reproduced using the benchmark suite:

```bash
# Install dependencies
git clone https://github.com/yourusername/llm-system-benchmarks.git
cd llm-system-benchmarks
uv sync

# Part 1: CPU Preprocessing Analysis
uv run python -m preprocessing.benchmarks.experiments.exp1_baseline --quick
uv run python -m preprocessing.benchmarks.experiments.exp2_scaling --quick
uv run python -m preprocessing.benchmarks.experiments.exp3_concurrency --quick
uv run python -m preprocessing.benchmarks.experiments.exp4_threading --quick

# Part 2: GPU Starvation (requires GPU + inference server)
# Start SGLang server first:
python -m sglang.launch_server --model-path Qwen/Qwen2.5-0.5B-Instruct --port 30000

# Then run experiments:
uv run python -m preprocessing.benchmarks.experiments.exp5_vllm_latency_wall
uv run python -m preprocessing.benchmarks.experiments.exp6_sglang_radix_latency
uv run python -m preprocessing.benchmarks.experiments.exp7_sidecar_latency
```

---

## References

- [Part 1: The Preprocessing Barrier in LLMs](https://deepsystems.substack.com/p/the-preprocessing-barrier-in-llms) — Deep Systems
- [Part 2: How CPU Preprocessing Can Starve Your GPUs](https://deepsystems.substack.com/p/the-preprocessing-barrier-part-2) — Deep Systems
- [vLLM: Easy, Fast, and Cheap LLM Serving](https://github.com/vllm-project/vllm)
- [SGLang: Fast Serving Framework for Large Language Models](https://github.com/sgl-project/sglang)
- [Kingman's Formula (Queuing Theory)](https://en.wikipedia.org/wiki/Kingman%27s_formula)

---

*Built for the AI Systems Engineering community. Follow [Deep Systems](https://deepsystems.substack.com) for more research on LLM infrastructure.*
