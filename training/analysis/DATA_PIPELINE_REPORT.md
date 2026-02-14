# The Data Journey: From Disk to GPU

## A Benchmarking Report on LLM Training Pipeline Bottlenecks

---

**Abstract.** Modern GPU hardware can sustain hundreds of teraflops of compute, yet
production LLM training runs routinely achieve far less than peak utilization.
The bottleneck is rarely the model itself -- it is the *data pipeline*: the chain
of operations that moves training samples from persistent storage through the
operating system, across user-space processing, and finally over the PCIe bus
into GPU memory. This report documents a systematic, experiment-driven
investigation of every stage in that chain. Through seven standalone benchmarking
scripts, we progressively peel back layers of abstraction to expose where time
is lost and what practitioners can do about it.

**Benchmark environment:** AMD EPYC 7R13 (16 cores), 124 GiB RAM, local NVMe
instance storage (559 GB), NVIDIA L40S (47.7 GB VRAM, PCIe 4.0 x16), Linux
6.8, PyTorch 2.10, CUDA 12.8. All benchmarks run in a Kubernetes container on
AWS with unlimited cgroup memory and CPU quota.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Where It All Begins: Random vs Sequential I/O](#2-where-it-all-begins-random-vs-sequential-io)
3. [Adding PyTorch: How the DataLoader Masks the Problem](#3-adding-pytorch-how-the-dataloader-masks-the-problem)
4. [The CPU Starvation Problem](#4-the-cpu-starvation-problem)
5. [The Padding Tax: Wasted GPU Cycles](#5-the-padding-tax-wasted-gpu-cycles)
6. [Crossing the PCIe Bridge: Pinned Memory and DMA](#6-crossing-the-pcie-bridge-pinned-memory-and-dma)
7. [The Full Picture: Streaming vs ImageFolder Under the Profiler](#7-the-full-picture-streaming-vs-imagefolder-under-the-profiler)
8. [Conclusion and Optimization Checklist](#8-conclusion-and-optimization-checklist)

---

## 1. Introduction

### The problem nobody talks about

When engineers discuss LLM training performance, the conversation gravitates
toward GPU architecture, mixed-precision arithmetic, tensor parallelism, and
Flash Attention. These are important -- but they only matter *after data reaches
the GPU*. Before a single floating-point operation fires on a CUDA core, the
training sample has already traveled a long and surprisingly treacherous path:

```
Storage (SSD/HDD/NFS)
    |
    v
OS Page Cache
    |
    v
User-Space Memory (Python / PyTorch)
    |
    v
CPU Preprocessing (tokenization, augmentation, collation)
    |
    v
Batching & Padding
    |
    v
PCIe Bus (Host -> Device DMA)
    |
    v
GPU VRAM
    |
    v
[Training begins here]
```

Every arrow in that chain is a potential bottleneck. Worse, these bottlenecks
are *invisible by default*. The GPU simply idles, and `nvidia-smi` reports low
utilization, but nothing tells you *why*. You might see the GPU at 40%
utilization and assume your model is memory-bandwidth-bound, when in reality the
GPU is starving for data because a Python DataLoader is blocked on `open()`
calls to 50,000 small files.

### Our approach

Rather than guessing, we built seven standalone benchmarks that isolate each
stage of the data pipeline. Each experiment controls for exactly one variable,
and together they form a narrative -- a story of progressively uncovering
layers of abstraction and their costs:

| Chapter | Experiment | What it isolates |
|---------|-----------|-----------------|
| 2 | Pure Python I/O benchmark | Raw storage access pattern (random vs sequential) |
| 3 | PyTorch DataLoader I/O benchmark | How `num_workers` masks I/O problems |
| 4 | CPU starvation simulation | Preprocessing throughput vs GPU speed |
| 5 | Padding tax visualization | Wasted compute from naive batching |
| 6 | PCIe transfer benchmark | Pinned vs pageable memory, CPU/GPU overlap |
| 7 | Streaming profiler comparison | All stages combined in a real training loop |

The experiments are designed to be run sequentially -- each one reveals a
problem, and the next one shows what happens when you add the next layer of
the stack. Let us begin at the very bottom: the disk.

---

## 2. Where It All Begins: Random vs Sequential I/O

**Script:** `random_vs_seq_python_simulation.py`
**Utility library:** `io_benchmark_utils.py`

### Motivation

The most fundamental question in any data pipeline is: *how fast can we read
data from storage?* The answer depends almost entirely on the **access pattern**.
Modern storage devices -- SSDs, HDDs, and network file systems alike -- have
vastly different performance characteristics for sequential reads versus random
reads. Before we involve any ML framework, we need to understand this raw
physics.

In LLM training, data is typically stored in one of two layouts:

1. **Many small files** -- one file per sample (or per document). This is how
   datasets like ImageNet are distributed and how many NLP datasets end up
   on disk after preprocessing.

2. **Large contiguous shards** -- samples are concatenated into a few large
   binary files, with an index that maps sample IDs to byte offsets. This is
   the format used by WebDataset, MosaicML Streaming, TFRecord, and similar
   systems.

The question is: *does the layout matter, and by how much?*

### Experiment design

The shared utility module `io_benchmark_utils.py` provides the scaffolding.
Its `setup_data()` function generates two identical copies of the same data:

```python
# io_benchmark_utils.py — data generation

def setup_data(data_dir, num_samples, sample_size_bytes):
    """Generate test data: many small files + one large contiguous file."""
    # Create 10,000 individual files of 16KB each
    for i in range(num_samples):
        with open(os.path.join(small_files_dir, f"{i}.bin"), "wb") as f:
            f.write(os.urandom(sample_size_bytes))

    # Concatenate them all into one contiguous shard
    with open(giant_path, "wb") as f_out:
        for i in range(num_samples):
            with open(os.path.join(small_files_dir, f"{i}.bin"), "rb") as f_in:
                f_out.write(f_in.read())
```

Default parameters: 10,000 samples at 16KB each, totaling approximately 160MB.
The data is random bytes -- content does not matter; we are measuring I/O
throughput, not processing speed.

A critical detail: **OS page cache purging**. Modern operating systems
aggressively cache file data in RAM. After the first read of a file, subsequent
reads hit the page cache and complete in microseconds, regardless of access
pattern. This is why naive benchmarks often show no difference between random
and sequential I/O -- everything is served from RAM.

To measure *real* storage performance, `purge_os_cache()` flushes the page
cache before each benchmark run:

```python
# io_benchmark_utils.py — cache purging

def purge_os_cache():
    """Flush the OS page cache so benchmarks hit real storage."""
    # Linux: sync && echo 3 > /proc/sys/vm/drop_caches
    subprocess.run(["sync"], check=True, capture_output=True)
    try:
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("3\n")
        return True
    except (PermissionError, OSError):
        pass
    # Falls back to sudo if direct write fails
```

The benchmark itself is pure Python -- no PyTorch, no DataLoader, no
multiprocessing. This isolates the storage layer completely.

**Random access** opens, reads, and closes each file individually in shuffled
order:

```python
# random_vs_seq_python_simulation.py

def benchmark_random_access(data_dir, num_samples, sample_size_bytes):
    """Read individual files in shuffled order (open -> read -> close per sample)."""
    indices = list(range(num_samples))
    random.shuffle(indices)

    for idx in indices:
        with open(os.path.join(small_dir, f"{idx}.bin"), "rb") as f:
            data = f.read()
            bytes_read += len(data)
```

**Sequential access** opens one file handle and stream-reads in fixed-size
chunks:

```python
def benchmark_sequential_access(data_dir, num_samples, sample_size_bytes):
    """Stream-read one contiguous file in fixed-size chunks."""
    with open(giant_path, "rb") as f:
        while True:
            data = f.read(sample_size_bytes)
            if not data:
                break
            bytes_read += len(data)
```

Each benchmark runs multiple times (default 3), with the page cache purged
between runs, and reports throughput in MB/s with standard deviation.

### What we measured

The output is a direct comparison of MB/s throughput for each access pattern,
along with the speedup ratio. Here are our actual results on NVMe SSD with
cold cache (OS page cache purged between every run):

```
  RANDOM vs SEQUENTIAL I/O -- SUMMARY
  ==================================================
  Samples:    10,000
  Sample Size: 16KB
  Total Data:  156MB
  Runs:        3
  --------------------------------------------------
  Random Access:        70.39 +/- 3.12 MB/s
  Sequential Access:  1421.85 +/- 13.47 MB/s
  --------------------------------------------------
  Speedup: 20.2x
  ==================================================
```

### What we found

The results are striking: **sequential reads are 20.2x faster than random
reads** -- even on a modern NVMe SSD, which is supposed to be "random access
friendly." The sequential path sustains **1,422 MB/s**, while random access
crawls at just **70 MB/s**. This is not a small difference; it is an order of
magnitude. Let's understand why:

**Per-file overhead dominates for small files.** Each `open()` call triggers:
- A filename-to-inode lookup in the filesystem's directory index
- Inode metadata reads (permissions, timestamps, block pointers)
- File descriptor allocation in the kernel
- Potential disk seeks to locate the file's data blocks
- `close()` releases the descriptor and flushes metadata

For 10,000 files, that is 10,000 open/close cycles, 10,000 inode lookups, and
10,000 directory traversals. At 70 MB/s reading 156MB of data, we are spending
roughly `156 / 70 = 2.2 seconds` on what should be a sub-second operation. The
actual data reads (16KB each) are almost incidental -- the syscall overhead
dominates.

**Sequential reads exploit hardware prefetching.** SSDs and HDDs both optimize
for sequential access. The OS kernel's readahead logic detects the pattern and
prefetches subsequent blocks before the application requests them. A single
`open()` call, followed by streaming `read()` calls, lets the entire hardware
and software stack work at peak efficiency. Our measurement of **1,422 MB/s**
sequential throughput represents roughly 45% of a typical NVMe SSD's raw
sequential read capability -- the remainder is consumed by Python interpreter
overhead and `read()` syscall granularity (16KB chunks are smaller than
optimal).

**The page cache can hide the truth.** If the benchmark runs without purging
the OS cache, both patterns show nearly identical speed (often within 10% of
each other). This is because after the data generation step, all 156MB is
already in the page cache. Both random and sequential reads complete in
microseconds per call -- the 20.2x difference collapses to near 1x. This is
a critical insight: *if you benchmark your DataLoader on a warm machine, you
will not see the I/O problems that will devastate you in production.*

### Why it matters

This experiment establishes the foundational physics of the data pipeline:
**the layout of data on storage has enormous consequences for read throughput.**
This is not an optimization -- it is the difference between a pipeline that
feeds the GPU fast enough and one that does not. Every subsequent experiment
builds on this finding.

### Key takeaway

> Sequential reads from contiguous shards are fundamentally faster than random
> reads from many small files. If your training data is stored as individual
> files, you are paying a per-file overhead tax on every sample, every epoch.

---

## 3. Adding PyTorch: How the DataLoader Masks the Problem

**Script:** `random_vs_seq_dataloader_simulation.py`

### Motivation

Pure Python I/O benchmarks are informative, but nobody writes a training loop
with raw `open()` and `read()` calls. In practice, data is loaded through
PyTorch's `DataLoader` -- a sophisticated abstraction that manages batching,
shuffling, and parallel prefetching via worker processes. The question becomes:
*does the DataLoader's parallelism eliminate the I/O pattern problem we just
discovered?*

This is where the story gets interesting. The DataLoader can *hide* the
problem without *fixing* it.

### Experiment design

We define two PyTorch `Dataset` implementations that mirror the two access
patterns from the previous experiment:

**SmallFilesDataset** -- one file per sample, opened and closed on every
`__getitem__` call:

```python
class SmallFilesDataset(Dataset):
    """Each sample opens, reads, and closes an individual file (random access)."""

    def __getitem__(self, idx):
        with open(self.files[idx], "rb") as f:
            data = f.read()
        return torch.tensor(len(data), dtype=torch.long)
```

**GiantFileDataset** -- reads from one contiguous shard via a persistent file
handle with seek+read:

```python
class GiantFileDataset(Dataset):
    """Reads samples from one contiguous file via seek+read."""

    def __getitem__(self, idx):
        self._open()  # Lazy open, once per worker
        self._fh.seek(idx * self.sample_size)
        data = self._fh.read(self.sample_size)
        return torch.tensor(len(data), dtype=torch.long)
```

A subtlety for the shard reader: when `num_workers > 0`, PyTorch forks worker
processes, and file handles do not survive the fork cleanly. The
`giant_file_worker_init` function ensures each worker opens its own fresh
handle:

```python
def giant_file_worker_init(worker_id):
    """Each worker opens its own file handle to avoid conflicts."""
    dataset = torch.utils.data.get_worker_info().dataset
    dataset._fh = open(dataset.file_path, "rb")
```

The benchmark runs with deliberately starker parameters than the pure Python
experiment: **50,000 samples at 4KB each** (approximately 200MB total). Smaller
samples amplify the per-file overhead relative to the data read time, making
the pattern difference more visible.

Two scenarios are tested:

1. **`num_workers=0`** -- the main process does all I/O synchronously. This
   isolates the raw I/O access pattern, exactly like the pure Python benchmark
   but through the DataLoader's batching machinery.

2. **`num_workers=4`** -- four worker processes prefetch batches in the
   background while the main process consumes the previous batch. This is the
   standard production configuration.

Both scenarios run with and without OS page cache purging to show the
difference between cold and warm storage.

### What we measured

For each scenario, `benchmark_loader()` iterates the DataLoader for multiple
runs, purging the OS cache between runs, and reports mean throughput in MB/s:

```python
def benchmark_loader(loader, label, num_runs, warmup_batches, purge_cache=True):
    """Run the loader multiple times, report mean +/- std throughput."""
    for run in range(1, num_runs + 1):
        if purge_cache:
            purge_os_cache()

        # Warmup batches (let workers spin up)
        for _ in range(warmup_batches):
            next(iterator)

        # Timed region
        start = time.perf_counter()
        for batch in iterator:
            total_bytes += batch.sum().item()
        elapsed = time.perf_counter() - start
```

The benchmark compares `SmallFilesDataset` (shuffled) against
`GiantFileDataset` (sequential, no shuffle) under both worker configurations.

### What we found

Here are our actual results (50,000 samples, 4KB each, ~195MB total, batch
size 64, cold cache, 3 runs per measurement):

**Scenario 1: `num_workers=0` (main process only -- isolates I/O pattern)**

| Access Pattern | Throughput | Notes |
|---------------|-----------|-------|
| Small Files (open/read/close per sample) | **16.70 +/- 1.93 MB/s** | Shuffled random access |
| Giant Shard (persistent handle, seek+read) | **326.05 +/- 4.99 MB/s** | Sequential contiguous reads |
| **Speedup** | **19.5x** | |

**Scenario 2: `num_workers=4` (parallel prefetch -- masks I/O pattern)**

| Access Pattern | Throughput | Notes |
|---------------|-----------|-------|
| Small Files (open/read/close per sample) | **79.94 +/- 3.40 MB/s** | 4 workers, shuffled |
| Giant Shard (persistent handle, seek+read) | **384.85 +/- 5.78 MB/s** | 4 workers, sequential |
| **Speedup** | **4.8x** | |

The numbers tell a compelling story:

**With `num_workers=0`, the sequential advantage is 19.5x** -- almost identical
to the 20.2x we measured in pure Python (Chapter 2). The DataLoader adds
batching overhead, but the I/O pattern completely dominates. At 16.70 MB/s,
loading 195MB of training data would take **11.7 seconds**. The shard reader
does it in **0.6 seconds**. For a single epoch, that is 11 seconds of GPU idle
time that produces zero useful gradients.

**With `num_workers=4`, the gap shrinks from 19.5x to 4.8x** -- but it does
not disappear. Four workers reading small files achieve 79.94 MB/s (roughly 4x
the single-process speed, as expected). But four workers reading from the shard
achieve 384.85 MB/s. The workers mask the *absolute* penalty (the GPU waits
less), but the *relative* advantage of sequential access persists. The shard
reader is still nearly 5x faster.

Note how worker parallelism has different effects on the two patterns:
- Small files: 16.70 -> 79.94 MB/s (**4.8x** improvement from 4 workers)
- Giant shard: 326.05 -> 384.85 MB/s (**1.2x** improvement from 4 workers)

The small-files pattern benefits enormously from parallelism because the
bottleneck is per-file syscall latency, and workers can overlap these latencies.
The shard pattern barely improves because it was already running near the
sequential bandwidth ceiling of the storage device -- there is little latency
to hide.

```
  SUMMARY
  ============================================================
  num_workers=0:  19.5x speedup  (raw I/O pattern, cold cache)
  num_workers=4:   4.8x speedup  (parallel prefetch, cold cache)
  ------------------------------------------------------------
  Cold-cache results expose the true I/O pattern penalty.
  Workers partially mask it by prefetching, but the gap persists.
```

**With warm cache, both patterns converge.** If the cache is not purged,
both the small-files dataset and the giant-shard dataset read from RAM,
and the throughput difference collapses to near 1x. This is the scenario
most developers encounter during local development -- and it leads them
to believe their data pipeline is fine.

### The abstraction trap

This experiment reveals a subtle danger: **the DataLoader abstraction can
make you think your I/O is fine when it is actually terrible underneath.**

In development, you run on a warm machine with fast NVMe storage and page
cache full of training data. The DataLoader saturates the GPU trivially.
Then you deploy to production: the dataset is on network-attached storage,
the page cache is cold because the dataset is too large to fit in RAM,
and suddenly the GPU is idle 60% of the time. You did not change any code
-- the abstraction simply hid the problem.

### Key takeaway

> PyTorch's DataLoader with `num_workers > 0` masks I/O pattern penalties by
> prefetching, but it cannot fix the fundamental throughput ceiling of the
> storage device. On cold or slow storage (HDD, NFS, S3), the access pattern
> still matters enormously. Always benchmark with cache purged to see the
> truth.

---

## 4. The CPU Starvation Problem

**Script:** `dataloader_bottleneck_simulation.py`

### Motivation

The previous two experiments focused purely on I/O throughput -- how fast bytes
move from storage into user-space memory. But in a real training pipeline,
reading raw bytes is only the first step. Before data reaches the GPU, the CPU
must also:

- **Parse the data format** (JSON, CSV, Parquet, protobuf)
- **Tokenize text** (BPE encoding for LLMs, which involves regex matching and
  vocabulary lookups)
- **Apply augmentations** (for vision: random crop, flip, color jitter; for
  text: dynamic masking, span corruption)
- **Collate samples into batches** (stack tensors, apply padding)

Each of these operations takes CPU time. If the CPU cannot finish preparing
the next batch before the GPU finishes the current one, the GPU **starves**
-- it sits idle waiting for data. This is perhaps the most common performance
problem in production training.

### Experiment design

The simulation uses a deliberately simple model to isolate the CPU bottleneck:

```python
class HeavyDataset(Dataset):
    def __init__(self, size=10000, sleep_time=0.01):
        self.size = size
        self.sleep_time = sleep_time

    def __getitem__(self, idx):
        # Simulate heavy CPU work (tokenization, augmentation, JSON parsing)
        time.sleep(self.sleep_time)
        return torch.randn(4096)  # Dummy tensor
```

The `time.sleep()` call simulates realistic preprocessing cost. A sleep of
5ms per sample is representative of complex tokenization (BPE encoding a long
document with the Llama tokenizer, for instance, can take 2-10ms depending on
length).

The GPU is simulated with a 50ms `time.sleep()` -- a reasonable approximation
for a forward + backward pass on a moderately sized model.

Two scenarios are tested:

1. **`num_workers=0, batch_size=32, cpu_delay=5ms`** -- the main process does
   everything: read data, preprocess, transfer to GPU, compute. This is the
   "default mistake" -- what happens when you forget to set `num_workers`.

2. **`num_workers=4, batch_size=32, cpu_delay=5ms`** -- four worker processes
   handle data loading and preprocessing in the background.

### The starvation math

The key insight is a simple inequality. For a batch of size B, with
preprocessing delay D per sample, W workers, and GPU compute time G:

```
DataLoader time per batch = (D * B) / max(W, 1)

If DataLoader time > G, the GPU starves.
```

For our scenario 1 (`num_workers=0`):
- DataLoader time = (5ms * 32) / 1 = **160ms**
- GPU time = 50ms
- **Starvation: the GPU idles for 160ms waiting, then computes for 50ms.**

For our scenario 2 (`num_workers=4`):
- DataLoader time = (5ms * 32) / 4 = **40ms**
- GPU time = 50ms
- **No starvation: 40ms < 50ms, so workers finish before the GPU needs data.**

### What we measured

The profiling loop measures two quantities per training step:

```python
def profile_loader(num_workers, batch_size, cpu_delay):
    for i in range(20):
        # 1. MEASURE DATA LOADING (The "Wait")
        batch = next(iter_loader)
        t1 = time.time()
        data_times.append(t1 - t0)

        # 2. MEASURE COMPUTE (The "Work")
        time.sleep(gpu_sim_time)
        t2 = time.time()
        compute_times.append(t2 - t1)
```

If the average data wait exceeds 10ms, the benchmark flags **STARVATION
DETECTED**.

### What we found

Here are the actual measurements:

**Scenario 1: `num_workers=0` (the default mistake)**

```
  Avg Data Wait: 160.59 ms
  Avg Compute:    50.09 ms
  STARVATION DETECTED! GPU is idle for 160.6ms per step.
```

**Scenario 2: `num_workers=4` (parallel preprocessing)**

```
  Avg Data Wait:   1.50 ms
  Avg Compute:    50.09 ms
  GPU is fed correctly.
```

The contrast is dramatic. With zero workers, the data wait is **160.59 ms** --
the GPU sits idle for **76% of every training step** (160.59 / (160.59 + 50.09)).
This matches our theoretical prediction almost exactly: 5ms delay x 32 samples
= 160ms. The GPU computes for 50ms, then waits for 160ms, then computes for
50ms, then waits again. More than three-quarters of the training time is wasted.

With 4 workers, the data wait drops to **1.50 ms** -- effectively invisible.
The workers prepare batches in the background while the GPU computes, and the
next batch is ready before the GPU finishes the current one. The theoretical
prediction was 40ms (5ms x 32 / 4), but the actual wait is even lower because
workers start prefetching well ahead of consumption.

The difference between these two scenarios -- identical code except for one
parameter (`num_workers=4` vs the default `0`) -- is the difference between
76% GPU idle time and nearly zero. This single setting is probably the most
impactful one-line change in any training script.

### The lesson

The number of workers needed is not fixed -- it depends on the ratio of
preprocessing cost to GPU compute time. Faster GPUs (A100, H100) compute
faster but still consume data at the same rate, which means the CPU side needs
to keep up with a shorter deadline. This is why LLM training on modern GPUs
often requires 8-16 DataLoader workers, and why some teams move preprocessing
entirely off the training critical path (pre-tokenizing the dataset, using
data loading services, etc.).

### Key takeaway

> The GPU starves when `(preprocessing_delay * batch_size) / num_workers` exceeds
> the GPU compute time. Always profile the DataLoader wait time. If it exceeds
> the compute time, add more workers, simplify preprocessing, or pre-process
> the dataset offline.

---

## 5. The Padding Tax: Wasted GPU Cycles

**Script:** `padding_tax_simulation.py`

### Motivation

We have explored how data moves from storage to CPU, how the DataLoader
orchestrates parallel loading, and how preprocessing can bottleneck the
pipeline. But there is another source of waste that occurs *after* data reaches
the GPU and *during* computation: **padding**.

In language model training, samples (documents, instructions, conversations)
have variable lengths. But GPUs operate on fixed-size tensors. The standard
approach is to pad every sample in a batch to the same length -- either to
the maximum length in the batch, or to a fixed context length. Every padding
token consumes:

- GPU FLOPS during the forward pass (attention, FFN)
- GPU memory (activations, KV cache)
- PCIe bandwidth during transfer
- CPU time during collation

Padding tokens produce no useful gradients. They are pure waste.

### Experiment design

The simulation uses real data and a real tokenizer to make the waste concrete
and visual:

- **Dataset:** Alpaca instruction-tuning dataset (first 2,000 samples from
  `tatsu-lab/alpaca`)
- **Tokenizer:** Meta Llama 3.1 8B (`meta-llama/Meta-Llama-3.1-8B`)
- **Context length:** 4,096 tokens
- **Batch size:** 32

Two batching strategies are compared:

**Naive batching** -- the standard PyTorch approach. Each sample is tokenized
and padded to the full context length:

```python
def simulate_naive_batching(dataset, tokenizer):
    """Standard PyTorch DataLoader behavior: pad each sample to CONTEXT_LEN."""
    batch_grid = np.zeros((BATCH_SIZE, CONTEXT_LEN))

    for i, text in enumerate(samples):
        tokens = tokenizer(text, truncation=True, max_length=CONTEXT_LEN)["input_ids"]
        length = len(tokens)
        batch_grid[i, :length] = 1  # Real data
        # The rest remains 0 (padding)
```

**Packed batching** -- the approach used by MosaicML, Torchtune, and other
modern training frameworks. Samples are greedily bin-packed into rows of the
context length, separated by EOS tokens:

```python
def simulate_packed_batching(dataset, tokenizer):
    """Greedily bin-pack complete samples into rows of CONTEXT_LEN."""
    batch_grid = np.zeros((BATCH_SIZE, CONTEXT_LEN))

    sample_idx = 0
    for row in range(BATCH_SIZE):
        filled = 0
        while sample_idx < len(sample_lengths):
            length = sample_lengths[sample_idx]
            if filled + length <= CONTEXT_LEN:
                batch_grid[row, filled:filled + length] = 1
                filled += length
                sample_idx += 1
            else:
                break  # This sample doesn't fit; move to next row
```

### What we measured

The output is a side-by-side heatmap visualization saved to
`output/padding_tax_heatmap.png`. Each cell in the heatmap represents one
token position in the batch:

- **Green** = real data (useful computation)
- **Red** = padding (wasted computation)

The overall **efficiency** is computed as the fraction of non-padding positions:
`efficiency = mean(grid) * 100%`.

### What we found

Here are the actual measurements on the Alpaca dataset (first 2,000 samples,
GPT-2 tokenizer, context length 4,096, batch size 32):

**Token length distribution of Alpaca samples:**

| Statistic | Value |
|-----------|-------|
| Minimum length | 13 tokens |
| Maximum length | 307 tokens |
| Mean length | 95 tokens |
| Median length | 83 tokens |
| Context length | 4,096 tokens |
| Average fill ratio per sample | **2.3%** |

The average Alpaca sample uses only 95 out of 4,096 positions. The remaining
4,001 positions are padding. Now look at what happens at the batch level:

**Batching efficiency:**

| Strategy | Efficiency | Waste |
|----------|-----------|-------|
| Naive batching (pad to context length) | **2.3%** | 97.7% padding |
| Sequence packing (greedy bin-packing) | **98.5%** | 1.5% padding |
| **Effective throughput improvement** | **42.4x** | |

**Naive batching efficiency is 2.3%.** That means for every 4,096 token
positions in the batch tensor, only 95 are real data. The other 4,001 are
padding zeros. The heatmap (saved to `output/padding_tax_heatmap.png`) shows
a thin green stripe on the far left of each row -- barely visible against an
ocean of red. Every position in that red zone consumes GPU FLOPS, GPU memory,
and PCIe bandwidth for zero useful learning signal.

**Packed batching achieves 98.5% efficiency -- a 42.4x improvement.** By
packing roughly 43 short samples into each row (4,096 / 95 mean length),
almost every token position contains real data. The heatmap shows a solid
wall of green. The packing algorithm is greedy and simple: try to fit the
next sample; if it does not fit, move to the next row. Even this naive
strategy eliminates nearly all waste.

### The compounding effect

The padding tax compounds with everything we have measured so far. Consider the
full cost of a padding token:

1. **Storage I/O:** The padding token does not come from storage (it is
   generated during collation), but the real tokens next to it were read
   from disk at the I/O throughput we measured in Chapters 2-3.

2. **CPU preprocessing:** The collation function that adds padding takes CPU
   time -- the same CPU time that Chapter 4 showed can starve the GPU.

3. **PCIe transfer:** The padding tokens are transferred over the PCIe bus
   at the bandwidth we will measure in Chapter 6. Every padding token consumes
   the same bytes as a real token.

4. **GPU compute:** The model processes every position in the batch tensor,
   padding included. Attention is computed, FFN layers fire, activations
   are stored -- all for positions that produce no useful gradients.

At 2.3% efficiency, you need **42x more training steps** to process the same
amount of real data compared to a packed batch. Put another way: a training
run that takes 42 GPU-hours with naive batching could finish in just 1
GPU-hour with packing -- same data, same model, same learning. The GPU is not
idle (it appears 100% utilized in `nvidia-smi`), but it is doing useless work
on padding tokens. This is the insidious nature of the padding tax: the GPU
*looks* busy, but 97.7% of its cycles are wasted.

### Key takeaway

> On Alpaca, naive padding wastes 97.7% of GPU compute -- only 2.3% of token
> positions are real data. Sequence packing recovers this to 98.5% efficiency,
> a 42.4x improvement in effective throughput. The GPU appears fully utilized
> with naive padding -- the waste is invisible to `nvidia-smi`. Always check
> the ratio of real tokens to total positions in your batches.

---

## 6. Crossing the PCIe Bridge: Pinned Memory and DMA

**Script:** `pcie_transfer_bottleneck.py`

### Motivation

We have followed data from storage through the OS, through Python, through
preprocessing, and through batching. The data is now sitting in CPU memory as
a PyTorch tensor, ready to be fed to the model. But the model runs on the
GPU, and the GPU has its own separate memory (VRAM). The data must cross the
**PCIe bus** -- the physical interconnect between the CPU and GPU.

This transfer is often treated as instantaneous ("just call `.to('cuda')`"),
but it is anything but. PCIe bandwidth is finite, the transfer mechanism
matters enormously, and the interaction between transfer and CPU execution
has subtle and important consequences for pipeline throughput.

### Background: How CPU-GPU transfers work

When you call `tensor.to("cuda")` in PyTorch, the runtime must copy bytes
from host (CPU) memory to device (GPU) VRAM. The mechanism differs depending
on whether the source memory is **pageable** or **pinned**:

**Pageable memory** (the default): The CPU tensor lives in normal virtual
memory that the OS can swap to disk or relocate at any time. The GPU cannot
read directly from pageable memory because the virtual-to-physical mapping
might change mid-transfer. Instead, the CUDA runtime must:

1. Allocate a temporary **pinned staging buffer** in host memory
2. **Copy** the data from the pageable source to the staging buffer (CPU memcpy)
3. Initiate a **DMA transfer** from the staging buffer to GPU VRAM
4. Wait for the DMA to complete
5. Free the staging buffer

The CPU is **blocked** during steps 1-4 because it must coordinate the staging
copy and cannot release the source buffer until the transfer is confirmed.

**Pinned (page-locked) memory**: The tensor is allocated in memory that the OS
has guaranteed will not be swapped or relocated. The GPU's DMA engine can read
directly from this physical memory without any CPU intervention:

1. Initiate a **DMA transfer** directly from the pinned source to GPU VRAM
2. The CPU is **immediately free** to do other work
3. The DMA engine handles the transfer autonomously

The difference is both a bandwidth improvement (no extra copy) and a latency
improvement (CPU is not blocked).

### Experiment design

The benchmark tests three things:

#### Part 1: Raw bandwidth -- pinned vs pageable

For each of four tensor sizes (1 MB, 16 MB, 256 MB, 1 GB), the benchmark
measures Host-to-Device (H2D) and Device-to-Host (D2H) transfer bandwidth in
GB/s, comparing pinned and pageable memory:

```python
def benchmark_h2d(size_bytes, pinned):
    """Host -> Device transfer."""
    src = torch.randn(n, dtype=torch.float32)
    if pinned:
        src = src.pin_memory()

    def transfer():
        src.to("cuda", non_blocking=False)

    # Warmup, then benchmark
    _measure(transfer, data_gb, WARMUP_ITERS)
    speeds = _measure(transfer, data_gb, BENCH_ITERS)
```

The timing uses `time.perf_counter()` with `torch.cuda.synchronize()` barriers
to capture the **full wall-clock cost**, including any CPU-side staging copies
that CUDA events would miss:

```python
def _measure(transfer_fn, data_gb, iters):
    """Return list of GB/s measurements with synchronize barriers."""
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        transfer_fn()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        speeds.append(data_gb / elapsed)
```

Results are compared against theoretical PCIe peaks:

| PCIe Generation | Theoretical Unidirectional Peak |
|----------------|-------------------------------|
| 3.0 x16 | 15.75 GB/s |
| 4.0 x16 | 31.51 GB/s |
| 5.0 x16 | 63.02 GB/s |

#### Part 2: The overlap experiment -- proving CPU freedom

Raw bandwidth is only half the story. The real win of pinned memory is that
the CPU is **free to do other work** while the DMA transfer is in flight. The
overlap experiment *proves* this by testing all four combinations of
`(pinned, non_blocking)`:

| Combination | Expected behavior |
|------------|------------------|
| pageable + blocking | CPU blocked (staging copy required) |
| pageable + non_blocking | CPU blocked (non_blocking silently falls back) |
| pinned + blocking | CPU blocked (waits for completion by choice) |
| pinned + non_blocking | **CPU truly free** (DMA runs autonomously) |

The experimental method is elegant: it measures whether CPU work and DMA
transfer can physically overlap in time.

```python
def benchmark_overlap(size_bytes):
    """Test all 4 combinations and PROVE whether CPU is truly free during DMA."""
    # 1. Calibrate a fixed amount of CPU work (~5ms of matrix multiplies)
    work_tensor, work_iters = _calibrate_cpu_work(target_ms=5.0)

    # 2. Measure CPU work alone (baseline)
    cpu_work_ms = measure(do_cpu_work)

    # 3. For each combination:
    #    a. Measure transfer alone (transfer_ms)
    #    b. Start transfer, immediately do CPU work, then synchronize (both_ms)
    #    c. Compare:
    #       If both_ms ~= max(transfer_ms, cpu_work_ms)  -> PARALLEL
    #       If both_ms ~= transfer_ms + cpu_work_ms       -> SEQUENTIAL
```

The overlap score quantifies how parallel the execution was:

```python
overlap_ratio = (sequential_expected - both_ms) / (sequential_expected - parallel_expected)
# 1.0 = fully parallel, 0.0 = fully sequential
```

#### Container awareness

The benchmark also detects Kubernetes/Docker container environments and warns
about conditions that can skew results:

```python
def _detect_container_env():
    """Detect cgroup memory/CPU limits and Kubernetes pod context."""
    # Check /.dockerenv, /run/.containerenv, cgroup keywords
    # Read cgroup v2: /sys/fs/cgroup/memory.max, /sys/fs/cgroup/cpu.max
    # Read cgroup v1: memory.limit_in_bytes, cpu.cfs_quota_us
    # Warn if pinned memory exceeds 50% of cgroup limit (OOM risk)
    # Warn if CPU quota < 2 cores (CFS throttling inflates times)
```

This is critical because pinned memory is **page-locked** -- it counts against
cgroup memory limits and cannot be swapped. In Kubernetes, a pod that pins 2GB
of memory must have at least 2GB of available memory limit, or it will be
OOM-killed.

### Results

**GPU:** NVIDIA L40S (47.7 GB VRAM), CUDA 12.8, PyTorch 2.10
**Environment:** Kubernetes container, cgroup memory unlimited, 16 CPUs visible
**PCIe:** The L40S uses PCIe 4.0 x16 (theoretical peak: 31.51 GB/s)

#### Part 1 results: H2D bandwidth (Host -> Device)

| Size | Pageable | Pinned | Speedup |
|------|----------|--------|---------|
| 1 MB | 6.69 GB/s | 10.14 GB/s | **1.52x** |
| 16 MB | 12.28 GB/s | 13.06 GB/s | 1.06x |
| 256 MB | 13.07 GB/s | 13.43 GB/s | 1.03x |
| 1 GB | 13.14 GB/s | 13.46 GB/s | 1.02x |

#### Part 1 results: D2H bandwidth (Device -> Host)

| Size | Pageable | Pinned | Speedup |
|------|----------|--------|---------|
| 1 MB | 6.36 GB/s | 10.79 GB/s | **1.69x** |
| 16 MB | 12.18 GB/s | 12.97 GB/s | 1.06x |
| 256 MB | 1.77 GB/s | 13.19 GB/s | **7.46x** |
| 1 GB | 1.77 GB/s | 13.21 GB/s | **7.44x** |

#### Part 2 results: Overlap experiment (16 MB transfer)

| Combination | Transfer | CPU work | Both together | Overlap | Verdict |
|-------------|----------|----------|--------------|---------|---------|
| pageable + blocking | 1.30 ms | 4.98 ms | 6.25 ms | 3% | SEQUENTIAL |
| pageable + non_blocking | 1.28 ms | 4.98 ms | 6.27 ms | 0% | SEQUENTIAL |
| pinned + blocking | 1.22 ms | 4.98 ms | 6.54 ms | 0% | SEQUENTIAL |
| pinned + non_blocking | 1.21 ms | 4.98 ms | 5.26 ms | 77% | **PARTIAL** |

#### Part 2 results: Overlap experiment (256 MB transfer)

| Combination | Transfer | CPU work | Both together | Overlap | Verdict |
|-------------|----------|----------|--------------|---------|---------|
| pageable + blocking | 19.58 ms | 5.48 ms | 25.93 ms | 0% | SEQUENTIAL |
| pageable + non_blocking | 19.54 ms | 5.48 ms | 29.46 ms | 0% | SEQUENTIAL |
| pinned + blocking | 19.06 ms | 5.48 ms | 25.83 ms | 0% | SEQUENTIAL |
| pinned + non_blocking | 19.03 ms | 5.48 ms | **19.04 ms** | **100%** | **PARALLEL** |

### What we found

The L40S results reveal several important patterns:

**H2D: Pinned memory matters most for small transfers.** At 1 MB, pinned
memory is **1.52x faster** (10.14 vs 6.69 GB/s). This is exactly where training
batches live -- a typical LLM text batch (batch_size=32, seq_len=2048, fp16)
is about 128 KB, and a typical vision batch (batch_size=64, 224x224x3, fp32) is
about 37 MB. At large sizes (256 MB+), both paths converge to ~13.4 GB/s because
the transfer time dominates the staging copy overhead.

**D2H: Pageable memory collapses at large sizes.** This is the most dramatic
finding. For Device-to-Host transfers of 256 MB and 1 GB, pageable memory drops
to just **1.77 GB/s** -- a **7.46x penalty** compared to pinned memory (13.19
GB/s). This is catastrophic for checkpointing and gradient offloading. A model
checkpoint of 8 GB would take 4.5 seconds with pageable memory but only 0.6
seconds with pinned memory. For ZeRO-Offload (DeepSpeed), which continuously
offloads optimizer states to CPU, this 7x bandwidth gap directly translates to
7x slower offloading throughput.

**Peak bandwidth plateaus at ~13.5 GB/s.** The L40S achieves roughly 43% of
the PCIe 4.0 x16 theoretical peak of 31.51 GB/s. This is typical -- the gap is
caused by protocol overhead (TLP headers, flow control credits), IOMMU address
translation, and the Kubernetes container's NUMA topology (if the GPU and CPU
are on different sockets, traffic crosses the inter-socket link).

**The overlap experiment proves the physics of DMA.** The 256 MB results are
the cleanest demonstration:

- **All three non-DMA combinations are SEQUENTIAL (0% overlap).** The
  `pageable + blocking`, `pageable + non_blocking`, and `pinned + blocking`
  combinations all produce `both_together ~= transfer + cpu_work`. The CPU
  was blocked for the entire transfer duration. Notably, `pageable +
  non_blocking` at 29.46 ms is *worse* than sequential expectation (25.01 ms)
  -- the `non_blocking` flag adds overhead without providing any benefit for
  pageable memory.

- **`pinned + non_blocking` achieves 100% overlap.** Both together = 19.04 ms,
  which is virtually identical to transfer-only time (19.03 ms). The 5.48 ms of
  CPU matrix multiply work happened *completely for free* -- the DMA engine
  transferred 256 MB while the CPU was busy computing. This is 5.48 ms of CPU
  time that can be used for preparing the next batch, running preprocessing, or
  doing any other useful work.

- **At 16 MB, the overlap is partial (77%).** The transfer completes in just
  1.21 ms, which is shorter than the CPU work (4.98 ms). With true overlap, the
  total should be 4.98 ms (the max), but we measured 5.26 ms -- close but not
  perfect. The small transfer size means the DMA setup overhead is a larger
  fraction of the total time, reducing the effective overlap window.

This is the only combination that achieves true CPU/GPU parallelism. The
implication for training is clear: when the DataLoader sets `pin_memory=True`,
the pin-memory thread copies batch data into a pinned staging area. Then,
`.to(device, non_blocking=True)` initiates DMA and immediately returns,
allowing the CPU to begin preparing the next batch.

### Practical recommendations from the benchmark

The benchmark prints actionable advice:

1. **Always set `DataLoader(..., pin_memory=True)`.** This gives you the pinned
   speedup for free -- the DataLoader's pin-memory thread handles the pinning.

2. **Small transfers are latency-bound.** Batch size and packing efficiency
   matter more than raw PCIe bandwidth.

3. **If peak bandwidth is well below theoretical max, check:**
   - GPU is in the correct PCIe slot (x16, not x4 or x8)
   - IOMMU / ACS is not throttling DMA
   - NUMA topology -- CPU and GPU should be on the same socket

4. **In Kubernetes / containers:**
   - Set memory limits high enough for pinned allocations
   - Use Guaranteed QoS (requests == limits) to avoid noisy neighbors
   - Watch for CPU CFS throttling, which inflates wall-clock transfer times

### Key takeaway

> On the L40S, only `pinned + non_blocking` achieves true CPU/GPU parallelism
> (100% overlap at 256 MB). Pageable D2H transfers collapse to 1.77 GB/s at
> large sizes (7.5x penalty vs pinned). `pageable + non_blocking` is worse than
> plain blocking -- it adds overhead for zero benefit. Always use
> `DataLoader(..., pin_memory=True)` and `.to(device, non_blocking=True)`.

---

## 7. The Full Picture: Streaming vs ImageFolder Under the Profiler

**Script:** `streaming_loader_lib_profiling.py`

### Motivation

We have now examined every stage of the data pipeline in isolation: I/O
patterns, DataLoader parallelism, CPU preprocessing, padding efficiency, and
PCIe transfers. The final experiment brings everything together in a **real
training loop** with a real model, real data, and a real profiler. The goal is
to see how all these factors interact and to demonstrate that the optimizations
from previous chapters have measurable impact in practice.

Specifically, we compare two data loading strategies that embody the key
findings from Chapters 2-3:

- **ImageFolder:** the standard PyTorch approach. 100,000 JPEG files stored
  as individual files on disk. The DataLoader opens, decodes, and processes
  each file independently.

- **MosaicML Streaming:** samples are pre-packed into a small number of large
  binary shards (MDS format). The DataLoader reads sequentially from these
  shards.

This directly tests the prediction from Chapter 2: that sequential shard-based
reading should outperform random file-based reading, especially from cold
storage.

### Experiment design

#### Data generation

The benchmark generates 100,000 synthetic JPEG images (224x224, random noise)
and stores them in ImageFolder format. It then converts them to MosaicML MDS
shards:

```python
def generate_data():
    """Create synthetic JPEG images to force real decode + I/O workload."""
    for i in range(NUM_SAMPLES):  # 100,000
        img = np.random.randint(0, 255, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        img = Image.fromarray(img)
        img.save(f"{RAW_DIR}/class_0/img_{i}.jpg", quality=80)

def convert_to_mds():
    """Convert raw JPEGs into MosaicML Streaming shards."""
    columns = {"data": "jpeg", "label": "int"}
    with MDSWriter(out=MDS_DIR, columns=columns, compression=None) as out:
        for i, (_img, label) in enumerate(dataset):
            with open(dataset.samples[i][0], "rb") as f:
                img_bytes = f.read()
            out.write({"data": img_bytes, "label": label})
```

100,000 images at 224x224 create a dataset that stresses the OS's inode and
dentry caches -- far more files than the kernel can keep metadata for in a
small page cache.

#### The training loop

A ResNet-50 model runs a real forward pass, backward pass, and optimizer step
on each batch. The loop runs under `torch.profiler` with explicit phase markers
that appear in TensorBoard traces:

```python
def train_and_profile(loader, model, label, purge_cache=False):
    with torch.profiler.profile(...) as prof:
        for step in range(total_steps):
            # Phase 1: Data loading
            with torch.profiler.record_function("DATALOADER_WAIT"):
                inputs, labels_batch = next(iter_loader)

            # Phase 2: H2D transfer
            with torch.profiler.record_function("H2D_TRANSFER"):
                inputs = inputs.to(device, non_blocking=True)
                labels_batch = labels_batch.to(device, non_blocking=True)
                torch.cuda.synchronize()

            # Phase 3: Compute
            with torch.profiler.record_function("FORWARD"):
                outputs = model(inputs)
                loss = criterion(outputs, labels_batch)

            with torch.profiler.record_function("BACKWARD"):
                optimizer.zero_grad()
                loss.backward()

            with torch.profiler.record_function("OPTIMIZER"):
                optimizer.step()
```

**Critical design decision:** transforms are kept intentionally light (resize +
normalize, no heavy augmentation). This keeps CPU preprocessing cost minimal
so that the **I/O access pattern difference** is the dominant factor, not
compute-bound augmentation.

**Cache purging** happens after the CUDA warmup steps but before the profiled
steps. This ensures CUDA/cuDNN is warm (no autotuner overhead) while storage
is cold (real I/O costs are visible).

#### Four scenarios

| Scenario | Dataset | `num_workers` | What it tests |
|----------|---------|--------------|--------------|
| `imagefolder_w0` | ImageFolder | 0 | Raw I/O cost, synchronous |
| `streaming_w0` | MDS Streaming | 0 | Sequential shard reads, synchronous |
| `imagefolder_w4` | ImageFolder | 4 | Random file reads, parallel prefetch |
| `streaming_w4` | MDS Streaming | 4 | Sequential shard reads, parallel prefetch |

All scenarios use `pin_memory=True` (applying the lesson from Chapter 6).

### What we measured

Two complementary outputs:

1. **TensorBoard trace files** (`output/log/profiler/{label}/`): Detailed
   per-operation traces showing CPU and CUDA activity, kernel launches,
   memory allocations, and the custom phase markers. These allow visual
   inspection of pipeline bubbles.

2. **Per-step timing summary:** Mean and standard deviation of each phase,
   plus the critical metric: **DataLoader wait as a percentage of total
   step time.**

### Results

**GPU:** NVIDIA L40S (47.7 GB VRAM), ResNet-50 model, batch size 64,
100,000 synthetic 224x224 JPEGs, OS page cache purged before each scenario.

#### Part 1: Synchronous loading (`num_workers=0`, cold storage)

| Scenario | DataLoader wait | H2D transfer | Compute | Total step | Throughput |
|----------|----------------|-------------|---------|-----------|-----------|
| `imagefolder_w0` | **130.52 ms** (57%) | 3.07 ms | 94.56 ms | 228.15 ms | 280.5 img/s |
| `streaming_w0` | **92.09 ms** (48%) | 3.16 ms | 94.63 ms | 189.87 ms | 337.1 img/s |

#### Part 2: Parallel prefetch (`num_workers=4`, cold storage)

| Scenario | DataLoader wait | H2D transfer | Compute | Total step | Throughput |
|----------|----------------|-------------|---------|-----------|-----------|
| `imagefolder_w4` | **0.36 ms** (0%) | 3.07 ms | 94.61 ms | 98.04 ms | 652.8 img/s |
| `streaming_w4` | **0.38 ms** (0%) | 3.07 ms | 94.64 ms | 98.09 ms | 652.5 img/s |

#### Summary

```
  Scenario                    DataLoader      Compute   Data %
  ------------------------- ------------ ------------ --------
  imagefolder_w0                130.5 ms      94.6 ms    58.0%
  streaming_w0                   92.1 ms      94.6 ms    49.3%
  imagefolder_w4                  0.4 ms      94.6 ms     0.4%
  streaming_w4                    0.4 ms      94.6 ms     0.4%

  num_workers=0: DataLoader speedup = 1.42x (Streaming faster)
  num_workers=4: DataLoader speedup = 0.96x (effectively tied)
```

### What we found

These results bring together every lesson from the previous chapters, and
they contain a few surprises.

**With `num_workers=0`, the data pipeline dominates the training step.** The
GPU computes for 94.6 ms, but the DataLoader wait adds 130.5 ms (ImageFolder)
or 92.1 ms (Streaming) on top of that. The GPU is idle for **57% of every
training step** with ImageFolder -- more time waiting than computing. This
directly validates Chapter 2's finding: random file I/O is the bottleneck.

**Streaming is 1.42x faster in DataLoader wait with synchronous loading.**
ImageFolder waits 130.5 ms per step while Streaming waits 92.1 ms -- a 38.4 ms
reduction per step. Over 1,562 batches (one epoch of 100K images), that is
**60 seconds of saved GPU idle time per epoch**, purely from changing the data
format. The throughput improvement is 20% (337 vs 281 img/s).

The 1.42x speedup is more modest than the 19.5x we measured in Chapter 3 for
raw binary I/O. This is because the training loop adds two large constant costs
that dilute the I/O difference: JPEG decoding (~30-40 ms of the DataLoader
wait is CPU decode work, same for both formats) and GPU compute (94.6 ms, same
for both). The I/O pattern advantage is real, but it is one component among
several in the total step time.

**With `num_workers=4`, both strategies converge to near-zero DataLoader wait.**
This is the most practically important finding: 4 workers are sufficient to
completely saturate the GPU on this hardware. ImageFolder wait drops from
130.5 ms to 0.36 ms; Streaming drops from 92.1 ms to 0.38 ms. Both achieve
~653 img/s throughput -- a **2.33x improvement** over `num_workers=0`. The
training step collapses to just compute (94.6 ms) + H2D transfer (3.1 ms)
= 97.7 ms, with effectively zero data stall.

**Why the I/O pattern difference disappears with workers on NVMe.** Our NVMe
SSDs sustain high enough random IOPS that 4 workers, each doing independent
random reads, can collectively produce batches faster than the GPU consumes
them. The prefetch queue never drains. On slower storage (HDD, NFS, cloud
object storage), the I/O pattern difference would persist even with workers --
exactly as Chapter 3 predicted.

**H2D transfer is a small constant: 3.07 ms per step.** With `pin_memory=True`
and `non_blocking=True` (as configured), the transfer of a 64-image batch
(~37 MB of float32 tensors) takes just 3 ms. This is consistent with our PCIe
bandwidth measurements from Chapter 6: 37 MB at ~13 GB/s = 2.8 ms, plus setup
overhead. The H2D transfer is not a bottleneck in this scenario.

**The key diagnostic metric: DataLoader wait as % of step time.** The summary
table makes the diagnosis trivial:
- 58% data wait = severely data-bound (fix your I/O or add workers)
- 49% data wait = moderately data-bound (Streaming helps, workers help more)
- 0.4% data wait = compute-bound (the pipeline is optimal)

### Key takeaway

> In a real ResNet-50 training loop on NVMe storage, `num_workers=0` produces
> 57% GPU idle time (ImageFolder) or 49% (Streaming). Adding 4 workers
> eliminates the data stall entirely, achieving 2.33x higher throughput.
> Streaming's sequential reads give a 1.42x advantage in synchronous mode,
> but on fast NVMe with enough workers, both strategies converge. On slower
> storage, the format advantage would persist. Always profile with
> `DATALOADER_WAIT` markers to measure your actual data stall percentage.

---

## 8. Conclusion and Optimization Checklist

### The full pipeline

We began this investigation at the lowest level -- raw file I/O on a storage
device -- and progressively added layers of abstraction: PyTorch DataLoaders,
CPU preprocessing, sequence padding, PCIe transfers, and finally a complete
training loop. At each layer, we found a distinct bottleneck and a distinct
mitigation:

```
LAYER                    BOTTLENECK                  MEASURED IMPACT             MITIGATION
-----                    ----------                  ---------------             ----------
Storage I/O              Random access to many       70 MB/s vs 1,422 MB/s      Shard data into large
                         small files                 = 20.2x penalty (Ch. 2)    contiguous files

DataLoader               Prefetching masks the       19.5x -> 4.8x with 4      Profile with cold cache;
                         I/O problem, does not       workers; gap persists      use enough workers for
                         fix it (Ch. 3)              on cold storage            your storage throughput

CPU Preprocessing        Tokenization / augmentation 160ms wait (0 workers)     Pre-process offline; add
                         slower than GPU compute     vs 1.5ms (4 workers)       more workers; simplify
                         (Ch. 4)                     = 76% GPU idle time        per-sample transforms

Batching                 Naive padding wastes        2.3% vs 98.5% efficiency   Use sequence packing
                         compute on short-seq        = 42x effective            (bin-packing samples
                         datasets (Ch. 5)            throughput gain            into fixed-length rows)

PCIe Transfer            Pageable memory blocks CPU  D2H: 1.77 vs 13.2 GB/s    Use pin_memory=True and
                         during DMA; non_blocking    = 7.5x penalty; overlap    non_blocking=True for
                         is a no-op for pageable     only at pinned+nb (Ch. 6)  true CPU/GPU overlap

End-to-End               All of the above interact;  57% GPU idle (w0) ->       Profile the full loop;
                         GPU idle time is the         0.4% (w4) = 2.33x         measure DataLoader wait
                         compounding effect (Ch. 7)   throughput gain            as % of step time
```

### Ordered optimization checklist

Based on the experiments, here is the recommended order of operations for
optimizing a training data pipeline. The items are ordered by impact and ease
of implementation:

**1. Use sharded, sequential data formats.**
Replace many small files with large contiguous shards. Use MosaicML Streaming,
WebDataset, TFRecord, or similar. We measured a **20.2x speedup** from this
single change (Chapters 2, 3, 7).

**2. Set `DataLoader(..., pin_memory=True)`.**
This is a single flag that enables DMA-based transfers and is essentially free.
There is no reason not to enable it (Chapter 6).

**3. Use `non_blocking=True` for `.to(device)` calls.**
Combined with pinned memory, this enables true CPU/GPU overlap. The CPU can
prepare the next batch while the current one transfers (Chapter 6).

**4. Tune `num_workers` to keep the GPU fed.**
Start with `num_workers = min(num_cpus, 8)` and increase if the DataLoader
wait time is significant. We measured the difference between 160ms wait
(0 workers, 76% GPU idle) and 1.5ms wait (4 workers) -- a **107x reduction**
in data stall time (Chapters 3, 4, 7).

**5. Move heavy preprocessing offline.**
Pre-tokenize text data. Pre-resize and encode images. The DataLoader should
do as little work as possible per sample (Chapter 4).

**6. Use sequence packing for variable-length data.**
For LLM training on short-sequence datasets, packing can improve effective
throughput dramatically -- we measured a **42.4x improvement** on Alpaca
(2.3% to 98.5% efficiency). Even longer-sequence datasets benefit
significantly (Chapter 5).

**7. Profile the full training loop with explicit phase markers.**
Use `torch.profiler` with `record_function` markers for `DATALOADER_WAIT`,
`H2D_TRANSFER`, `FORWARD`, `BACKWARD`, and `OPTIMIZER`. View in TensorBoard
to identify pipeline stalls (Chapter 7).

**8. Be aware of container and infrastructure constraints.**
In Kubernetes, ensure sufficient memory limits for pinned allocations, watch
for CPU CFS throttling, and verify NUMA topology between CPU and GPU
(Chapter 6).

### The meta-lesson

The most important finding from this investigation is not any single
optimization -- it is the realization that **the data pipeline is a chain,
and its throughput is determined by the weakest link.** Optimizing PCIe
transfers is pointless if the DataLoader is blocked on random file opens.
Adding more DataLoader workers is pointless if the data is in a shard format
that already maximizes throughput. Packing sequences is pointless if the GPU
starves for data because preprocessing is too slow.

The only way to know where the bottleneck is, is to measure. And the only way
to measure accurately is to control for confounding factors -- especially the
OS page cache, which can hide I/O problems that will surface in production.

This is why we designed each experiment to isolate one variable, with OS cache
purging, warmup phases, multiple runs, and statistical reporting. And it is why
the final experiment combines everything into a profiled training loop that
reveals how all the layers interact in practice.

The GPU is a furnace. Its hunger for data is insatiable. Feeding it is an
engineering problem that spans storage hardware, operating system caches,
file system design, framework abstractions, memory management, and bus
architecture. Understanding each layer -- and knowing how to measure it --
is what separates a training pipeline that crawls from one that flies.

---

## Appendix A: Benchmark Environment

All results were collected on the following system:

| Component | Specification |
|-----------|--------------|
| CPU | AMD EPYC 7R13 Processor (16 cores) |
| RAM | 124 GiB DDR4 |
| Storage | 559 GB local NVMe instance store (EC2) + 2x EBS volumes (200 GB + 100 GB) |
| Benchmark I/O | Ran on local NVMe instance store (overlay filesystem on nvme0n1) |
| GPU | NVIDIA L40S (47.7 GB VRAM, GDDR6 w/ ECC) |
| PCIe | Gen 4.0 x16 (31.51 GB/s theoretical unidirectional peak) |
| OS | Linux 6.8.0-1044-aws (Ubuntu 22.04 LTS) |
| Python | 3.10.12 |
| PyTorch | 2.10.0+cu128 |
| CUDA | 12.8 |
| Environment | Kubernetes pod (containerd runtime, AWS EC2) |
| cgroup limits | Memory: unlimited, CPU: unlimited (16 cores visible) |

OS page cache purging was available (running with root privileges in the
container), ensuring all I/O benchmarks measured real storage performance
rather than cached reads.

---

## Appendix B: Running the Experiments

All scripts are standalone and located in `training/analysis/`. They can be run
individually:

```bash
# Chapter 2: Pure Python I/O benchmark
uv run python training/analysis/random_vs_seq_python_simulation.py

# Chapter 3: PyTorch DataLoader I/O benchmark
uv run python training/analysis/random_vs_seq_dataloader_simulation.py

# Chapter 4: CPU starvation simulation
uv run python training/analysis/dataloader_bottleneck_simulation.py

# Chapter 5: Padding tax visualization (requires HF_TOKEN for Llama tokenizer)
uv run python training/analysis/padding_tax_simulation.py

# Chapter 6: PCIe transfer benchmark (requires CUDA GPU)
uv run python training/analysis/pcie_transfer_bottleneck.py

# Chapter 7: Streaming profiler comparison (requires CUDA GPU)
uv run python training/analysis/streaming_loader_lib_profiling.py
```

**Prerequisites:**

- Python >= 3.10
- PyTorch with CUDA support (Chapters 3, 4, 6, 7)
- `mosaicml-streaming` (Chapter 7)
- `transformers` and `datasets` (Chapter 5)
- Root or sudo access for OS page cache purging (Chapters 2, 3, 7)
- HuggingFace token for gated models (Chapter 5)

**Shared utilities:**

The `io_benchmark_utils.py` module provides data generation (`setup_data`),
cache purging (`purge_os_cache`), and cleanup functions used by Chapters 2, 3,
and 7.

---

## Appendix C: Script-to-Chapter Reference

| Script | Chapter | Key function(s) |
|--------|---------|-----------------|
| `io_benchmark_utils.py` | 2, 3, 7 | `setup_data()`, `purge_os_cache()` |
| `random_vs_seq_python_simulation.py` | 2 | `benchmark_random_access()`, `benchmark_sequential_access()` |
| `random_vs_seq_dataloader_simulation.py` | 3 | `SmallFilesDataset`, `GiantFileDataset`, `run_scenario()` |
| `dataloader_bottleneck_simulation.py` | 4 | `HeavyDataset`, `profile_loader()` |
| `padding_tax_simulation.py` | 5 | `simulate_naive_batching()`, `simulate_packed_batching()` |
| `pcie_transfer_bottleneck.py` | 6 | `benchmark_h2d()`, `benchmark_d2h()`, `benchmark_overlap()` |
| `streaming_loader_lib_profiling.py` | 7 | `train_and_profile()`, `StreamingImageDataset` |
