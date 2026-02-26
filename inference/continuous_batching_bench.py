"""Continuous Batching vs. Static Batching Benchmark

Quantifies the "Effective Throughput" gain of Iteration-level Scheduling
(Continuous Batching) by measuring the Tensor Occupancy Padding Waste
inherent in Static Batching.

In static batching, every request in a batch must stay in flight until the
longest "straggler" finishes. This forces the GPU to perform matrix
multiplications on Tensor Padding for the slots that finished early.

vLLM's continuous batching allows finished requests to exit and new ones
to enter at every iteration, maximizing functional GPU occupancy.

Workload:
  - 100 requests total
  - Input Length: Fixed at 128 tokens
  - Output Lengths: 50% "Short" (32 tokens) + 50% "Long" (512 tokens)
  - Requests are shuffled so Short and Long are randomly mixed

Run A (Static Batching Simulation):
  - Process 100 requests in fixed batches of size 16
  - For each batch, identify max_output_len and force every request in that
    batch to generate exactly max_output_len tokens (using min_tokens, max_tokens,
    and ignore_eos=True)
  - This simulates the synchronous waste of a static batching system
  - Measure wall-clock time per batch

Run B (Continuous Batching):
  - Submit all 100 requests simultaneously
  - Let vLLM handle the scheduling naturally (ignore_eos=False)
  - Requests exit as soon as they finish; new ones fill vacated slots

Output:
  - Save summary to results/continuous_batching_summary.json
  - Include: total_time, useful_tokens, useful_tps for both runs
  - Note: Padding tokens in Run A are NOT counted as "useful"

Usage
-----
  # Run both passes and save summary:
  python continuous_batching_bench.py

  # Run only static batching:
  python continuous_batching_bench.py --static-only

  # Run only continuous batching:
  python continuous_batching_bench.py --continuous-only

  # Print summary from saved JSON:
  python continuous_batching_bench.py --summary
"""
import argparse
import asyncio
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

NUM_REQUESTS = 100
INPUT_LENGTH = 128
SHORT_OUTPUT_LENGTH = 32
LONG_OUTPUT_LENGTH = 512
BATCH_SIZE = 16

# Seed for reproducibility
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RequestConfig:
    """Configuration for a single request."""
    request_id: str
    input_length: int
    output_length: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RunResult:
    """Result from a single run (A or B)."""
    run_name: str
    total_time_s: float
    total_useful_tokens: int
    useful_tps: float
    num_requests: int
    avg_output_length: float


# ---------------------------------------------------------------------------
# Workload generation
# ---------------------------------------------------------------------------

def generate_workload(num_requests: int = NUM_REQUESTS,
                      input_length: int = INPUT_LENGTH,
                      short_len: int = SHORT_OUTPUT_LENGTH,
                      long_len: int = LONG_OUTPUT_LENGTH,
                      seed: int = RANDOM_SEED) -> list[RequestConfig]:
    """Generate a pool of requests with mixed output lengths.

    50% short output, 50% long output, shuffled randomly.
    """
    random.seed(seed)

    requests = []
    # First half: short outputs
    for i in range(num_requests // 2):
        requests.append(
            RequestConfig(
                request_id=f"req_{i:03d}_short",
                input_length=input_length,
                output_length=short_len,
            )
        )
    # Second half: long outputs
    for i in range(num_requests // 2, num_requests):
        requests.append(
            RequestConfig(
                request_id=f"req_{i:03d}_long",
                input_length=input_length,
                output_length=long_len,
            )
        )

    # Shuffle
    random.shuffle(requests)
    return requests


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

def make_token_prompt(tokenizer, n_tokens: int) -> list[int]:
    """Return a list of exactly n_tokens token IDs using a repeated filler word."""
    base = tokenizer.encode(" hello", add_special_tokens=False)
    if not base:
        base = [tokenizer.unk_token_id or 0]
    return (base * (n_tokens // len(base) + 1))[:n_tokens]


# ---------------------------------------------------------------------------
# Run A: Static Batching Simulation
# ---------------------------------------------------------------------------

async def run_static_batching(engine, tokenizer, requests: list[RequestConfig]) -> RunResult:
    """Process requests in fixed batches; force each batch to wait for max straggler.

    For each batch, all requests generate exactly max(output_length) tokens.
    """
    from vllm import SamplingParams  # noqa: PLC0415

    print("\n" + "=" * 70)
    print("RUN A: Static Batching (Synchronous Waste Simulation)")
    print("=" * 70)
    print(f"Processing {len(requests)} requests in batches of {BATCH_SIZE}...")

    total_time = 0.0
    total_useful_tokens = 0
    num_batches = (len(requests) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, len(requests))
        batch_requests = requests[start_idx:end_idx]

        # Identify the max output length in this batch (the "straggler")
        max_output_len = max(r.output_length for r in batch_requests)

        print(f"  Batch {batch_idx + 1}/{num_batches}: {len(batch_requests)} requests, "
              f"max_output_len={max_output_len}")

        # Create prompts and parameters
        prompts = [make_token_prompt(tokenizer, r.input_length) for r in batch_requests]

        # Force all requests to generate exactly max_output_len tokens
        params = SamplingParams(
            min_tokens=max_output_len,
            max_tokens=max_output_len,
            ignore_eos=True,
        )

        # Track timing for this batch
        batch_start = time.perf_counter()

        # Generate in parallel
        tasks = []
        for i, prompt in enumerate(prompts):
            req_id = batch_requests[i].request_id
            async def generate_batch_request(pid: str, p: list[int]) -> tuple[str, int]:
                """Generate and return request ID and output length."""
                async for out in engine.generate(
                    {"prompt_token_ids": p}, params, request_id=pid
                ):
                    pass
                return pid, max_output_len

            tasks.append(generate_batch_request(req_id, prompt))

        # Wait for all requests in batch to complete
        await asyncio.gather(*tasks)

        batch_elapsed = time.perf_counter() - batch_start

        # Count useful tokens (actual output from each request)
        batch_useful_tokens = sum(r.output_length for r in batch_requests)
        total_useful_tokens += batch_useful_tokens

        # Count padding waste (forced extra tokens due to max straggler)
        batch_padding_tokens = sum(
            max_output_len - r.output_length for r in batch_requests
        )

        total_time += batch_elapsed

        print(f"    → {batch_elapsed:.2f}s | "
              f"useful: {batch_useful_tokens} | "
              f"padding waste: {batch_padding_tokens}")

    useful_tps = total_useful_tokens / total_time if total_time > 0 else 0.0

    print(f"\nRun A Summary:")
    print(f"  Total wall-clock time: {total_time:.2f}s")
    print(f"  Total useful tokens:  {total_useful_tokens}")
    print(f"  Useful TPS:           {useful_tps:.2f}")

    return RunResult(
        run_name="static_batching",
        total_time_s=total_time,
        total_useful_tokens=total_useful_tokens,
        useful_tps=useful_tps,
        num_requests=len(requests),
        avg_output_length=np.mean([r.output_length for r in requests]),
    )


# ---------------------------------------------------------------------------
# Run B: Continuous Batching
# ---------------------------------------------------------------------------

async def run_continuous_batching(engine, tokenizer, requests: list[RequestConfig]) -> RunResult:
    """Submit all requests simultaneously; let vLLM schedule naturally."""
    from vllm import SamplingParams  # noqa: PLC0415

    print("\n" + "=" * 70)
    print("RUN B: Continuous Batching (vLLM Standard)")
    print("=" * 70)
    print(f"Submitting {len(requests)} requests simultaneously...")

    # Track timing
    total_time_start = time.perf_counter()
    total_useful_tokens = 0

    # Generate all requests in parallel, each with its own params
    async def generate_request(req_id: str, prompt: list[int], output_len: int) -> int:
        """Generate a single request and return the actual output length."""
        # Each request gets its exact output length (min=max=output_len)
        params = SamplingParams(
            min_tokens=output_len,
            max_tokens=output_len,
            ignore_eos=True,
        )
        async for out in engine.generate(
            {"prompt_token_ids": prompt}, params, request_id=req_id
        ):
            pass
        return output_len

    # Create all prompts
    prompts = [make_token_prompt(tokenizer, r.input_length) for r in requests]
    
    tasks = [
        generate_request(requests[i].request_id, prompts[i], requests[i].output_length)
        for i in range(len(requests))
    ]

    # Wait for all to complete
    actual_outputs = await asyncio.gather(*tasks)

    total_time = time.perf_counter() - total_time_start
    total_useful_tokens = sum(actual_outputs)

    useful_tps = total_useful_tokens / total_time if total_time > 0 else 0.0

    print(f"\nRun B Summary:")
    print(f"  Total wall-clock time: {total_time:.2f}s")
    print(f"  Total useful tokens:  {total_useful_tokens}")
    print(f"  Useful TPS:           {useful_tps:.2f}")

    return RunResult(
        run_name="continuous_batching",
        total_time_s=total_time,
        total_useful_tokens=total_useful_tokens,
        useful_tps=useful_tps,
        num_requests=len(requests),
        avg_output_length=np.mean([r.output_length for r in requests]),
    )


# ---------------------------------------------------------------------------
# Summary and analysis
# ---------------------------------------------------------------------------

def compute_summary(result_a: RunResult, result_b: RunResult) -> dict:
    """Compute summary metrics comparing the two runs."""
    throughput_gain = (result_b.useful_tps / result_a.useful_tps - 1.0) * 100 if result_a.useful_tps > 0 else 0.0
    time_reduction = (result_a.total_time_s / result_b.total_time_s - 1.0) * 100 if result_b.total_time_s > 0 else 0.0

    return {
        "workload": {
            "num_requests": result_a.num_requests,
            "input_length": INPUT_LENGTH,
            "short_output_length": SHORT_OUTPUT_LENGTH,
            "long_output_length": LONG_OUTPUT_LENGTH,
            "batch_size_static": BATCH_SIZE,
            "avg_output_length": float(result_a.avg_output_length),
        },
        "run_a_static_batching": {
            "total_time_s": result_a.total_time_s,
            "useful_tokens": result_a.total_useful_tokens,
            "useful_tps": result_a.useful_tps,
        },
        "run_b_continuous_batching": {
            "total_time_s": result_b.total_time_s,
            "useful_tokens": result_b.total_useful_tokens,
            "useful_tps": result_b.useful_tps,
        },
        "improvement": {
            "throughput_gain_percent": throughput_gain,
            "time_reduction_percent": time_reduction,
            "speedup_factor": result_b.useful_tps / result_a.useful_tps if result_a.useful_tps > 0 else 0.0,
        },
    }


def print_summary(result_a: RunResult, result_b: RunResult) -> None:
    """Print a formatted comparison table."""
    summary = compute_summary(result_a, result_b)

    print("\n" + "=" * 70)
    print("CONTINUOUS BATCHING vs. STATIC BATCHING — SUMMARY")
    print("=" * 70)

    print("\nWorkload Configuration:")
    print(f"  Number of requests:      {summary['workload']['num_requests']}")
    print(f"  Input length:            {summary['workload']['input_length']} tokens")
    print(f"  Short output length:     {summary['workload']['short_output_length']} tokens (50%)")
    print(f"  Long output length:      {summary['workload']['long_output_length']} tokens (50%)")
    print(f"  Static batch size:       {summary['workload']['batch_size_static']}")
    print(f"  Average output length:   {summary['workload']['avg_output_length']:.1f} tokens")

    print("\nRun A — Static Batching (Synchronous Waste):")
    print(f"  Total wall-clock time:   {summary['run_a_static_batching']['total_time_s']:.2f}s")
    print(f"  Total useful tokens:     {summary['run_a_static_batching']['useful_tokens']}")
    print(f"  Useful Throughput:       {summary['run_a_static_batching']['useful_tps']:.2f} tokens/sec")

    print("\nRun B — Continuous Batching (vLLM Standard):")
    print(f"  Total wall-clock time:   {summary['run_b_continuous_batching']['total_time_s']:.2f}s")
    print(f"  Total useful tokens:     {summary['run_b_continuous_batching']['useful_tokens']}")
    print(f"  Useful Throughput:       {summary['run_b_continuous_batching']['useful_tps']:.2f} tokens/sec")

    print("\nImprovement (Continuous vs. Static):")
    improvement = summary['improvement']
    print(f"  Throughput gain:         {improvement['throughput_gain_percent']:.1f}%")
    print(f"  Speedup factor:          {improvement['speedup_factor']:.2f}x")
    print(f"  Wall-clock time saved:   {improvement['time_reduction_percent']:.1f}%")
    print("=" * 70)


def save_summary(output_dir: Path, result_a: RunResult, result_b: RunResult) -> None:
    """Save summary to JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "continuous_batching_summary.json"

    summary = compute_summary(result_a, result_b)
    summary["metadata"] = {
        "model": DEFAULT_MODEL,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to: {output_path}")


def load_and_print_summary(output_dir: Path) -> None:
    """Load and print summary from previously saved JSON."""
    output_path = output_dir / "continuous_batching_summary.json"

    if not output_path.exists():
        print(f"Summary file not found: {output_path}")
        return

    with open(output_path) as f:
        summary = json.load(f)

    print("\n" + "=" * 70)
    print("CONTINUOUS BATCHING vs. STATIC BATCHING — SUMMARY")
    print("=" * 70)

    print("\nWorkload Configuration:")
    wl = summary["workload"]
    print(f"  Number of requests:      {wl['num_requests']}")
    print(f"  Input length:            {wl['input_length']} tokens")
    print(f"  Short output length:     {wl['short_output_length']} tokens (50%)")
    print(f"  Long output length:      {wl['long_output_length']} tokens (50%)")
    print(f"  Static batch size:       {wl['batch_size_static']}")
    print(f"  Average output length:   {wl['avg_output_length']:.1f} tokens")

    print("\nRun A — Static Batching (Synchronous Waste):")
    ra = summary["run_a_static_batching"]
    print(f"  Total wall-clock time:   {ra['total_time_s']:.2f}s")
    print(f"  Total useful tokens:     {ra['useful_tokens']}")
    print(f"  Useful Throughput:       {ra['useful_tps']:.2f} tokens/sec")

    print("\nRun B — Continuous Batching (vLLM Standard):")
    rb = summary["run_b_continuous_batching"]
    print(f"  Total wall-clock time:   {rb['total_time_s']:.2f}s")
    print(f"  Total useful tokens:     {rb['useful_tokens']}")
    print(f"  Useful Throughput:       {rb['useful_tps']:.2f} tokens/sec")

    print("\nImprovement (Continuous vs. Static):")
    imp = summary["improvement"]
    print(f"  Throughput gain:         {imp['throughput_gain_percent']:.1f}%")
    print(f"  Speedup factor:          {imp['speedup_factor']:.2f}x")
    print(f"  Wall-clock time saved:   {imp['time_reduction_percent']:.1f}%")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Async entry point
# ---------------------------------------------------------------------------

async def main_benchmark(args: argparse.Namespace) -> None:
    """Run the full benchmark (A and/or B)."""
    from transformers import AutoTokenizer  # noqa: PLC0415
    from vllm import AsyncEngineArgs, AsyncLLMEngine  # noqa: PLC0415

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate workload once
    print(f"\nGenerating workload: {NUM_REQUESTS} requests "
          f"(50% {SHORT_OUTPUT_LENGTH}-token, 50% {LONG_OUTPUT_LENGTH}-token)...")
    requests = generate_workload()

    # Initialize engine
    print(f"\nInitializing AsyncLLMEngine ({args.model})...")
    engine_kwargs = dict(
        model=args.model,
        tensor_parallel_size=1,
        dtype="half",
        gpu_memory_utilization=0.90,
        max_model_len=16_384,
        enforce_eager=False,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,  # We're measuring continuous batching, not chunking
    )
    engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(**engine_kwargs))

    print("Warming up (3 s)...")
    await asyncio.sleep(3)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Determine which runs to execute
    run_both = not args.static_only and not args.continuous_only
    
    # Run A: Static Batching
    result_a = None
    if args.static_only or run_both:
        result_a = await run_static_batching(engine, tokenizer, requests)

    # Run B: Continuous Batching
    result_b = None
    if args.continuous_only or run_both:
        result_b = await run_continuous_batching(engine, tokenizer, requests)

    # Summary
    if result_a and result_b:
        print_summary(result_a, result_b)
        save_summary(output_dir, result_a, result_b)
    elif result_a:
        print("\n(Run A only — no summary comparison)")
    elif result_b:
        print("\n(Run B only — no summary comparison)")
    
    # Always print final message
    if result_a and result_b:
        print("\n" + "=" * 70)
        print("Benchmark complete! Results saved to:")
        print(f"  {output_dir / 'continuous_batching_summary.json'}")
        print("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Continuous Batching vs. Static Batching Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--static-only",
        action="store_true",
        help="Run only the static batching simulation (Run A).",
    )
    parser.add_argument(
        "--continuous-only",
        action="store_true",
        help="Run only the continuous batching benchmark (Run B).",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print summary from existing results (no inference).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"HuggingFace model ID (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory for JSON output / input (default: results/)",
    )

    args = parser.parse_args()

    if args.summary:
        load_and_print_summary(Path(args.output_dir))
    else:
        asyncio.run(main_benchmark(args))


if __name__ == "__main__":
    main()
