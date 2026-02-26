"""HOL Blocking Benchmark: Naive Prefill vs. Chunked Prefill

Demonstrates how a massive prefill ("Elephant", 8 192 input tokens) stalls
concurrent decode streams ("Mice", 20 × 64-token inputs / 128-token outputs)
and how vLLM's enable_chunked_prefill=True eliminates the resulting "UI freeze"
by interleaving prefill chunks with ongoing decode steps.

Usage
-----
  # Naive (no chunked prefill):
  python prefill_blocking_bench.py --no-chunked-prefill

  # Chunked prefill (512-token chunks):
  python prefill_blocking_bench.py --chunked-prefill

  # Print comparison summary from saved CSVs:
  python prefill_blocking_bench.py --summary

Output
------
  results/chunked.csv  (or naive.csv)
  Columns: request_id, token_index, timestamp_ms, latency_ms
"""
import argparse
import asyncio
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

NUM_MICE = 20
MICE_INPUT_TOKENS = 64
MICE_OUTPUT_TOKENS = 128

ELEPHANT_INPUT_TOKENS = 8_192
ELEPHANT_OUTPUT_TOKENS = 10
ELEPHANT_DELAY_S = 2.0  # seconds after mice start before elephant is injected

# Chunk size for Run B.  512 tokens keeps the L40S Tensor Cores busy while
# still yielding to ongoing decode requests every ~50 ms.
CHUNK_SIZE = 512


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
# Core async benchmark
# ---------------------------------------------------------------------------

async def run_benchmark(engine, tokenizer) -> list[dict]:
    """Launch 20 mice + 1 delayed elephant; stream every token; return records."""
    # Import here so the module is importable without vllm in the summary path.
    from vllm import SamplingParams  # noqa: PLC0415

    records: list[dict] = []

    mice_prompts = [
        make_token_prompt(tokenizer, MICE_INPUT_TOKENS) for _ in range(NUM_MICE)
    ]
    elephant_prompt = make_token_prompt(tokenizer, ELEPHANT_INPUT_TOKENS)

    async def track_mouse(req_id: str, prompt_ids: list[int]) -> None:
        """Stream a single mouse request; record arrival time of every token."""
        params = SamplingParams(max_tokens=MICE_OUTPUT_TOKENS, ignore_eos=True)
        prev_t = time.perf_counter()
        async for out in engine.generate(
            {"prompt_token_ids": prompt_ids}, params, request_id=req_id
        ):
            now = time.perf_counter()
            # outputs[0].token_ids is the cumulative list; subtract 1 for 0-based index.
            token_idx = len(out.outputs[0].token_ids) - 1
            records.append(
                {
                    "request_id": req_id,
                    "token_index": token_idx,
                    "timestamp_ms": now * 1_000.0,
                    "latency_ms": (now - prev_t) * 1_000.0,
                }
            )
            prev_t = now

    async def inject_elephant(delay: float, prompt_ids: list[int]) -> None:
        """Wait `delay` seconds then submit the massive prefill request."""
        await asyncio.sleep(delay)
        print(
            f"  [elephant] injecting {ELEPHANT_INPUT_TOKENS}-token prefill "
            f"request at t={delay:.1f}s"
        )
        params = SamplingParams(max_tokens=ELEPHANT_OUTPUT_TOKENS, ignore_eos=True)
        async for _ in engine.generate(
            {"prompt_token_ids": prompt_ids}, params, request_id="elephant"
        ):
            pass
        print("  [elephant] done")

    tasks = [
        asyncio.create_task(track_mouse(f"mouse_{i}", mice_prompts[i]))
        for i in range(NUM_MICE)
    ]
    tasks.append(asyncio.create_task(inject_elephant(ELEPHANT_DELAY_S, elephant_prompt)))

    print(f"  Launched {NUM_MICE} mice + 1 elephant (elephant delayed {ELEPHANT_DELAY_S}s)")
    await asyncio.gather(*tasks)
    print(f"  All requests complete.  Recorded {len(records)} token events.")
    return records


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(df: pd.DataFrame, run_label: str) -> dict:
    """Derive P99 ITL, jitter (StdDev ITL), and max stall from a records DataFrame.

    token_index == 0  →  TTFT (excluded from ITL stats)
    token_index >= 1  →  Inter-Token Latency (decode step time)
    """
    mice = df[df["request_id"] != "elephant"]
    itl = mice[mice["token_index"] >= 1]["latency_ms"].values

    if len(itl) == 0:
        return {
            "run": run_label,
            "p99_itl_ms": float("nan"),
            "jitter_ms": float("nan"),
            "max_stall_ms": float("nan"),
            "n_itl_samples": 0,
        }

    return {
        "run": run_label,
        "p99_itl_ms": float(np.percentile(itl, 99)),
        "jitter_ms": float(np.std(itl)),
        "max_stall_ms": float(np.max(itl)),
        "n_itl_samples": int(len(itl)),
    }


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(output_dir: Path) -> None:
    """Load run_a.csv and run_b.csv, compute metrics, and print a comparison table."""
    rows = []
    for label, path in [
        ("naive", output_dir / "naive.csv"),
        ("chunked", output_dir / "chunked.csv"),
    ]:
        if not path.exists():
            print(f"  WARNING: {path} not found, skipping.")
            continue
        df = pd.read_csv(path)
        rows.append(compute_metrics(df, label))

    if not rows:
        print("No result files found in", output_dir)
        return

    summary = pd.DataFrame(rows).set_index("run")
    cols = ["p99_itl_ms", "jitter_ms", "max_stall_ms", "n_itl_samples"]
    print("\n" + "=" * 70)
    print("HOL BLOCKING BENCHMARK — SUMMARY")
    print("=" * 70)
    print(summary[cols].round(2).to_string())
    print("=" * 70)

    if len(rows) == 2:
        stall_a, stall_b = rows[0]["max_stall_ms"], rows[1]["max_stall_ms"]
        p99_a, p99_b = rows[0]["p99_itl_ms"], rows[1]["p99_itl_ms"]

        if stall_a > 0 and stall_b > 0:
            print(
                f"\nMax stall:  {stall_a:.1f} ms → {stall_b:.1f} ms  "
                f"({(stall_a - stall_b) / stall_a * 100:.1f}% reduction)"
            )
        if p99_a > 0 and p99_b > 0:
            print(
                f"P99 ITL:    {p99_a:.1f} ms → {p99_b:.1f} ms  "
                f"({(p99_a - p99_b) / p99_a * 100:.1f}% reduction)"
            )


# ---------------------------------------------------------------------------
# Async entry point for a single run (A or B)
# ---------------------------------------------------------------------------

async def main_run(args: argparse.Namespace) -> None:
    # Deferred imports so --summary works without vllm installed.
    from transformers import AutoTokenizer  # noqa: PLC0415
    from vllm import AsyncEngineArgs, AsyncLLMEngine  # noqa: PLC0415

    chunked = args.enable_chunked_prefill

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / ("chunked.csv" if chunked else "naive.csv")

    header = (
        "enable_chunked_prefill=True, "
        f"max_num_batched_tokens={CHUNK_SIZE}" if chunked
        else "enable_chunked_prefill=False (naive)"
    )
    print(f"\n{'=' * 70}\n{header}\n{'=' * 70}")

    engine_kwargs: dict = dict(
        model=args.model,
        tensor_parallel_size=1,
        dtype="half",
        gpu_memory_utilization=0.90,
        max_model_len=16_384,  # benchmark max is 8192 input tokens; cap avoids OOM on <80 GiB GPUs
        enforce_eager=False,
        enable_prefix_caching=False,
        enable_chunked_prefill=chunked,
    )
    if chunked:
        engine_kwargs["max_num_batched_tokens"] = CHUNK_SIZE

    print(f"Initializing AsyncLLMEngine ({args.model})...")
    engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(**engine_kwargs))

    print("Warming up (3 s)...")
    await asyncio.sleep(3)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    t_start = time.perf_counter()
    records = await run_benchmark(engine, tokenizer)
    elapsed = time.perf_counter() - t_start

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df)} rows → {output_path}  (wall time: {elapsed:.1f}s)")

    run_label = "chunked" if chunked else "naive"
    m = compute_metrics(df, run_label)
    print(f"  P99 ITL:    {m['p99_itl_ms']:.2f} ms")
    print(f"  Jitter:     {m['jitter_ms']:.2f} ms  (StdDev ITL)")
    print(f"  Max Stall:  {m['max_stall_ms']:.2f} ms  (n={m['n_itl_samples']} samples)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="HOL Blocking Benchmark — Prefill vs. Chunked Prefill",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--chunked-prefill",
        action=argparse.BooleanOptionalAction,
        default=None,
        dest="enable_chunked_prefill",
        help=f"Enable chunked prefill (max_num_batched_tokens={CHUNK_SIZE}).",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print comparison table from existing naive.csv / chunked.csv (no inference).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"HuggingFace model ID (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory for CSV output / input (default: results/)",
    )

    args = parser.parse_args()

    if not args.summary and args.enable_chunked_prefill is None:
        parser.error("Specify --chunked-prefill, --no-chunked-prefill, or --summary.")

    if args.summary:
        print_summary(Path(args.output_dir))
    else:
        asyncio.run(main_run(args))


if __name__ == "__main__":
    main()
