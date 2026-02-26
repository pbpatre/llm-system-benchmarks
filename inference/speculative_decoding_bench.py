"""Speculative Decoding Benchmark: Standard vs. Speculative Inference

Quantifies the Tokens Per Second (TPS) speedup and Draft Acceptance Rate
of speculative decoding.  Two drafter strategies are supported:

  model  — a small neural draft model (requires a compatible HF model).
             For Llama-3.1-8B, compatible drafters share the Llama-3 tokenizer
             (128k vocab).  The smallest available options are:
               meta-llama/Llama-3.2-1B-Instruct  (gated, HF approval required)
               meta-llama/Llama-3.2-3B-Instruct  (gated, HF approval required)

  ngram  — prompt-lookup (n-gram) speculative decoding built into vLLM.
             No separate model is needed.  Candidate tokens are proposed by
             scanning for repeated n-gram patterns in the prompt+context.
             This is the default and works without any HF model permissions.

Target model:  meta-llama/Llama-3.1-8B-Instruct
Hardware:      NVIDIA L40S (48 GB)

Why ngram is a good stand-in for the predictability experiment
--------------------------------------------------------------
N-gram hits are entirely driven by how repetitive the output text is:
  Code (Task 1):     boilerplate is full of repeated tokens → high hit rate
  Creative (Task 2): novel output has no prior n-gram matches → near-zero hits
This makes the speedup gap between tasks even more pronounced than with a
neural drafter, cleanly proving the predictability thesis.

Workload:
  Task 1 (Predictable):   Python code generation — boilerplate functions
  Task 2 (Unpredictable): Creative writing / complex riddles

Run A (Baseline):
  Standard 8B inference — no speculative decoding.
  Saves: results/speculative_baseline.json

Run B (Speculative):
  8B + ngram drafter  OR  8B + neural draft model.
  Saves: results/speculative_spec.json

Each run is a separate process invocation so the GPU is fully freed between
runs.  The shell script orchestrates both passes and then calls --summary.

Metrics:
  - TPS: Useful output tokens per second
  - Mean Accepted Tokens: Out of k drafted tokens, how many verified on avg?
  - Relative Speedup Factor: TPS(speculative) / TPS(baseline)

Usage
-----
  # Ngram drafter (default, no model permissions needed):
  python speculative_decoding_bench.py --baseline-only
  python speculative_decoding_bench.py --speculative-only --draft-method ngram
  python speculative_decoding_bench.py --summary

  # Neural draft model (once HF access is granted):
  python speculative_decoding_bench.py --speculative-only --draft-method model \\
      --draft-model meta-llama/Llama-3.2-1B-Instruct

  # Typically invoked by run_speculative_decoding_bench.sh which runs all three.
"""
import argparse
import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_TARGET_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_DRAFT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
DEFAULT_DRAFT_METHOD = "model"   # "model" | "ngram"
NUM_SPECULATIVE_TOKENS = 5
NGRAM_PROMPT_LOOKUP_MAX = 4      # max n-gram window size for prompt-lookup

GPU_MEMORY_UTILIZATION = 0.80
MAX_OUTPUT_TOKENS = 256
NUM_REQUESTS_PER_TASK = 10

# ---------------------------------------------------------------------------
# Prompts — two categories of predictability
# ---------------------------------------------------------------------------
PREDICTABLE_PROMPTS = [
    "Write a Python function that reads a CSV file and returns a list of dictionaries.",
    "Write a Python class called DatabaseConnection with connect, disconnect, and execute_query methods.",
    "Write a Python function to validate an email address using a regular expression.",
    "Write a Python decorator that logs the execution time of a function.",
    "Write a Python function that implements binary search on a sorted list.",
    "Write a Python context manager for handling temporary files.",
    "Write a Python function that converts a nested dictionary to a flat dictionary with dot-separated keys.",
    "Write a Python class implementing a simple LRU cache with get and put methods.",
    "Write a Python function that reads a JSON config file and returns the parsed configuration with default values.",
    "Write a Python function that implements the merge sort algorithm.",
]

UNPREDICTABLE_PROMPTS = [
    "Write a surreal short story where gravity reverses every Tuesday and the protagonist is a sentient umbrella.",
    "Compose an avant-garde poem about quantum entanglement from the perspective of a photon experiencing existential dread.",
    "A snail can climb 3 feet up a wall during the day but slips back 2 feet at night. If the wall is 30 feet high, on which day does the snail reach the top? Explain step by step.",
    "Write a philosophical dialogue between a time-traveling medieval knight and an AI about the meaning of consciousness.",
    "Invent a new board game with novel mechanics never seen before. Describe the rules, pieces, and winning conditions.",
    "Three boxes are labeled Apples, Oranges, and Mixed. All labels are wrong. You can pick one fruit from one box. Which box do you pick from and why?",
    "Write a story in which each paragraph contradicts the previous one, yet the overall narrative still makes sense.",
    "Describe a color that doesn't exist and explain how it would change art, fashion, and human emotion.",
    "Compose a limerick about Godel's incompleteness theorem that also serves as an informal proof of the theorem.",
    "Write a conversation between two AIs debating whether mathematics is discovered or invented, using only cooking metaphors.",
]


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass
class TaskResult:
    """Result from running one task category under one configuration."""
    task_name: str
    run_name: str
    total_time_s: float
    total_output_tokens: int
    tps: float
    num_requests: int
    per_request_tokens: list[int]
    per_request_times: list[float]


# ---------------------------------------------------------------------------
# speculative_config builder
# ---------------------------------------------------------------------------

def build_speculative_cfg(draft_method: str, draft_model: str) -> dict:
    """Return the vLLM speculative_config dict for the chosen draft strategy.

    vLLM 0.16 uses a single ``speculative_config: dict`` on AsyncEngineArgs
    that maps directly to SpeculativeConfig fields.

    Valid ``method`` literals (from SpeculativeMethod):
      "draft_model" — small neural draft model (same tokenizer as target)
      "ngram"       — prompt-lookup n-gram, no extra model required

    Important: the ``method`` key must be set explicitly.  Omitting it causes
    SpeculativeConfig.__post_init__ to raise NotImplementedError even when
    ``model`` is provided, because auto-detection requires target_model_config
    to already be resolved — which hasn't happened yet at this call site.
    """
    if draft_method == "ngram":
        return {
            "method": "ngram",
            "num_speculative_tokens": NUM_SPECULATIVE_TOKENS,
            "prompt_lookup_max": NGRAM_PROMPT_LOOKUP_MAX,
        }
    if draft_method == "model":
        return {
            "method": "draft_model",
            "model": draft_model,
            "num_speculative_tokens": NUM_SPECULATIVE_TOKENS,
        }
    raise ValueError(f"Unknown draft_method: {draft_method!r}. Choose 'ngram' or 'model'.")


# ---------------------------------------------------------------------------
# Core benchmark runner
# ---------------------------------------------------------------------------

async def run_task(
    engine,
    tokenizer,
    prompts: list[str],
    task_name: str,
    run_name: str,
    max_tokens: int = MAX_OUTPUT_TOKENS,
) -> TaskResult:
    """Run a set of prompts through the engine sequentially and measure TPS."""
    from vllm import SamplingParams  # noqa: PLC0415

    print(f"\n  [{run_name}] {task_name}: {len(prompts)} prompts, max_tokens={max_tokens}")

    params = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.7,
        top_p=0.95,
    )

    per_request_tokens: list[int] = []
    per_request_times: list[float] = []

    total_start = time.perf_counter()

    for i, prompt in enumerate(prompts):
        req_id = f"{run_name}_{task_name}_{i:03d}"

        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

        req_start = time.perf_counter()
        output_len = 0
        async for out in engine.generate(formatted, params, request_id=req_id):
            output_len = len(out.outputs[0].token_ids)
        req_elapsed = time.perf_counter() - req_start

        per_request_tokens.append(output_len)
        per_request_times.append(req_elapsed)

        req_tps = output_len / req_elapsed if req_elapsed > 0 else 0.0
        print(f"    request {i + 1}/{len(prompts)}: "
              f"{output_len} tokens in {req_elapsed:.2f}s ({req_tps:.1f} tok/s)")

    total_elapsed = time.perf_counter() - total_start
    total_tokens = sum(per_request_tokens)
    tps = total_tokens / total_elapsed if total_elapsed > 0 else 0.0

    print(f"  [{run_name}] {task_name} done: "
          f"{total_tokens} tokens in {total_elapsed:.2f}s → {tps:.2f} tok/s")

    return TaskResult(
        task_name=task_name,
        run_name=run_name,
        total_time_s=total_elapsed,
        total_output_tokens=total_tokens,
        tps=tps,
        num_requests=len(prompts),
        per_request_tokens=per_request_tokens,
        per_request_times=per_request_times,
    )


# ---------------------------------------------------------------------------
# Engine initialization
# ---------------------------------------------------------------------------

async def create_engine(model: str, speculative_cfg: dict | None = None):
    """Create an AsyncLLMEngine with the benchmark config.

    vLLM 0.16 accepts speculative decoding settings via a single
    ``speculative_config`` dict.  When None, standard inference is used.
    """
    from vllm import AsyncEngineArgs, AsyncLLMEngine  # noqa: PLC0415

    engine_kwargs: dict = dict(
        model=model,
        tensor_parallel_size=1,
        dtype="half",
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_model_len=4096,
        enforce_eager=False,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
    )
    if speculative_cfg is not None:
        engine_kwargs["speculative_config"] = speculative_cfg

    engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(**engine_kwargs))
    print("Warming up (3 s)...")
    await asyncio.sleep(3)
    return engine


# ---------------------------------------------------------------------------
# Single-run entry point (one engine config, both task categories)
# ---------------------------------------------------------------------------

async def run_single(args: argparse.Namespace, run_name: str,
                     speculative_cfg: dict | None = None) -> list[TaskResult]:
    """Spin up one engine, run both tasks, shut down, return results."""
    from transformers import AutoTokenizer  # noqa: PLC0415

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    engine = await create_engine(args.model, speculative_cfg=speculative_cfg)

    results = []
    results.append(await run_task(
        engine, tokenizer, PREDICTABLE_PROMPTS[:NUM_REQUESTS_PER_TASK],
        "predictable_code", run_name,
    ))
    results.append(await run_task(
        engine, tokenizer, UNPREDICTABLE_PROMPTS[:NUM_REQUESTS_PER_TASK],
        "unpredictable_creative", run_name,
    ))

    engine.shutdown()
    return results


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def results_to_dict(results: list[TaskResult]) -> dict:
    return {
        r.task_name: {
            "run_name": r.run_name,
            "total_time_s": r.total_time_s,
            "total_output_tokens": r.total_output_tokens,
            "tps": r.tps,
            "num_requests": r.num_requests,
            "per_request_tokens": r.per_request_tokens,
            "per_request_times": r.per_request_times,
        }
        for r in results
    }


def save_run(output_dir: Path, filename: str, run_name: str, results: list[TaskResult],
             model: str, speculative_cfg: dict | None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_name": run_name,
        "config": {
            "target_model": model,
            "speculative_config": speculative_cfg,
            "num_speculative_tokens": NUM_SPECULATIVE_TOKENS,
            "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "num_requests_per_task": NUM_REQUESTS_PER_TASK,
        },
        "tasks": results_to_dict(results),
        "metadata": {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S")},
    }
    path = output_dir / filename
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nRun results saved to: {path}")


# ---------------------------------------------------------------------------
# Summary / comparison
# ---------------------------------------------------------------------------

def load_run(output_dir: Path, filename: str) -> dict | None:
    path = output_dir / filename
    if not path.exists():
        print(f"  WARNING: {path} not found.")
        return None
    with open(path) as f:
        return json.load(f)


def _drafter_label(speculative_cfg: dict | None) -> str:
    """Human-readable label for the speculative config."""
    if speculative_cfg is None:
        return "none"
    if speculative_cfg.get("method") == "ngram":
        return f"ngram (prompt_lookup_max={speculative_cfg.get('prompt_lookup_max', '?')})"
    if "model" in speculative_cfg:
        return speculative_cfg["model"]
    return str(speculative_cfg)


def print_comparison(baseline: dict, speculative: dict) -> None:
    """Print a formatted comparison table between the two run payloads."""
    print("\n" + "=" * 70)
    print("SPECULATIVE DECODING BENCHMARK — SUMMARY")
    print("=" * 70)

    cfg = baseline["config"]
    spec_cfg = speculative["config"].get("speculative_config")
    k = speculative["config"]["num_speculative_tokens"]

    print(f"\n  Target model:            {cfg['target_model']}")
    print(f"  Drafter:                 {_drafter_label(spec_cfg)}")
    print(f"  Speculative tokens (k):  {k}")
    print(f"  GPU memory utilization:  {cfg['gpu_memory_utilization']}")
    print(f"  Max output tokens:       {cfg['max_output_tokens']}")
    print(f"  Requests per task:       {cfg['num_requests_per_task']}")

    speedups: list[float] = []

    for task_name, b in baseline["tasks"].items():
        s = speculative["tasks"].get(task_name)
        if s is None:
            print(f"\n  WARNING: task '{task_name}' missing from speculative results, skipping.")
            continue

        label = "Predictable (Code)" if task_name == "predictable_code" else "Unpredictable (Creative)"
        b_tps = b["tps"]
        s_tps = s["tps"]
        speedup = s_tps / b_tps if b_tps > 0 else 0.0
        speedups.append(speedup)

        # Mean accepted token estimate:
        # Each speculative step produces (accepted + 1) tokens vs 1 for baseline.
        # If TPS ratio is X, mean tokens per step ≈ X → mean_accepted ≈ X - 1.
        mean_accepted = max(0.0, min(speedup - 1.0, float(k)))

        print(f"\n  {'─' * 60}")
        print(f"  Task: {label}")
        print(f"  {'─' * 60}")
        print(f"    Baseline TPS:          {b_tps:.2f} tok/s  ({b['total_time_s']:.2f}s, {b['total_output_tokens']} tokens)")
        print(f"    Speculative TPS:       {s_tps:.2f} tok/s  ({s['total_time_s']:.2f}s, {s['total_output_tokens']} tokens)")
        print(f"    Speedup Factor:        {speedup:.2f}x")
        print(f"    Mean Accepted (est):   {mean_accepted:.2f} / {k}")

    print("\n" + "=" * 70)

    if len(speedups) == 2:
        pred_sp, unpred_sp = speedups
        pred_acc = max(0.0, min(pred_sp - 1.0, float(k)))
        unpred_acc = max(0.0, min(unpred_sp - 1.0, float(k)))

        print(f"\n  Predictability Impact:")
        print(f"    Code (predictable):       {pred_sp:.2f}x speedup, ~{pred_acc:.1f} accepted/step")
        print(f"    Creative (unpredictable): {unpred_sp:.2f}x speedup, ~{unpred_acc:.1f} accepted/step")

        if pred_sp > unpred_sp and unpred_sp > 0:
            advantage = (pred_sp / unpred_sp - 1.0) * 100
            print(f"    → Predictable text gains {advantage:.1f}% more speedup from speculation")

    print("=" * 70)


def save_combined_summary(output_dir: Path, baseline: dict, speculative: dict) -> None:
    """Write a single combined JSON for downstream analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "speculative_decoding_summary.json"

    k = speculative["config"]["num_speculative_tokens"]
    tasks = {}
    for task_name, b in baseline["tasks"].items():
        s = speculative["tasks"].get(task_name, {})
        b_tps = b["tps"]
        s_tps = s.get("tps", 0.0)
        speedup = s_tps / b_tps if b_tps > 0 else 0.0
        tasks[task_name] = {
            "baseline_tps": b_tps,
            "speculative_tps": s_tps,
            "speedup_factor": speedup,
            "mean_accepted_tokens_estimate": max(0.0, min(speedup - 1.0, float(k))),
        }

    summary = {
        "config": {
            "target_model": baseline["config"]["target_model"],
            "speculative_config": speculative["config"].get("speculative_config"),
            "num_speculative_tokens": k,
            "gpu_memory_utilization": baseline["config"]["gpu_memory_utilization"],
            "max_output_tokens": baseline["config"]["max_output_tokens"],
            "num_requests_per_task": baseline["config"]["num_requests_per_task"],
        },
        "tasks": tasks,
        "metadata": {
            "baseline_timestamp": baseline["metadata"]["timestamp"],
            "speculative_timestamp": speculative["metadata"]["timestamp"],
        },
    }

    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nCombined summary saved to: {path}")


# ---------------------------------------------------------------------------
# CLI entry points
# ---------------------------------------------------------------------------

async def main_baseline(args: argparse.Namespace) -> None:
    print("\n" + "=" * 70)
    print("RUN A: Baseline — Standard 8B Inference (no speculation)")
    print("=" * 70)

    results = await run_single(args, run_name="baseline")
    save_run(
        Path(args.output_dir), "speculative_baseline.json",
        "baseline", results, args.model, speculative_cfg=None,
    )


async def main_speculative(args: argparse.Namespace) -> None:
    spec_cfg = build_speculative_cfg(args.draft_method, args.draft_model)
    # Snapshot before run_single: vLLM's AsyncEngineArgs.__init__ mutates the
    # dict in-place, injecting non-JSON-serializable ModelConfig objects.
    spec_cfg_for_save = dict(spec_cfg)

    print("\n" + "=" * 70)
    print("RUN B: Speculative Decoding")
    print(f"  draft_method={args.draft_method}  num_speculative_tokens={NUM_SPECULATIVE_TOKENS}")
    if args.draft_method == "model":
        print(f"  draft_model={args.draft_model}")
    elif args.draft_method == "ngram":
        print(f"  prompt_lookup_max={NGRAM_PROMPT_LOOKUP_MAX}")
    print("=" * 70)

    results = await run_single(args, run_name="speculative", speculative_cfg=spec_cfg)
    save_run(
        Path(args.output_dir), "speculative_spec.json",
        "speculative", results, args.model, speculative_cfg=spec_cfg_for_save,
    )


def main_summary(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    baseline = load_run(output_dir, "speculative_baseline.json")
    speculative = load_run(output_dir, "speculative_spec.json")

    if baseline is None or speculative is None:
        print("\nRun --baseline-only and --speculative-only first.")
        return

    print_comparison(baseline, speculative)
    save_combined_summary(output_dir, baseline, speculative)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Speculative Decoding Benchmark — Standard vs. Speculative Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--baseline-only",
        action="store_true",
        help="Run A: standard 8B inference. Saves speculative_baseline.json.",
    )
    mode.add_argument(
        "--speculative-only",
        action="store_true",
        help="Run B: speculative inference. Saves speculative_spec.json.",
    )
    mode.add_argument(
        "--summary",
        action="store_true",
        help="Print comparison from both saved JSON files (no inference).",
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_TARGET_MODEL,
        help=f"Target model (default: {DEFAULT_TARGET_MODEL})",
    )
    parser.add_argument(
        "--draft-method",
        default=DEFAULT_DRAFT_METHOD,
        choices=["ngram", "model"],
        help=(
            "Speculative draft strategy. "
            "'ngram' uses prompt-lookup (no extra model needed, default). "
            "'model' uses a small neural draft model (requires --draft-model)."
        ),
    )
    parser.add_argument(
        "--draft-model",
        default=DEFAULT_DRAFT_MODEL,
        help=(
            f"Draft model HF id, used only when --draft-method=model "
            f"(default: {DEFAULT_DRAFT_MODEL})"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory for JSON output (default: results/)",
    )

    args = parser.parse_args()

    if args.speculative_only and args.draft_method == "model" and not args.draft_model:
        parser.error("--draft-model is required when --draft-method=model")

    if args.baseline_only:
        asyncio.run(main_baseline(args))
    elif args.speculative_only:
        asyncio.run(main_speculative(args))
    else:
        main_summary(args)


if __name__ == "__main__":
    main()
