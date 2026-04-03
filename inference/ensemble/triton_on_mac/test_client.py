"""
test_client.py
==============
End-to-end gRPC test client for the Triton ensemble pipeline:

    Client  →  ensemble_model  →  preprocess  →  llama_vllm_shim  →  Client

Usage
-----
1. Start Triton (see ensemble_model/config.pbtxt for the full docker command):

    docker run --rm -it \\
      -v $(pwd)/inference/ensemble/triton_on_mac:/models \\
      -p 8000:8000 -p 8001:8001 -p 8002:8002 \\
      nvcr.io/nvidia/tritonserver:24.08-py3 \\
        tritonserver \\
          --model-repository=/models \\
          --backend-config=python,shm-default-byte-size=4194304 \\
          --log-verbose=1

2. Install the client library:
    pip install tritonclient[grpc]

3. Run this script:
    python inference/ensemble/triton_on_mac/test_client.py

   Or with custom options:
    python inference/ensemble/triton_on_mac/test_client.py \\
        --url localhost:8001 \\
        --prompts "Tell me a joke" "Explain transformers in one sentence" \\
        --batch-size 4 \\
        --concurrency 2
"""

import argparse
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import numpy as np

try:
    import tritonclient.grpc as grpcclient
    from tritonclient.utils import InferenceServerException
except ImportError as _e:
    raise SystemExit(
        f"tritonclient import failed: {_e}\n"
        "Run:  pip install tritonclient[grpc]\n"
        "If already installed, ensure you are using the correct Python:\n"
        f"  python3 {__file__} --skip-concurrent"
    )


# ---------------------------------------------------------------------------
# Core inference helper
# ---------------------------------------------------------------------------

def build_request(
    prompts: list[str],
    model_name: str = "ensemble_model",
) -> tuple[list, list, str]:
    """Build Triton gRPC input/output objects for a batch of prompts.

    Returns
    -------
    inputs, outputs, model_name
    """
    batch_size = len(prompts)

    # TYPE_STRING tensors: numpy object array of bytes, shape [batch, 1]
    text_data = np.array(
        [[p.encode("utf-8")] for p in prompts], dtype=object
    )  # [batch, 1]

    inp = grpcclient.InferInput("PROMPT", text_data.shape, "BYTES")
    inp.set_data_from_numpy(text_data)

    out = grpcclient.InferRequestedOutput("GENERATED_TEXT")

    return [inp], [out], model_name


def infer_batch(
    client: grpcclient.InferenceServerClient,
    prompts: list[str],
    model_name: str = "ensemble_model",
    request_id: str = "0",
) -> list[str]:
    """Send one batched inference request and return decoded text outputs."""
    inputs, outputs, model_name = build_request(prompts, model_name)

    result = client.infer(
        model_name=model_name,
        inputs=inputs,
        outputs=outputs,
        request_id=request_id,
    )

    # result shape: [batch, 1], dtype object (bytes)
    raw = result.as_numpy("GENERATED_TEXT")  # numpy object array
    decoded = []
    for row in raw:
        cell = row[0] if hasattr(row, "__len__") else row
        if isinstance(cell, bytes):
            cell = cell.decode("utf-8")
        decoded.append(cell)
    return decoded


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

def check_server_health(client: grpcclient.InferenceServerClient) -> None:
    """Raise if Triton is not live or the ensemble model is not ready."""
    if not client.is_server_live():
        raise RuntimeError("Triton server is not live.")
    if not client.is_server_ready():
        raise RuntimeError("Triton server is not ready.")

    for model in ("preprocess", "llama_vllm_shim", "ensemble_model"):
        if not client.is_model_ready(model):
            raise RuntimeError(
                f"Model '{model}' is not ready.  "
                "Check tritonserver logs for load errors."
            )
    print("✅  All models are live and ready.")


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

def test_single_prompt(client: grpcclient.InferenceServerClient) -> None:
    """Test: single prompt, batch size 1."""
    print("\n── Test 1: single prompt ──")
    prompts = ["What is the capital of France?"]
    t0 = time.perf_counter()
    results = infer_batch(client, prompts, request_id="test-single")
    latency_ms = (time.perf_counter() - t0) * 1000
    print(f"  Prompt : {prompts[0]!r}")
    print(f"  Output : {results[0]!r}")
    print(f"  Latency: {latency_ms:.1f} ms")


def test_batch_prompts(
    client: grpcclient.InferenceServerClient,
    batch_size: int = 4,
) -> None:
    """Test: variable-length prompts in a single batched request.

    Verifies that padding in the preprocess step is handled correctly
    (shorter sequences are padded to match the longest one in the batch).
    """
    print(f"\n── Test 2: batch of {batch_size} prompts (variable length) ──")
    prompts = [
        "Hi",
        "Explain the difference between a mutex and a semaphore.",
        "Write a haiku about distributed systems.",
        "What is 2 + 2?",
    ][:batch_size]

    t0 = time.perf_counter()
    results = infer_batch(client, prompts, request_id="test-batch")
    latency_ms = (time.perf_counter() - t0) * 1000

    for i, (p, r) in enumerate(zip(prompts, results)):
        print(f"  [{i}] Prompt : {p!r}")
        print(f"  [{i}] Output : {r!r}")
    print(f"  Batch latency: {latency_ms:.1f} ms  ({latency_ms/len(prompts):.1f} ms/req)")


def test_concurrent_requests(
    url: str,
    prompts: list[str],
    concurrency: int = 4,
) -> None:
    """Test: N concurrent gRPC connections sending requests in parallel.

    This exercises the instance_group scaling of the preprocess model.
    Each thread opens its own InferenceServerClient (gRPC channels are not
    thread-safe to share across threads in all versions).
    """
    print(f"\n── Test 3: {concurrency} concurrent clients ──")

    lock = threading.Lock()
    latencies: list[float] = []

    def worker(idx: int) -> str:
        prompt = prompts[idx % len(prompts)]
        with grpcclient.InferenceServerClient(url=url) as c:
            t0 = time.perf_counter()
            result = infer_batch(c, [prompt], request_id=f"concurrent-{idx}")
            elapsed_ms = (time.perf_counter() - t0) * 1000
        with lock:
            latencies.append(elapsed_ms)
        return result[0]

    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(worker, i) for i in range(concurrency)]
        results = [f.result() for f in as_completed(futures)]
    wall_ms = (time.perf_counter() - t_start) * 1000

    print(f"  Completed {concurrency} requests in {wall_ms:.1f} ms wall time")
    print(f"  Per-request latency — "
          f"avg: {sum(latencies)/len(latencies):.1f} ms  "
          f"min: {min(latencies):.1f} ms  "
          f"max: {max(latencies):.1f} ms")
    for i, r in enumerate(results):
        print(f"  [{i}] {r!r}")


def test_model_metadata(client: grpcclient.InferenceServerClient) -> None:
    """Print metadata for all models – useful for debugging config."""
    print("\n── Test 4: model metadata ──")
    for model in ("preprocess", "llama_vllm_shim", "postprocess", "ensemble_model"):
        meta = client.get_model_metadata(model)
        print(f"\n  {model}:")
        print(f"    inputs : {[(i.name, i.datatype, i.shape) for i in meta.inputs]}")
        print(f"    outputs: {[(o.name, o.datatype, o.shape) for o in meta.outputs]}")


def test_output_cleaning(client: grpcclient.InferenceServerClient) -> None:
    """Test: verify postprocess strips role headers and stop tokens.

    Calls postprocess directly (bypassing the ensemble) to unit-test
    the cleaning logic independently of the LLM.
    """
    print("\n── Test 5: postprocess output cleaning ──")

    # Simulate raw llama-cpp-python outputs including common artifacts.
    raw_cases = [
        "assistant\n\nThe capital of France is Paris.",          # role header
        "assistant\n\n2 + 2 = 4<|eot_id|>",                     # header + stop token
        "<|start_header_id|>assistant<|end_header_id|>\n\nHello!", # full header
        "No header, already clean output.",                       # no cleanup needed
        "  Extra whitespace   ",                                  # whitespace only
    ]
    expected = [
        "The capital of France is Paris.",
        "2 + 2 = 4",
        "Hello!",
        "No header, already clean output.",
        "Extra whitespace",
    ]

    raw_array = np.array(
        [[s.encode("utf-8")] for s in raw_cases], dtype=object
    )
    inp = grpcclient.InferInput("RAW_OUTPUT", raw_array.shape, "BYTES")
    inp.set_data_from_numpy(raw_array)
    out = grpcclient.InferRequestedOutput("GENERATED_TEXT")

    result = client.infer(
        model_name="postprocess",
        inputs=[inp],
        outputs=[out],
        request_id="test-cleaning",
    )
    cleaned = result.as_numpy("GENERATED_TEXT")

    all_passed = True
    for i, (raw, exp) in enumerate(zip(raw_cases, expected)):
        got = cleaned[i][0]
        if isinstance(got, bytes):
            got = got.decode("utf-8")
        passed = got == exp
        status = "✅" if passed else "❌"
        print(f"  {status} [{i}] raw    : {raw!r}")
        print(f"       got    : {got!r}")
        if not passed:
            print(f"       expect : {exp!r}")
            all_passed = False

    if all_passed:
        print("  All cleaning tests passed ✅")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end test client for the Triton ensemble pipeline."
    )
    parser.add_argument(
        "--url",
        default="localhost:8001",
        help="Triton gRPC endpoint (default: localhost:8001)",
    )
    parser.add_argument(
        "--model",
        default="ensemble_model",
        help="Top-level model name to call (default: ensemble_model)",
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=["What is the capital of France?"],
        help="One or more prompt strings for the batch test.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for test 2 (default: 4)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Number of concurrent clients for test 3 (default: 4)",
    )
    parser.add_argument(
        "--skip-concurrent",
        action="store_true",
        help="Skip the concurrent-requests test.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Connecting to Triton at {args.url} …")
    client = grpcclient.InferenceServerClient(url=args.url, verbose=False)

    try:
        check_server_health(client)
        test_model_metadata(client)
        test_output_cleaning(client)
        test_single_prompt(client)
        test_batch_prompts(client, batch_size=args.batch_size)
        if not args.skip_concurrent:
            test_concurrent_requests(
                url=args.url,
                prompts=args.prompts,
                concurrency=args.concurrency,
            )
    except InferenceServerException as exc:
        print(f"\n❌  Inference error: {exc}")
        raise
    finally:
        client.close()

    print("\n✅  All tests passed.")


if __name__ == "__main__":
    main()
