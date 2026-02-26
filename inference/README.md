# Inference Benchmarks

Benchmarks for LLM inference serving behaviour, focusing on request scheduling
and latency characteristics in [vLLM](https://github.com/vllm-project/vllm).

## Benchmarks

### Head-of-Line Blocking: Naive vs. Chunked Prefill

`prefill_blocking_bench.py` demonstrates how a large prefill request ("elephant",
8 192 input tokens) stalls concurrent decode streams ("mice", 20 × 64-token
inputs generating 128 tokens each) when chunked prefill is disabled, and how
enabling it eliminates the stall by interleaving 512-token prefill chunks with
ongoing decode steps.

Metrics captured per decode token across all mouse requests:

| Metric | Description |
|--------|-------------|
| P99 ITL | 99th-percentile inter-token latency |
| Jitter | Standard deviation of inter-token latency |
| Max stall | Longest single gap between consecutive tokens |

#### Running the benchmark

```bash
# Naive — elephant monopolises the GPU during its prefill
python prefill_blocking_bench.py --no-chunked-prefill

# Chunked — elephant is sliced into 512-token chunks, mice keep decoding
python prefill_blocking_bench.py --chunked-prefill

# Print a side-by-side comparison from previously saved CSVs
python prefill_blocking_bench.py --summary
```

Optional flags:

```
--model <hf_model_id>   Model to serve (default: meta-llama/Llama-3.1-8B-Instruct)
--output-dir <path>     Directory for CSV output / summary input (default: results/)
```

To run both passes and print the comparison in one step:

```bash
./run_prefill_blocking_bench.sh [--model <model_id>]
```

Results are written to:

```
results/naive.csv    # --no-chunked-prefill pass
results/chunked.csv  # --chunked-prefill pass
```

---

## Environment Setup

### Why a separate virtual environment?

The main project (preprocessing benchmarks, training utilities) is managed by
[uv](https://github.com/astral-sh/uv) and pins `torch>=2.10.0` in
`pyproject.toml`. At the time of writing, no vLLM wheel is published for
PyTorch 2.10. The inference benchmarks therefore run in a **completely separate
`inference/.venv/`** that installs `torch==2.5.1` alongside a compatible vLLM
build.

The two environments must remain isolated because:

- **Conflicting torch versions.** The training env requires `torch>=2.10`; the
  latest stable vLLM wheel targets `torch~=2.5`. Installing both in the same
  env would downgrade (or break) the training dependencies.
- **`uv.lock` integrity.** uv manages a single lock file for the project. Adding
  vLLM there would pull in a conflicting torch pin and corrupt the lock for all
  other contributors.
- **CUDA ABI compatibility.** PyTorch and vLLM ship pre-compiled CUDA extensions
  that must be built against the same torch ABI. Mixing versions causes silent
  correctness failures or import errors at runtime.

### Creating the inference environment

```bash
# From the repo root or the inference/ directory:
bash inference/setup_inference_env.sh
```

The script:

1. Detects your CUDA version via `nvcc` or `nvidia-smi` (falls back to `cu121`).
2. Installs `python3.10-dev` if `Python.h` is absent — required by Triton's
   NVIDIA backend, which JIT-compiles a C extension at runtime.
3. Creates `inference/.venv/` with `python3 -m venv`.
4. Installs `torch==2.5.1` from the matching PyTorch wheel index.
5. Installs the latest stable `vllm`, `transformers`, `numpy`, and `pandas`.

The environment is created once; re-running the script is a no-op if
`inference/.venv/` already exists. To rebuild from scratch, delete the directory
first:

```bash
rm -rf inference/.venv
bash inference/setup_inference_env.sh
```

### Activating the environment manually

```bash
source inference/.venv/bin/activate
```

The `run_prefill_blocking_bench.sh` orchestrator activates the venv
automatically before running any Python.

### System requirements

| Requirement | Notes |
|-------------|-------|
| Python 3.10 | Must match the venv interpreter; 3.11+ untested with this vLLM pin |
| `python3.10-dev` | Provides `Python.h` for Triton's C extension compilation |
| CUDA-capable GPU | Tested on L40S (48 GiB); minimum ~12 GiB GPU RAM recommended |
| CUDA driver ≥ 12.1 | Required by `torch==2.5.1+cu121` wheels |
| HuggingFace token | Required for gated models (e.g., Llama-3.1-8B-Instruct) |

Set your token before running:

```bash
export HF_TOKEN="your_token_here"
```
