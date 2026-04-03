# Triton Inference Server Ensemble — Mac M3 Development Setup

A local development scaffold for an NVIDIA Triton Inference Server **Ensemble pipeline** that runs on Apple Silicon (M3) without CUDA. Validates the full Sidecar architecture — tokenisation GIL scaling, POSIX shared memory tensor transfer, and GGUF inference — before deploying to an L40S GPU cluster.

---

## Architecture

```
gRPC Client (test_client.py)
        │
        │  PORT 8001
        ▼
┌─────────────────────────────────────────────────────┐
│  Triton Inference Server (Docker, Linux container)  │
│                                                     │
│  ensemble_model  [pure scheduler, no code]          │
│        │                                            │
│        ▼  PROMPT (TYPE_STRING)                      │
│  ┌─────────────────────┐                            │
│  │ preprocess (×2 GIL) │  AutoTokenizer             │
│  │  apply_chat_template│  apply_chat_template()     │
│  │  + tokenize         │  + encode() → INT32        │
│  └─────────────────────┘                            │
│        │  INPUT_IDS (TYPE_INT32)                    │
│        │  ← POSIX Shared Memory (zero copy) →       │
│        ▼                                            │
│  ┌─────────────────────┐                            │
│  │  llama_vllm_shim    │  llama-cpp-python          │
│  │  (CPU inference)    │  GGUF model                │
│  └─────────────────────┘                            │
│        │  RAW_OUTPUT (TYPE_STRING)                  │
│        ▼                                            │
│  ┌─────────────────────┐                            │
│  │ postprocess (×2 GIL)│  Strip role headers        │
│  │                     │  stop tokens, whitespace   │
│  └─────────────────────┘                            │
│        │  GENERATED_TEXT (TYPE_STRING)              │
└────────┼────────────────────────────────────────────┘
         │
         ▼
   Clean generated text
```

### Why Three Models?

| Model | Platform | Role |
|---|---|---|
| `preprocess` | Python backend | CPU-bound tokenisation, scales via `instance_group` |
| `llama_vllm_shim` | Python backend | LLM inference (GGUF/CPU locally, vLLM on L40S) |
| `postprocess` | Python backend | Output cleaning — strips Llama-3 role headers |
| `ensemble_model` | Ensemble | Pure config — chains the three steps |

### Shared Memory Between Steps

Token IDs flow from `preprocess` → `llama_vllm_shim` via **POSIX shared memory** — the same physical memory pages are mapped into both Python subprocesses. No serialisation, no network copy. Size the region at startup:

```
batch × seq_len × bytes = 32 × 4096 × 4 (INT32) = 512 KiB → use 2 MiB
--backend-config=python,shm-default-byte-size=2097152
```

### GIL Scaling via `instance_group`

`preprocess` and `postprocess` each use `instance_group [{ kind: KIND_CPU count: 2 }]`. Each `count` entry spawns a **separate Python subprocess** with its own GIL, so CPU-bound tokenisation scales horizontally without async workarounds.

---

## Repository Layout

```
inference/ensemble/triton_on_mac/
├── Dockerfile                    ← extends tritonserver with Python deps
├── ensemble_model/
│   └── config.pbtxt              ← 3-step ensemble pipeline definition
├── preprocess/
│   ├── config.pbtxt              ← Python backend, 2 GIL instances
│   ├── requirements.txt          ← transformers, tokenizers, jinja2
│   ├── tokenizer/                ← committed tokenizer files (~4 MB)
│   └── 1/model.py                ← apply_chat_template + tokenize
├── llama_vllm_shim/
│   ├── config.pbtxt              ← llama_cpp mode, dynamic batching
│   └── 1/model.py                ← dummy + llama_cpp backends
├── postprocess/
│   ├── config.pbtxt              ← 2 GIL instances
│   └── 1/model.py                ← strip role headers + stop tokens
├── scripts/
│   └── save_tokenizer.py         ← one-time tokenizer download
└── test_client.py                ← gRPC end-to-end test suite
```

Model weights (GGUF files) live in `inference/models/gguf/` which is gitignored.

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Docker Desktop | ≥ 4.x | Enable VirtioFS for best volume mount performance |
| Python | 3.10+ | Project uses uv for venv management |
| uv | any | `pip install uv` |
| `tritonclient[grpc]` | ≥ 2.x | `uv add "tritonclient[grpc]"` |
| huggingface_hub | any | `uv add huggingface_hub` |

---

## Quick Start

### Step 1 — Build the Docker image (one-time, ~10 min)

Extends `nvcr.io/nvidia/tritonserver:24.08-py3` with `transformers`, `jinja2`, and `llama-cpp-python`.

```bash
# Run from repo root
docker build -t triton-mac-ensemble:latest \
  inference/ensemble/triton_on_mac/
```

### Step 2 — Download the GGUF model (~700 MB)

[Llama-3.2-1B-Instruct-Q4_K_M](https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF) is a public community quantization — no HuggingFace token or license acceptance required.

```bash
huggingface-cli download \
  bartowski/Llama-3.2-1B-Instruct-GGUF \
  Llama-3.2-1B-Instruct-Q4_K_M.gguf \
  --local-dir inference/models/gguf/llama-3.2-1b/
```

The tokenizer is already committed to the repo at `preprocess/tokenizer/`. To regenerate it (e.g. after switching models):

```bash
python inference/ensemble/triton_on_mac/scripts/save_tokenizer.py \
  --model meta-llama/Llama-3.2-1B-Instruct
```

### Step 3 — Start Triton (Terminal 1)

```bash
docker run --rm -it \
  -e CUDA_VISIBLE_DEVICES="" \
  -v $(pwd)/inference/ensemble/triton_on_mac:/models \
  -v $(pwd)/inference/models/gguf:/weights \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  triton-mac-ensemble:latest \
  tritonserver \
    --model-repository=/models \
    --backend-config=python,shm-default-byte-size=2097152 \
    --log-verbose=1 \
    --strict-readiness=false \
    --exit-on-error=false
```

**Volume mounts explained:**

| Mount | Host path | Container path | Purpose |
|---|---|---|---|
| Models | `inference/ensemble/triton_on_mac/` | `/models` | Triton model repository |
| Weights | `inference/models/gguf/` | `/weights` | GGUF files |

**Why `CUDA_VISIBLE_DEVICES=""`?** The Triton image links against CUDA. On M3 (no CUDA driver), the ensemble scheduler emits a non-fatal CUDA stream error. Setting this env var prevents the error from triggering `--strict-readiness` and aborting startup.

Expected output once healthy:
```
+-----------------+---------+--------+
| Model           | Version | Status |
+-----------------+---------+--------+
| ensemble_model  | 1       | READY  |
| llama_vllm_shim | 1       | READY  |
| postprocess     | 1       | READY  |
| preprocess      | 1       | READY  |
+-----------------+---------+--------+
```

### Step 4 — Run the test suite (Terminal 2)

```bash
python inference/ensemble/triton_on_mac/test_client.py --skip-concurrent
```

Expected output:
```
✅  All models are live and ready.

── Test 5: postprocess output cleaning ──
  ✅ [0] 'assistant\n\nThe capital of France is Paris.' → 'The capital of France is Paris.'
  ✅ [1] 'assistant\n\n2 + 2 = 4<|eot_id|>'           → '2 + 2 = 4'
  All cleaning tests passed ✅

── Test 1: single prompt ──
  Prompt : 'What is the capital of France?'
  Output : 'The capital of France is Paris.'
  Latency: ~500 ms

── Test 2: batch of 4 prompts (variable length) ──
  Batch latency: ~3500 ms  (~875 ms/req)

✅  All tests passed.
```

---

## Configuration Reference

### `preprocess/config.pbtxt`

| Parameter | Default | Description |
|---|---|---|
| `model_name` | `/models/preprocess/tokenizer` | Local tokenizer path inside container |
| `fallback_model_name` | `meta-llama/Llama-3.2-1B-Instruct` | HF Hub fallback if local path missing |
| `max_length` | `4096` | Max tokenized sequence length |

### `llama_vllm_shim/config.pbtxt`

| Parameter | Default | Description |
|---|---|---|
| `backend_mode` | `llama_cpp` | `dummy` for SHM tests, `llama_cpp` for real inference |
| `model_path` | `/weights/llama-3.2-1b/Llama-3.2-1B-Instruct-Q4_K_M.gguf` | GGUF path inside container |
| `n_gpu_layers` | `0` | CPU only in container; `-1` = all layers on GPU |
| `max_new_tokens` | `128` | Max tokens to generate |

### `postprocess/config.pbtxt`

| Parameter | Default | Description |
|---|---|---|
| `stop_strings` | `<\|eot_id\|>,<\|end_of_text\|>` | Comma-separated stop markers |
| `strip_role_header` | `true` | Strip `assistant\n\n` prefix from output |

---

## Debugging

```bash
# Full logs from running container
docker logs -f <container_name>

# Shell inside container
docker exec -it <container_name> bash

# Inspect SHM regions (inside container)
ls /dev/shm
df -h /dev/shm

# HTTP health checks (no client library needed)
curl localhost:8000/v2/health/live
curl localhost:8000/v2/health/ready
curl localhost:8000/v2/models/ensemble_model | python3 -m json.tool

# Prometheus metrics
curl localhost:8002/metrics | grep -E "nv_inference_request_success|queue"
```

---

## L40S Cluster Migration

To deploy on the L40S GPU cluster, three changes are needed:

**1. Replace `llama_vllm_shim/` with the official vLLM backend:**
```
platform: "vllm"   (instead of backend: "python")
```
The `ensemble_model/config.pbtxt` stays **identical** — tensor names are unchanged.

**2. Update `instance_group` for GPU:**
```protobuf
# llama_vllm_shim/config.pbtxt
instance_group [ { kind: KIND_GPU  gpu: [0]  count: 1 } ]

# preprocess/postprocess — raise CPU count to match core allocation
instance_group [ { kind: KIND_CPU  count: 8 } ]
```

**3. Remove `CUDA_VISIBLE_DEVICES=""` and `--exit-on-error=false`** — not needed on a real CUDA host.

---

## Backend Mode Reference

| `backend_mode` | Weights needed | Metal | Use case |
|---|---|---|---|
| `dummy` | ❌ | ❌ | SHM/plumbing tests, instant response |
| `llama_cpp` | ✅ GGUF | ❌ (CPU in container) | Integration testing |
