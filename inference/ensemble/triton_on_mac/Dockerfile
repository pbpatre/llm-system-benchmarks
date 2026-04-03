# ==============================================================================
# Dockerfile
# Extends the official Triton server image with the Python packages needed
# by the preprocess and llama_vllm_shim backends.
#
# Why not rely on requirements.txt auto-install?
# -----------------------------------------------
# Triton's Python backend DOES support auto-installing from requirements.txt,
# but only when the python stub is configured to use a virtualenv
# (--backend-config=python,python-runtime=<path>).  In the default "stub"
# mode used by tritonserver:24.08-py3, the backend runs model.py directly
# inside the system Python of the container.  There is no pip install step
# unless you explicitly configure it.
#
# The cleanest solution for reproducible deployments is to bake the packages
# into a derived image.  This also eliminates the ~2 minute pip install delay
# on every container restart during development.
#
# Build (run from repo root):
#   docker build -t triton-mac-ensemble:latest \
#     inference/ensemble/triton_on_mac/
#
# Run (run from repo root):
#   IMPORTANT: mount triton_on_mac/ as /models, NOT the repo root.
#   Triton treats every subdirectory of --model-repository as a model.
#   Mounting the repo root would expose .git/, .venv/, training/, etc.
#
#   docker run --rm -it \
#     -e CUDA_VISIBLE_DEVICES="" \
#     -v $(pwd)/inference/ensemble/triton_on_mac:/models \
#     -v $(pwd)/inference/models/gguf:/weights \
#     -p 8000:8000 -p 8001:8001 -p 8002:8002 \
#     triton-mac-ensemble:latest \
#     tritonserver \
#       --model-repository=/models \
#       --backend-config=python,shm-default-byte-size=2097152 \
#       --log-verbose=1 \
#       --strict-readiness=false \
#       --exit-on-error=false
#
# NOTE on Metal + Docker:
#   The Linux container cannot access Apple's Metal GPU API.
#   llama-cpp-python inside the container runs on CPU only.
#   For Metal acceleration, use scripts/run_native_shim.py (see below)
#   which runs the LLM natively on macOS outside Docker.
# ==============================================================================

FROM nvcr.io/nvidia/tritonserver:24.08-py3

# Install Python dependencies for the Python backends.
# We pin versions to match what was tested; loosen as needed.
RUN pip install --no-cache-dir \
    transformers>=4.40.0 \
    tokenizers>=0.19.0 \
    huggingface_hub>=0.23.0 \
    jinja2>=3.1.0 \
    numpy>=1.24.0

# Install llama-cpp-python with Metal (Apple GPU) support.
# CMAKE_ARGS tells the build system to enable the Metal backend.
# llama-cpp-python compiles llama.cpp from source during pip install,
# so this step takes ~3-5 minutes but only runs once at image build time.
#
# NOTE: The Triton container runs Linux (inside Docker Desktop's Linux VM),
# NOT macOS. Metal is an Apple-only API and is NOT available inside the
# Linux container — even on M3 hardware.
# For the container, we build with CPU support only.
# Metal acceleration is available when running llama-cpp-python NATIVELY
# on macOS (outside Docker).
#
# Two options:
#   A) Run llama_vllm_shim natively on macOS (outside Triton) — gets Metal.
#   B) Run inside container with CPU — slower but simpler architecture.
#
# We install the CPU build here for container use.
# See scripts/run_native_shim.py for the Metal-native alternative.
RUN pip install --no-cache-dir llama-cpp-python

# Default command: start Triton with SHM sized for batch=32, seq=4096, INT32.
# 32 * 4096 * 4 = 524288 bytes. Use 2 MiB for headroom.
# CUDA_VISIBLE_DEVICES="" is critical on M3/CPU-only hosts.
# Without it, Triton's ensemble scheduler tries to create a CUDA stream for
# intermediate tensor transfers and fails with:
#   "CUDA driver version is insufficient for CUDA runtime version"
# Setting it to empty string makes Triton treat the host as GPU-less,
# forcing all memory operations (including ensemble inter-step tensors) to CPU.
ENV CUDA_VISIBLE_DEVICES=""

CMD [ \
    "tritonserver", \
    "--model-repository=/models", \
    "--backend-config=python,shm-default-byte-size=2097152", \
    "--log-verbose=1", \
    "--strict-readiness=false", \
    "--exit-on-error=false" \
]
