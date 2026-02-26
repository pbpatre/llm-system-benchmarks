#!/usr/bin/env bash
# Sets up an isolated venv for inference (separate from the uv-managed training env).
# The training environment pins torch==2.10.0, for which no vLLM wheel exists yet.
# This venv installs torch==2.5.1 + a compatible vLLM build and never touches uv.lock.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

echo "==> Setting up inference venv at: $VENV_DIR"

# Triton compiles a C extension at runtime that requires Python development headers.
# Install python3.10-dev if Python.h is missing (no-op if already present).
if ! python3.10-config --includes &>/dev/null 2>&1; then
    echo "==> Installing python3.10-dev (required by Triton's CUDA backend)..."
    apt-get install -y python3.10-dev
fi

# Detect CUDA version (major+minor) for the correct PyTorch wheel index.
# Falls back to cu121 (CUDA 12.1) which covers L40S.
CUDA_TAG=$(python3 -c "
import subprocess, re, sys

def try_nvcc():
    try:
        out = subprocess.run(['nvcc', '--version'], capture_output=True, text=True).stdout
        m = re.search(r'release (\d+)\.(\d+)', out)
        if m:
            return f'cu{m.group(1)}{m.group(2)}'
    except Exception:
        pass
    return None

def try_nvidia_smi():
    try:
        out = subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout
        m = re.search(r'CUDA Version: (\d+)\.(\d+)', out)
        if m:
            return f'cu{m.group(1)}{m.group(2)}'
    except Exception:
        pass
    return None

tag = try_nvcc() or try_nvidia_smi() or 'cu121'
print(tag)
" 2>/dev/null || echo "cu121")

echo "==> Detected CUDA tag: $CUDA_TAG"

if [[ -d "$VENV_DIR" ]]; then
    echo "==> Venv already exists at $VENV_DIR — skipping creation."
    echo "    Delete it and re-run this script to rebuild from scratch."
    exit 0
fi

python3 -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

pip install --upgrade pip --quiet

echo "==> Installing PyTorch 2.5.1 (${CUDA_TAG})..."
pip install "torch==2.5.1" \
    --extra-index-url "https://download.pytorch.org/whl/${CUDA_TAG}" \
    --quiet

echo "==> Installing vLLM (latest stable)..."
pip install vllm --quiet

echo "==> Installing remaining inference deps..."
pip install transformers numpy pandas --quiet

echo ""
echo "==> Done. Inference venv ready at: $VENV_DIR"
echo "    Activate manually with: source $VENV_DIR/bin/activate"
