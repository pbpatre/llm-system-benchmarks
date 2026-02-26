#!/usr/bin/env bash
# HOL Blocking Benchmark Orchestrator
#
# Runs the benchmark twice (naive vs. chunked prefill) using the isolated
# inference venv, then prints a side-by-side comparison table.
#
# Usage:
#   ./run_prefill_blocking_bench.sh [--model <model_id>]
#
# The inference venv (inference/.venv/) is created automatically on first run
# via setup_inference_env.sh if it does not already exist.
#
# Results are written to:
#   <repo_root>/results/naive.csv    (enable_chunked_prefill=False)
#   <repo_root>/results/chunked.csv  (enable_chunked_prefill=True)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
RESULTS_DIR="$SCRIPT_DIR/../results"
MODEL="meta-llama/Llama-3.1-8B-Instruct"

# ---------------------------------------------------------------------------
# Parse flags
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            MODEL="$2"
            shift 2
            ;;
        -h|--help)
            head -n 20 "$0" | grep "^#" | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Bootstrap inference venv if needed
# ---------------------------------------------------------------------------
if [[ ! -d "$VENV_DIR" ]]; then
    echo "==> Inference venv not found — running setup..."
    bash "$SCRIPT_DIR/setup_inference_env.sh"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

mkdir -p "$RESULTS_DIR"

# ---------------------------------------------------------------------------
# Helper: run a command and print elapsed wall-clock time
# ---------------------------------------------------------------------------
run_timed() {
    local label="$1"
    shift
    local t0
    t0=$(date +%s)
    "$@"
    local elapsed=$(( $(date +%s) - t0 ))
    echo "  [${label}] wall time: ${elapsed}s"
}

# ---------------------------------------------------------------------------
# Naive — no chunked prefill
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  Naive  (enable_chunked_prefill=False)"
echo "============================================================"
run_timed "naive" python "$SCRIPT_DIR/prefill_blocking_bench.py" \
    --no-chunked-prefill \
    --model "$MODEL" \
    --output-dir "$RESULTS_DIR"

# ---------------------------------------------------------------------------
# Chunked prefill — 512-token chunks
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  Chunked  (enable_chunked_prefill=True, chunk=512)"
echo "============================================================"
run_timed "chunked" python "$SCRIPT_DIR/prefill_blocking_bench.py" \
    --chunked-prefill \
    --model "$MODEL" \
    --output-dir "$RESULTS_DIR"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
python "$SCRIPT_DIR/prefill_blocking_bench.py" \
    --summary \
    --output-dir "$RESULTS_DIR"
