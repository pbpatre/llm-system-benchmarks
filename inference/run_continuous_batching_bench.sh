#!/usr/bin/env bash
# Continuous Batching vs. Static Batching Benchmark Orchestrator
#
# Runs both static batching (Run A) and continuous batching (Run B) using the
# isolated inference venv, then prints a comparison summary.
#
# Usage:
#   ./run_continuous_batching_bench.sh [--model <model_id>]
#
# The inference venv (inference/.venv/) is created automatically on first run
# via setup_inference_env.sh if it does not already exist.
#
# Results are written to:
#   <repo_root>/results/continuous_batching_summary.json
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
# Run both static and continuous batching
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  Continuous Batching vs. Static Batching Benchmark"
echo "============================================================"
echo "  Model: $MODEL"
echo "  Output: $RESULTS_DIR/continuous_batching_summary.json"
echo "============================================================"

run_timed "full_benchmark" python "$SCRIPT_DIR/continuous_batching_bench.py" \
    --model "$MODEL" \
    --output-dir "$RESULTS_DIR"

echo ""
echo "============================================================"
echo "  Benchmark Complete"
echo "============================================================"
echo ""
echo "To view the summary again:"
echo "  python $SCRIPT_DIR/continuous_batching_bench.py --summary --output-dir $RESULTS_DIR"
echo ""
