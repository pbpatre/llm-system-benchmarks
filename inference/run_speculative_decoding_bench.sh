#!/usr/bin/env bash
# Speculative Decoding Benchmark Orchestrator
#
# Runs the benchmark in two separate Python processes (one per engine config)
# so that GPU memory is fully reclaimed between runs, then prints a comparison.
#
#   Pass 1: Standard 8B inference (baseline)  → results/speculative_baseline.json
#   Pass 2: Speculative inference              → results/speculative_spec.json
#   Pass 3: --summary compares both files     → results/speculative_decoding_summary.json
#
# Draft strategies (--draft-method):
#   ngram  No extra model needed. Uses prompt-lookup n-gram matching (default).
#           Works immediately with no HF permissions required.
#   model  Uses a small neural draft model (--draft-model).
#           Requires a Llama-3 family model (same tokenizer as the target).
#           Recommended: meta-llama/Llama-3.2-1B-Instruct (HF access required).
#
# Usage:
#   # Default — ngram drafter (no HF permissions needed):
#   ./run_speculative_decoding_bench.sh
#
#   # Neural drafter — once HF access is granted:
#   ./run_speculative_decoding_bench.sh --draft-method model \
#       --draft-model meta-llama/Llama-3.2-1B-Instruct
#
#   # Override target model:
#   ./run_speculative_decoding_bench.sh --model meta-llama/Llama-3.1-8B-Instruct
#
# The inference venv (inference/.venv/) is created automatically on first run
# via setup_inference_env.sh if it does not already exist.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
RESULTS_DIR="$SCRIPT_DIR/../results"
MODEL="meta-llama/Llama-3.1-8B-Instruct"
DRAFT_MODEL="meta-llama/Llama-3.2-1B-Instruct"
DRAFT_METHOD="model"   # "model" | "ngram"

# ---------------------------------------------------------------------------
# Parse flags
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --draft-method)
            DRAFT_METHOD="$2"
            shift 2
            ;;
        --draft-model)
            DRAFT_MODEL="$2"
            shift 2
            ;;
        -h|--help)
            head -n 35 "$0" | grep "^#" | sed 's/^# \?//'
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

echo ""
echo "============================================================"
echo "  Speculative Decoding Benchmark"
echo "============================================================"
echo "  Target model:  $MODEL"
echo "  Draft method:  $DRAFT_METHOD"
if [[ "$DRAFT_METHOD" == "model" ]]; then
    echo "  Draft model:   $DRAFT_MODEL"
fi
echo "  Results dir:   $RESULTS_DIR"
echo "============================================================"

# ---------------------------------------------------------------------------
# Pass 1: Baseline — standard inference (no speculation)
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  Pass 1: Baseline (no speculation)"
echo "============================================================"
run_timed "baseline" python "$SCRIPT_DIR/speculative_decoding_bench.py" \
    --baseline-only \
    --model "$MODEL" \
    --output-dir "$RESULTS_DIR"

# ---------------------------------------------------------------------------
# Pass 2: Speculative inference
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  Pass 2: Speculative (draft_method=$DRAFT_METHOD, k=5)"
echo "============================================================"

SPEC_ARGS=(
    --speculative-only
    --model "$MODEL"
    --draft-method "$DRAFT_METHOD"
    --output-dir "$RESULTS_DIR"
)
if [[ "$DRAFT_METHOD" == "model" ]]; then
    SPEC_ARGS+=(--draft-model "$DRAFT_MODEL")
fi

run_timed "speculative" python "$SCRIPT_DIR/speculative_decoding_bench.py" "${SPEC_ARGS[@]}"

# ---------------------------------------------------------------------------
# Pass 3: Summary comparison
# ---------------------------------------------------------------------------
echo ""
python "$SCRIPT_DIR/speculative_decoding_bench.py" \
    --summary \
    --output-dir "$RESULTS_DIR"

echo ""
echo "============================================================"
echo "  Benchmark Complete"
echo "============================================================"
echo ""
echo "Output files:"
echo "  $RESULTS_DIR/speculative_baseline.json"
echo "  $RESULTS_DIR/speculative_spec.json"
echo "  $RESULTS_DIR/speculative_decoding_summary.json"
echo ""
echo "To view the summary again:"
echo "  python $SCRIPT_DIR/speculative_decoding_bench.py --summary --output-dir $RESULTS_DIR"
echo ""
echo "To switch to a neural drafter once HF access is granted:"
echo "  ./run_speculative_decoding_bench.sh --draft-method model \\"
echo "      --draft-model meta-llama/Llama-3.2-1B-Instruct"
echo ""
