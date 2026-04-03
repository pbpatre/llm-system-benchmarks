"""
scripts/save_tokenizer.py
=========================
Run this ONCE on your HOST machine (outside Docker) to download and save
the Llama-3.1-8B-Instruct tokenizer into the model repository.

The saved files are picked up by the volume mount inside the container:
    host:  inference/ensemble/triton_on_mac/preprocess/tokenizer/
    container: /models/preprocess/tokenizer/

This avoids:
  - Downloading ~4.5 GB of weights just to get vocab files
  - HuggingFace token auth inside the container
  - Network access from inside the container at runtime

Usage
-----
    # Option A: use Llama-3.1-8B-Instruct (requires HF token with access grant)
    huggingface-cli login
    python inference/ensemble/triton_on_mac/scripts/save_tokenizer.py

    # Option B: use a public Llama tokenizer stub (no token needed, good for testing)
    python inference/ensemble/triton_on_mac/scripts/save_tokenizer.py \
        --model hf-internal-testing/llama-tokenizer

    # Option C: point at a locally downloaded model dir
    python inference/ensemble/triton_on_mac/scripts/save_tokenizer.py \
        --model /path/to/local/Llama-3.1-8B-Instruct
"""

import argparse
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[4]  # llm-system-benchmarks/
    default_out = (
        repo_root
        / "inference/ensemble/triton_on_mac/preprocess/tokenizer"
    )
    parser = argparse.ArgumentParser(description="Save HF tokenizer to local disk.")
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="HuggingFace model ID or local path to load tokenizer from.",
    )
    parser.add_argument(
        "--output",
        default=str(default_out),
        help=f"Directory to save tokenizer files into. Default: {default_out}",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HuggingFace access token (or set HF_TOKEN env var).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Lazy import so this script can be run without the full training env.
    try:
        from transformers import AutoTokenizer
    except ImportError:
        raise SystemExit(
            "transformers is not installed on the host.\n"
            "Run:  pip install transformers tokenizers"
        )

    token = args.token or os.environ.get("HF_TOKEN")

    print(f"Loading tokenizer from: {args.model!r}")
    if token:
        print("  (using provided HF token)")
    elif args.model.startswith("meta-llama"):
        print(
            "  ⚠️  No HF token found. meta-llama models are gated.\n"
            "  Run: huggingface-cli login\n"
            "  Or:  export HF_TOKEN=hf_...\n"
            "  Or:  pass --model hf-internal-testing/llama-tokenizer for a public stub."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=True,
        token=token,
    )

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer.save_pretrained(str(out_dir))
    print(f"\n✅  Tokenizer saved to: {out_dir}")
    print(f"   Files: {sorted(f.name for f in out_dir.iterdir())}")
    print(
        "\nThe container will find it at /models/preprocess/tokenizer/ "
        "via the volume mount.\n"
        "No config changes needed — preprocess/config.pbtxt already points there."
    )


if __name__ == "__main__":
    main()
