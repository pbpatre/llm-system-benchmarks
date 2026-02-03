#!/usr/bin/env python3
"""
===================================================================================
Experiment 1: The Anatomy of a Request (Baseline)
===================================================================================

This experiment breaks down the latency of a single LLM preprocessing request
into its three main stages:

1. **Jinja Templating**: Converting chat messages into a formatted string
   - Python-bound, GIL-constrained
   - Uses Jinja2 template engine
   
2. **BPE Tokenization**: Converting the string into token IDs
   - Rust-bound (HuggingFace Tokenizers)
   - Releases the GIL for true parallelism
   
3. **Collation/Padding**: Creating GPU-ready tensors
   - Python/PyTorch operations
   - Generally fast, minimal overhead

Key Insight: Even for a single request, Jinja templating can represent a
significant portion of the total preprocessing time, especially for models
with complex chat templates.

Usage:
    python exp1_baseline.py                    # Run with default config
    python exp1_baseline.py --quick            # Quick mode (~1 min)
    python exp1_baseline.py --tokens 5000      # Custom token count
"""

import os
import gc
import time
import warnings
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, logging as hf_logging

# Suppress HuggingFace warnings
hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=FutureWarning)

# Import from common modules
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import CONFIG, ExperimentResult, generate_target_token_conversation


class Experiment1Baseline:
    """
    Experiment 1: Breakdown latency for a single request.
    
    Measures the time spent in each preprocessing stage for a single
    large request (~50k tokens by default) to establish baseline metrics.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the experiment with configuration."""
        self.config = config or CONFIG
        self.hf_token = os.getenv("HF_TOKEN")
        self.tokenizer: Optional[AutoTokenizer] = None
        
        if not self.hf_token:
            print("⚠️  WARNING: HF_TOKEN environment variable not set.")
            print("   Some models may require authentication.")
    
    def _load_tokenizer(self) -> bool:
        """Load the tokenizer with error handling and validation."""
        if self.tokenizer is not None:
            return True
            
        try:
            print(f"📥 Loading tokenizer: {self.config['model_id']}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config["model_id"],
                trust_remote_code=True,
                token=self.hf_token,
                use_fast=True
            )
            
            # Validate fast tokenizer (Rust-backed)
            assert self.tokenizer.is_fast, (
                "Error: Fast Tokenizer (Rust) not active. "
                "This benchmark requires a Rust-backed tokenizer."
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            print(f"   ✓ Loaded successfully (is_fast={self.tokenizer.is_fast})")
            return True
            
        except Exception as e:
            print(f"   ✗ Failed to load tokenizer: {e}")
            return False
    
    def _warmup(self, messages, num_runs: int = None) -> None:
        """Run warmup passes to stabilize measurements."""
        num_runs = num_runs or self.config["warmup_runs"]
        for _ in range(num_runs):
            raw_str = self.tokenizer.apply_chat_template(messages, tokenize=False)
            _ = self.tokenizer.encode(raw_str)
            gc.collect()
    
    def run(self) -> ExperimentResult:
        """
        Run the baseline experiment.
        
        Returns:
            ExperimentResult containing latency breakdown data
        """
        print("\n" + "="*70)
        print("EXPERIMENT 1: The Anatomy of a Request (Baseline)")
        print("="*70)
        
        if not self._load_tokenizer():
            return ExperimentResult("baseline", 1, pd.DataFrame(), {"error": "Failed to load tokenizer"})
        
        target_tokens = self.config["target_tokens"]
        print(f"   Target: ~{target_tokens:,} tokens (large context)")
        
        # Generate test data
        messages = generate_target_token_conversation(
            self.tokenizer, 
            target_tokens=target_tokens,
            unique_id=12345
        )
        
        print(f"   Running {self.config['warmup_runs']} warmup passes...")
        self._warmup(messages)
        
        # Measurement arrays
        jinja_times = []
        tokenize_times = []
        collate_times = []
        token_counts = []
        
        num_iterations = self.config["num_iterations"]
        print(f"   Running {num_iterations} measurement iterations...")
        
        for i in range(num_iterations):
            gc.collect()
            
            # Stage 1: Jinja Templating
            start = time.perf_counter()
            raw_str = self.tokenizer.apply_chat_template(messages, tokenize=False)
            jinja_times.append((time.perf_counter() - start) * 1000)
            
            # Stage 2: Tokenization (Rust)
            start = time.perf_counter()
            input_ids = self.tokenizer.encode(raw_str, add_special_tokens=False)
            tokenize_times.append((time.perf_counter() - start) * 1000)
            token_counts.append(len(input_ids))
            
            # Stage 3: Collation/Padding
            start = time.perf_counter()
            tensor = torch.tensor([input_ids])
            pad_to = 2 ** (tensor.shape[1] - 1).bit_length()
            padded = torch.nn.functional.pad(
                tensor, 
                (0, pad_to - tensor.shape[1]),
                value=self.tokenizer.pad_token_id
            )
            _ = (padded != self.tokenizer.pad_token_id).long()
            collate_times.append((time.perf_counter() - start) * 1000)
        
        # Build results dataframe
        results_data = {
            "Stage": ["Jinja Templating", "Tokenization (Rust)", "Collation/Padding", "TOTAL"],
            "Mean (ms)": [
                np.mean(jinja_times),
                np.mean(tokenize_times),
                np.mean(collate_times),
                np.mean(jinja_times) + np.mean(tokenize_times) + np.mean(collate_times)
            ],
            "Std (ms)": [
                np.std(jinja_times),
                np.std(tokenize_times),
                np.std(collate_times),
                np.sqrt(np.var(jinja_times) + np.var(tokenize_times) + np.var(collate_times))
            ],
            "Min (ms)": [
                np.min(jinja_times),
                np.min(tokenize_times),
                np.min(collate_times),
                np.min(jinja_times) + np.min(tokenize_times) + np.min(collate_times)
            ],
            "Max (ms)": [
                np.max(jinja_times),
                np.max(tokenize_times),
                np.max(collate_times),
                np.max(jinja_times) + np.max(tokenize_times) + np.max(collate_times)
            ],
        }
        
        df = pd.DataFrame(results_data)
        
        # Add percentage column
        total_mean = df[df["Stage"] == "TOTAL"]["Mean (ms)"].values[0]
        df["% of Total"] = df["Mean (ms)"].apply(
            lambda x: f"{(x / total_mean * 100):.1f}%" if x != total_mean else "100%"
        )
        
        # Metadata
        metadata = {
            "token_count": int(np.mean(token_counts)),
            "num_messages": len(messages),
            "raw_string_length": len(raw_str),
        }
        
        # Print results
        print(f"\n   Token count: ~{metadata['token_count']:,}")
        print(f"   Messages: {metadata['num_messages']}")
        print(f"   Raw string length: {metadata['raw_string_length']:,} chars")
        print(f"\n{df.to_string(index=False)}")
        
        return ExperimentResult("baseline", 1, df, metadata)


def main():
    """CLI entry point for Experiment 1."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Experiment 1: The Anatomy of a Request (Baseline)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--quick", action="store_true", help="Quick mode with fewer iterations")
    parser.add_argument("--tokens", type=int, default=None, help="Target token count")
    parser.add_argument("--iterations", type=int, default=None, help="Number of measurement iterations")
    
    args = parser.parse_args()
    
    # Build config
    config = CONFIG.copy()
    if args.quick:
        config["warmup_runs"] = 2
        config["num_iterations"] = 3
        config["target_tokens"] = 1000
    if args.tokens:
        config["target_tokens"] = args.tokens
    if args.iterations:
        config["num_iterations"] = args.iterations
    
    # Run experiment
    experiment = Experiment1Baseline(config)
    result = experiment.run()
    
    # Save results
    if not result.data.empty:
        output_path = "exp1_baseline_results.csv"
        result.data.to_csv(output_path, index=False)
        print(f"\n   ✓ Results saved to: {output_path}")
    
    return result


if __name__ == "__main__":
    main()
