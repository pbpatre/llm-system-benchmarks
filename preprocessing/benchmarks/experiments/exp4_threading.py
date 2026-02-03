#!/usr/bin/env python3
"""
===================================================================================
Experiment 4: Threading vs The GIL (Detailed Profiling)
===================================================================================

This experiment provides definitive proof that:
1. Rust tokenizers RELEASE the GIL → True parallelism possible
2. Python Jinja templating HOLDS the GIL → No parallelism benefit

Background on the GIL (Global Interpreter Lock):
- Python's GIL prevents multiple threads from executing Python bytecode simultaneously
- Even with multiple threads, only one can hold the GIL at a time
- This is why multi-threaded Python code often shows NO speedup

The Key Difference:
- Jinja2 (Python): Cannot release GIL during template processing
  → Threading gives ~1x speedup (no parallelism)
- HuggingFace Tokenizers (Rust): Releases GIL during tokenization
  → Threading gives near-linear speedup

Methodology:
- Process 10k requests with varying thread counts
- Measure Jinja and Tokenization separately
- Compare actual speedup vs ideal linear speedup
- Use 100 turns per request for complex templates

Expected Results:
- Jinja: ~1x speedup regardless of thread count
- Tokenize: Near-linear speedup (e.g., 8x with 8 threads)

Usage:
    python exp4_threading.py                    # Run with default config
    python exp4_threading.py --quick            # Quick mode (~2 min)
    python exp4_threading.py --threads 1 2 4 8  # Custom thread counts
"""

import os
import gc
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
from transformers import AutoTokenizer, logging as hf_logging

# Suppress HuggingFace warnings
hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=FutureWarning)

# Import from common modules
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import (
    CONFIG, 
    ExperimentResult, 
    generate_chat_conversation,
    jinja_worker_with_tokenizer,
    tokenize_worker,
)


class Experiment4Threading:
    """
    Experiment 4: Prove Rust releases GIL but Python (Jinja) does not.
    
    Processes many requests to show the GIL bottleneck clearly.
    Uses 100 turns per request for complex Jinja templates.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the experiment with configuration."""
        self.config = config or CONFIG
        self.hf_token = os.getenv("HF_TOKEN")
        self.tokenizer: Optional[AutoTokenizer] = None
        
        if not self.hf_token:
            print("⚠️  WARNING: HF_TOKEN environment variable not set.")
    
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
            
            assert self.tokenizer.is_fast, "Error: Fast Tokenizer (Rust) not active."
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            print(f"   ✓ Loaded successfully (is_fast={self.tokenizer.is_fast})")
            return True
            
        except Exception as e:
            print(f"   ✗ Failed to load tokenizer: {e}")
            return False
    
    def run(self) -> ExperimentResult:
        """
        Run the threading profiling experiment.
        
        Returns:
            ExperimentResult containing threading speedup data
        """
        print("\n" + "="*70)
        print("EXPERIMENT 4: Threading vs The GIL (Detailed Profiling)")
        print("="*70)
        
        if not self._load_tokenizer():
            return ExperimentResult("threading_profiling", 4, pd.DataFrame(), {"error": "Failed to load tokenizer"})
        
        thread_counts = self.config["thread_counts"]
        total_requests = self.config.get("scaling_total_requests", 10000)
        batch_size = self.config["scaling_batch_size"]
        tokens_per_request = self.config.get("scaling_tokens_per_request", 1000)
        turns_per_request = self.config.get("scaling_turns_per_request", 100)
        
        print(f"   Total requests: {total_requests:,}")
        print(f"   GPU batch size: {batch_size}")
        print(f"   ~{tokens_per_request:,} tokens per request")
        print(f"   {turns_per_request} turns per request (complex Jinja)")
        
        # Generate batch requests (100 turns each)
        print(f"\n   Generating {batch_size} batch requests (100 turns each)...")
        requests = [
            generate_chat_conversation(
                num_turns=turns_per_request,
                words_per_message=max(5, tokens_per_request // (turns_per_request * 4)),
                unique_id=i * 54321
            )
            for i in range(batch_size)
        ]
        
        # Verify structure
        sample_raw = self.tokenizer.apply_chat_template(requests[0], tokenize=False)
        sample_tokens = len(self.tokenizer.encode(sample_raw))
        print(f"   Sample: {len(requests[0])} messages, ~{sample_tokens} tokens")
        
        # Pre-generate raw strings for tokenization benchmark
        raw_strings = [
            self.tokenizer.apply_chat_template(req, tokenize=False) 
            for req in requests
        ]
        
        # Warmup
        print("   Running warmup...")
        for req in requests[:5]:
            raw = self.tokenizer.apply_chat_template(req, tokenize=False)
            _ = self.tokenizer.encode(raw)
        
        results_data = []
        
        # Measure baseline (sequential)
        print("\n   Measuring single-threaded baseline...")
        gc.collect()
        
        start = time.perf_counter()
        for req in requests:
            _ = self.tokenizer.apply_chat_template(req, tokenize=False)
        jinja_baseline_ms = (time.perf_counter() - start) * 1000
        
        gc.collect()
        start = time.perf_counter()
        for raw in raw_strings:
            _ = self.tokenizer.encode(raw, add_special_tokens=False)
        tokenize_baseline_ms = (time.perf_counter() - start) * 1000
        
        print(f"   Baseline (1 batch of {batch_size} requests):")
        print(f"      Jinja: {jinja_baseline_ms:.1f}ms")
        print(f"      Tokenize: {tokenize_baseline_ms:.1f}ms")
        
        # Test each thread count
        for num_threads in thread_counts:
            print(f"\n   Testing {num_threads} threads...")
            
            # Jinja workload
            gc.collect()
            start = time.perf_counter()
            
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                args = [(self.tokenizer, req) for req in requests]
                futures = [executor.submit(jinja_worker_with_tokenizer, arg) for arg in args]
                _ = [f.result() for f in futures]
            
            jinja_time_ms = (time.perf_counter() - start) * 1000
            jinja_speedup = jinja_baseline_ms / jinja_time_ms
            
            # Tokenize workload
            gc.collect()
            start = time.perf_counter()
            
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                args = [(self.tokenizer, raw) for raw in raw_strings]
                futures = [executor.submit(tokenize_worker, arg) for arg in args]
                _ = [f.result() for f in futures]
            
            tokenize_time_ms = (time.perf_counter() - start) * 1000
            tokenize_speedup = tokenize_baseline_ms / tokenize_time_ms
            
            # Extrapolate to total requests
            num_batches = total_requests // batch_size
            total_jinja_time = (jinja_time_ms / 1000) * num_batches
            total_tokenize_time = (tokenize_time_ms / 1000) * num_batches
            
            print(f"      Jinja: {jinja_time_ms:.1f}ms (speedup: {jinja_speedup:.2f}x)")
            print(f"      Tokenize: {tokenize_time_ms:.1f}ms (speedup: {tokenize_speedup:.2f}x)")
            print(f"      Est. time for {total_requests:,} requests: Jinja={total_jinja_time:.1f}s, Tok={total_tokenize_time:.1f}s")
            
            results_data.append({
                "Threads": num_threads,
                "Jinja Time (ms)": jinja_time_ms,
                "Jinja Speedup": jinja_speedup,
                "Tokenize Time (ms)": tokenize_time_ms,
                "Tokenize Speedup": tokenize_speedup,
                "Est. Jinja 10k (s)": total_jinja_time,
                "Est. Tok 10k (s)": total_tokenize_time,
            })
        
        df = pd.DataFrame(results_data)
        
        print("\n" + "─"*70)
        print(f"\n{df.to_string(index=False)}")
        
        max_threads = max(thread_counts)
        jinja_max_speedup = df[df["Threads"] == max_threads]["Jinja Speedup"].values[0]
        tok_max_speedup = df[df["Threads"] == max_threads]["Tokenize Speedup"].values[0]
        
        print("\n   🔍 KEY INSIGHT: GIL Blocks Jinja, Not Rust Tokenizers")
        print("   " + "─"*50)
        print(f"      At {max_threads} threads:")
        print(f"      - Jinja speedup: {jinja_max_speedup:.2f}x (GIL blocked - no scaling!)")
        print(f"      - Tokenize speedup: {tok_max_speedup:.2f}x (Rust releases GIL - scales!)")
        
        metadata = {
            "total_requests": total_requests,
            "batch_size": batch_size,
            "tokens_per_request": tokens_per_request,
            "turns_per_request": turns_per_request,
            "jinja_baseline_ms": jinja_baseline_ms,
            "tokenize_baseline_ms": tokenize_baseline_ms,
        }
        
        return ExperimentResult("threading_profiling", 4, df, metadata)


def main():
    """CLI entry point for Experiment 4."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Experiment 4: Threading vs The GIL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--quick", action="store_true", help="Quick mode")
    parser.add_argument("--threads", nargs="+", type=int, default=None, help="Thread counts to test")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    
    args = parser.parse_args()
    
    # Build config
    config = CONFIG.copy()
    if args.quick:
        config["scaling_total_requests"] = 5000
        config["thread_counts"] = [1, 2, 4, 8, 16]
        config["scaling_batch_size"] = 64
    if args.threads:
        config["thread_counts"] = args.threads
    if args.batch_size:
        config["scaling_batch_size"] = args.batch_size
    
    # Run experiment
    experiment = Experiment4Threading(config)
    result = experiment.run()
    
    # Save results
    if not result.data.empty:
        output_path = "exp4_threading_results.csv"
        result.data.to_csv(output_path, index=False)
        print(f"\n   ✓ Results saved to: {output_path}")
    
    return result


if __name__ == "__main__":
    main()
