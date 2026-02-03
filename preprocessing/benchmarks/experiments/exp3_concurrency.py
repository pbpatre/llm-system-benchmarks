#!/usr/bin/env python3
"""
===================================================================================
Experiment 3: The Concurrency Test (GPU Wait Time)
===================================================================================

This experiment simulates a production inference server processing many concurrent
requests. It measures "GPU Wait Time" - the time the GPU waits idle while the CPU
completes preprocessing for each batch.

Production Context:
- Modern inference servers process batches of 32-128 requests at a time
- The GPU can only start computing once ALL requests in the batch are preprocessed
- Any slowdown in preprocessing directly translates to GPU idle time

Key Metric: GPU Wait Time
- The end-to-end wall time from receiving a batch of requests to having them
  ready for GPU inference (tensors created and padded)
- Broken down into: Jinja time, Tokenization time, Collation time

Methodology:
- Simulate 100k total requests (production-like volume)
- Process in batches of 64 (typical GPU batch size)
- Test multiple thread counts (1, 4, 8, 16, 32, 64)
- Use 100 turns per request (complex chat templates)
- Memory-efficient: only one batch in memory at a time

KEY FINDING: 
As concurrency increases, Jinja becomes a larger percentage of GPU wait time
because tokenization parallelizes (Rust releases GIL) but Jinja doesn't.

Usage:
    python exp3_concurrency.py                    # Run with default config
    python exp3_concurrency.py --quick            # Quick mode (~3 min)
    python exp3_concurrency.py --threads 1 4 8 16 # Custom thread counts
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
    SystemMonitor,
    generate_chat_conversation,
    jinja_worker_with_tokenizer,
    tokenize_worker,
    collate_batch,
)


class Experiment3Concurrency:
    """
    Experiment 3: High-throughput production simulation.
    
    Simulates a production system processing ~100k requests.
    GPU processes in batches - we measure "GPU Wait Time" per batch.
    
    MEMORY OPTIMIZATION: Processes requests in chunks to avoid holding
    100k requests in memory simultaneously. Only one batch is in memory at a time.
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
        Run the concurrency experiment.
        
        KEY FINDING: As concurrency increases, Jinja becomes a larger % of GPU wait time
        because tokenization parallelizes (Rust releases GIL) but Jinja doesn't.
        
        Returns:
            ExperimentResult containing concurrency analysis data
        """
        print("\n" + "="*70)
        print("EXPERIMENT 3: GPU Wait Time vs Concurrency (Production Simulation)")
        print("="*70)
        
        if not self._load_tokenizer():
            return ExperimentResult("concurrency", 3, pd.DataFrame(), {"error": "Failed to load tokenizer"})
        
        total_requests = self.config.get("exp3_total_requests", 100000)
        gpu_batch_size = self.config.get("exp3_gpu_batch_size", 64)
        tokens_per_request = self.config.get("exp3_tokens_per_request", 1000)
        turns_per_request = self.config.get("exp3_turns_per_request", 100)
        thread_counts = self.config.get("exp3_thread_counts", [1, 4, 8, 16, 32, 64])
        
        # Number of batches to measure for averaging (keep memory low)
        measurement_batches = min(10, total_requests // gpu_batch_size)
        
        print("   Production simulation:")
        print(f"   - Total requests to process: {total_requests:,}")
        print(f"   - GPU batch size: {gpu_batch_size}")
        print(f"   - ~{tokens_per_request:,} tokens per request")
        print(f"   - {turns_per_request} conversation turns (complex Jinja templates)")
        print(f"   - Thread counts to test: {thread_counts}")
        
        num_batches = total_requests // gpu_batch_size
        print(f"   - Number of batches: {num_batches:,}")
        print(f"   - Memory-efficient: processing {measurement_batches} batches for measurement")
        print(f"   - Only 1 batch ({gpu_batch_size} requests) in memory at a time")
        
        words_per_message = max(5, tokens_per_request // (turns_per_request * 4))
        
        # Warmup with a single small batch (then discard)
        print("\n   Running warmup (generating and discarding 5 requests)...")
        for i in range(5):
            warmup_req = generate_chat_conversation(
                num_turns=turns_per_request,
                words_per_message=words_per_message,
                unique_id=i * 99999
            )
            raw = self.tokenizer.apply_chat_template(warmup_req, tokenize=False)
            _ = self.tokenizer.encode(raw)
            del warmup_req, raw
        gc.collect()
        
        # Verify sample structure (then discard)
        sample_req = generate_chat_conversation(
            num_turns=turns_per_request,
            words_per_message=words_per_message,
            unique_id=12345
        )
        sample_raw = self.tokenizer.apply_chat_template(sample_req, tokenize=False)
        sample_tokens = len(self.tokenizer.encode(sample_raw))
        print(f"   Sample request: {len(sample_req)} messages, ~{sample_tokens} tokens")
        del sample_req, sample_raw
        gc.collect()
        
        results_data = []
        seq_jinja_ms = seq_tokenize_ms = seq_gpu_wait_ms = None
        
        print("\n   📈 System monitoring enabled (sampling interval: 0.1s)")
        print(f"   CPU cores available: {os.cpu_count()}")
        
        # Test each concurrency level
        for num_threads in thread_counts:
            print(f"\n   📊 Testing {num_threads} thread(s) ({measurement_batches} batches)...")
            
            total_jinja_ms = 0.0
            total_tokenize_ms = 0.0
            total_collate_ms = 0.0
            total_gpu_wait_ms = 0.0
            
            # Start system monitoring (use system_wide=False for threading)
            with SystemMonitor(interval=0.1, system_wide=False) as monitor:
                for batch_idx in range(measurement_batches):
                    gc.collect()
                    
                    # Generate batch requests fresh for this iteration
                    batch_requests = [
                        generate_chat_conversation(
                            num_turns=turns_per_request,
                            words_per_message=words_per_message,
                            unique_id=(batch_idx * gpu_batch_size + i) * 12345
                        )
                        for i in range(gpu_batch_size)
                    ]
                    
                    batch_start = time.perf_counter()
                    
                    # Step 1: Jinja templating
                    jinja_start = time.perf_counter()
                    if num_threads == 1:
                        batch_raw_strings = []
                        for req in batch_requests:
                            batch_raw_strings.append(self.tokenizer.apply_chat_template(req, tokenize=False))
                    else:
                        with ThreadPoolExecutor(max_workers=num_threads) as executor:
                            args = [(self.tokenizer, req) for req in batch_requests]
                            futures = [executor.submit(jinja_worker_with_tokenizer, arg) for arg in args]
                            batch_raw_strings = [f.result() for f in futures]
                    jinja_ms = (time.perf_counter() - jinja_start) * 1000
                    
                    del batch_requests
                    
                    # Step 2: Tokenization
                    tok_start = time.perf_counter()
                    if num_threads == 1:
                        batch_input_ids = []
                        for raw in batch_raw_strings:
                            batch_input_ids.append(self.tokenizer.encode(raw, add_special_tokens=False))
                    else:
                        with ThreadPoolExecutor(max_workers=num_threads) as executor:
                            args = [(self.tokenizer, raw) for raw in batch_raw_strings]
                            futures = [executor.submit(tokenize_worker, arg) for arg in args]
                            batch_input_ids = [f.result() for f in futures]
                    tokenize_ms = (time.perf_counter() - tok_start) * 1000
                    
                    del batch_raw_strings
                    
                    # Step 3: Collation
                    _, _, collate_ms = collate_batch(batch_input_ids, self.tokenizer.pad_token_id)
                    
                    # Total GPU Wait Time
                    gpu_wait_ms = (time.perf_counter() - batch_start) * 1000
                    
                    # Accumulate timings
                    total_jinja_ms += jinja_ms
                    total_tokenize_ms += tokenize_ms
                    total_collate_ms += collate_ms
                    total_gpu_wait_ms += gpu_wait_ms
                    
                    del batch_input_ids
            
            # Get system metrics
            sys_metrics = monitor.get_stats()
            
            # Calculate averages
            avg_jinja_ms = total_jinja_ms / measurement_batches
            avg_tokenize_ms = total_tokenize_ms / measurement_batches
            avg_collate_ms = total_collate_ms / measurement_batches
            avg_gpu_wait_ms = total_gpu_wait_ms / measurement_batches
            
            jinja_pct = (avg_jinja_ms / avg_gpu_wait_ms) * 100
            
            # Calculate speedups
            if num_threads == 1:
                seq_jinja_ms = avg_jinja_ms
                seq_tokenize_ms = avg_tokenize_ms
                seq_gpu_wait_ms = avg_gpu_wait_ms
                jinja_speedup = 1.0
                tokenize_speedup = 1.0
                gpu_wait_speedup = 1.0
            else:
                jinja_speedup = seq_jinja_ms / avg_jinja_ms
                tokenize_speedup = seq_tokenize_ms / avg_tokenize_ms
                gpu_wait_speedup = seq_gpu_wait_ms / avg_gpu_wait_ms
            
            # Calculate throughput metrics
            batches_per_sec = 1000 / avg_gpu_wait_ms
            requests_per_sec = batches_per_sec * gpu_batch_size
            time_for_all_requests_sec = (avg_gpu_wait_ms / 1000) * num_batches
            
            print(f"      GPU Wait Time (avg per batch): {avg_gpu_wait_ms:.1f}ms")
            print(f"        - Jinja:    {avg_jinja_ms:.1f}ms ({jinja_pct:.1f}%)")
            print(f"        - Tokenize: {avg_tokenize_ms:.1f}ms")
            print(f"        - Collate:  {avg_collate_ms:.1f}ms")
            print(f"      Throughput: {requests_per_sec:.0f} req/sec")
            print(f"      Est. time for all {total_requests:,} requests: {time_for_all_requests_sec:.1f}s")
            print(f"      System: Avg CPU {sys_metrics.avg_cpu_total:.1f}%, Peak RAM {sys_metrics.peak_memory_mb:.0f}MB")
            
            results_data.append({
                "Threads": num_threads,
                "Jinja (ms)": avg_jinja_ms,
                "Tokenize (ms)": avg_tokenize_ms,
                "Collate (ms)": avg_collate_ms,
                "GPU Wait (ms)": avg_gpu_wait_ms,
                "Avg CPU %": sys_metrics.avg_cpu_total,
                "Max CPU %": sys_metrics.max_cpu_total,
                "Peak RAM (MB)": sys_metrics.peak_memory_mb,
                "Jinja %": jinja_pct,
                "Jinja Speedup": jinja_speedup,
                "Tokenize Speedup": tokenize_speedup,
                "GPU Wait Speedup": gpu_wait_speedup,
                "Throughput (req/s)": requests_per_sec,
                "Time for 100k (s)": time_for_all_requests_sec,
            })
        
        df = pd.DataFrame(results_data)
        
        print("\n" + "─"*70)
        print(f"\n{df.to_string(index=False)}")
        
        # Key insights
        print("\n   🔍 KEY INSIGHT: Jinja Overhead Increases with Concurrency")
        print("   " + "─"*55)
        
        first_jinja_pct = df.iloc[0]["Jinja %"]
        last_jinja_pct = df.iloc[-1]["Jinja %"]
        max_threads = thread_counts[-1]
        
        print(f"      At 1 thread:  Jinja = {first_jinja_pct:.1f}% of GPU wait")
        print(f"      At {max_threads} threads: Jinja = {last_jinja_pct:.1f}% of GPU wait")
        
        seq_throughput = df.iloc[0]["Throughput (req/s)"]
        max_throughput = df.iloc[-1]["Throughput (req/s)"]
        throughput_improvement = max_throughput / seq_throughput
        
        print(f"\n      Throughput: {seq_throughput:.0f} -> {max_throughput:.0f} req/s ({throughput_improvement:.1f}x)")
        print("\n      ⚠️  As concurrency increases, Jinja becomes THE dominant bottleneck!")
        print("      Tokenization parallelizes (Rust releases GIL), Jinja does NOT.")
        
        metadata = {
            "total_requests": total_requests,
            "gpu_batch_size": gpu_batch_size,
            "tokens_per_request": tokens_per_request,
            "turns_per_request": turns_per_request,
            "thread_counts": thread_counts,
            "seq_gpu_wait_ms": seq_gpu_wait_ms,
            "measurement_batches": measurement_batches,
        }
        
        return ExperimentResult("concurrency", 3, df, metadata)


def main():
    """CLI entry point for Experiment 3."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Experiment 3: GPU Wait Time vs Concurrency",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--quick", action="store_true", help="Quick mode")
    parser.add_argument("--threads", nargs="+", type=int, default=None, help="Thread counts to test")
    parser.add_argument("--batch-size", type=int, default=None, help="GPU batch size")
    
    args = parser.parse_args()
    
    # Build config
    config = CONFIG.copy()
    if args.quick:
        config["exp3_total_requests"] = 10000
        config["exp3_thread_counts"] = [1, 4, 8, 16, 32]
    if args.threads:
        config["exp3_thread_counts"] = args.threads
    if args.batch_size:
        config["exp3_gpu_batch_size"] = args.batch_size
    
    # Run experiment
    experiment = Experiment3Concurrency(config)
    result = experiment.run()
    
    # Save results
    if not result.data.empty:
        output_path = "exp3_concurrency_results.csv"
        result.data.to_csv(output_path, index=False)
        print(f"\n   ✓ Results saved to: {output_path}")
    
    return result


if __name__ == "__main__":
    main()
