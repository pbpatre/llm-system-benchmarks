#!/usr/bin/env python3
"""
===================================================================================
Experiment 2: The "Chat History" Tax (Scaling)
===================================================================================

This experiment demonstrates a crucial insight: Jinja template processing time
scales with the NUMBER OF MESSAGES, not the token count.

Methodology:
- Keep total tokens constant (~100k tokens)
- Vary the number of conversation turns (1, 10, 25, 50, 75, 100)
- Measure Jinja and tokenization time separately

Key Insight: 
As the number of turns increases, Jinja processing time increases proportionally
because it must iterate through each message in the template. However, tokenization
time stays relatively flat because the total token count is constant.

This is the "Chat History Tax" - longer conversations with more back-and-forth
messages incur increasing Jinja overhead even if the total content length is similar.

Real-world Impact:
- Single-shot prompts: Low Jinja overhead
- Multi-turn chat (10+ turns): Significant Jinja overhead
- Long conversations (100+ turns): Jinja can dominate preprocessing time

Usage:
    python exp2_scaling.py                    # Run with default config
    python exp2_scaling.py --quick            # Quick mode (~2 min)
    python exp2_scaling.py --turns 1 25 50    # Custom turn counts
"""

import os
import gc
import time
import warnings
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
from transformers import AutoTokenizer, logging as hf_logging

# Suppress HuggingFace warnings
hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=FutureWarning)

# Import from common modules
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import CONFIG, ExperimentResult, generate_fixed_token_conversation


class Experiment2Scaling:
    """
    Experiment 2: Show Jinja slowdown with conversation turns (fixed token count).
    
    KEY INSIGHT: Jinja template processing time scales with the NUMBER OF MESSAGES,
    not the token count. This experiment keeps total tokens constant (~100k) while
    varying turns to isolate Jinja's O(messages) complexity from tokenization's
    O(tokens) complexity.
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
        Run the scaling experiment.
        
        Uses multiple conversation samples and many iterations for statistical validity.
        
        Returns:
            ExperimentResult containing scaling analysis data
        """
        print("\n" + "="*70)
        print("EXPERIMENT 2: The 'Chat History' Tax (Fixed Token Count)")
        print("="*70)
        
        if not self._load_tokenizer():
            return ExperimentResult("scaling", 2, pd.DataFrame(), {"error": "Failed to load tokenizer"})
        
        turn_counts = self.config["turn_counts"]
        target_tokens = self.config.get("fixed_total_tokens", 100000)
        num_samples = self.config.get("exp2_num_samples", 5)
        num_iterations = self.config.get("exp2_iterations", 30)
        
        print(f"   Keeping total tokens ~constant (~{target_tokens:,}), increasing turns")
        print(f"   Statistical rigor: {num_samples} samples × {num_iterations} iterations = {num_samples * num_iterations} measurements per turn count")
        
        results_data = []
        
        for num_turns in turn_counts:
            print(f"\n   Testing {num_turns} turns...")
            
            all_jinja_times = []
            all_tokenize_times = []
            actual_tokens_list = []
            
            # Test multiple different conversations for this turn count
            for sample_idx in range(num_samples):
                # Generate a unique conversation with fixed token count
                messages = generate_fixed_token_conversation(
                    self.tokenizer, 
                    num_turns=num_turns,
                    target_tokens=target_tokens,
                    unique_id=num_turns * 10000 + sample_idx * 1000
                )
                
                # Warmup for this sample
                for _ in range(2):
                    raw = self.tokenizer.apply_chat_template(messages, tokenize=False)
                    _ = self.tokenizer.encode(raw)
                
                # Measure actual tokens
                raw_str = self.tokenizer.apply_chat_template(messages, tokenize=False)
                actual_tokens_list.append(len(self.tokenizer.encode(raw_str)))
                
                # Run multiple iterations for this sample
                for _ in range(num_iterations):
                    gc.collect()
                    
                    start = time.perf_counter()
                    raw_str = self.tokenizer.apply_chat_template(messages, tokenize=False)
                    all_jinja_times.append((time.perf_counter() - start) * 1000)
                    
                    start = time.perf_counter()
                    _ = self.tokenizer.encode(raw_str, add_special_tokens=False)
                    all_tokenize_times.append((time.perf_counter() - start) * 1000)
            
            # Calculate statistics across all samples and iterations
            jinja_mean = np.mean(all_jinja_times)
            jinja_std = np.std(all_jinja_times)
            jinja_sem = jinja_std / np.sqrt(len(all_jinja_times))
            
            tokenize_mean = np.mean(all_tokenize_times)
            tokenize_std = np.std(all_tokenize_times)
            tokenize_sem = tokenize_std / np.sqrt(len(all_tokenize_times))
            
            total_mean = jinja_mean + tokenize_mean
            actual_tokens_mean = np.mean(actual_tokens_list)
            
            # Calculate 95% confidence interval
            jinja_ci95 = 1.96 * jinja_sem
            tokenize_ci95 = 1.96 * tokenize_sem
            
            print(f"      Jinja: {jinja_mean:.3f} ± {jinja_ci95:.3f} ms (95% CI)")
            print(f"      Tokenize: {tokenize_mean:.3f} ± {tokenize_ci95:.3f} ms (95% CI)")
            print(f"      Tokens: ~{actual_tokens_mean:.0f}")
            
            results_data.append({
                "Turns": num_turns,
                "Messages": len(messages),
                "Actual Tokens": int(actual_tokens_mean),
                "Jinja (ms)": jinja_mean,
                "Jinja Std (ms)": jinja_std,
                "Jinja CI95 (ms)": jinja_ci95,
                "Tokenize (ms)": tokenize_mean,
                "Tokenize Std (ms)": tokenize_std,
                "Tokenize CI95 (ms)": tokenize_ci95,
                "Total (ms)": total_mean,
                "Jinja %": (jinja_mean / total_mean) * 100,
                "N Samples": len(all_jinja_times),
            })
        
        df = pd.DataFrame(results_data)
        
        # Calculate scaling factors relative to baseline
        baseline_jinja = df[df["Turns"] == turn_counts[0]]["Jinja (ms)"].values[0]
        baseline_tokenize = df[df["Turns"] == turn_counts[0]]["Tokenize (ms)"].values[0]
        df["Jinja Scaling"] = df["Jinja (ms)"] / baseline_jinja
        df["Tokenize Scaling"] = df["Tokenize (ms)"] / baseline_tokenize
        
        # Display key columns
        display_cols = ["Turns", "Actual Tokens", "Jinja (ms)", "Jinja CI95 (ms)", 
                       "Tokenize (ms)", "Tokenize CI95 (ms)", "Jinja Scaling", "Tokenize Scaling"]
        print(f"\n{df[display_cols].to_string(index=False)}")
        
        # Key insight summary
        print("\n   🔍 KEY INSIGHT: Jinja Scales with TURNS (not tokens)")
        print("   " + "─"*60)
        
        max_turns = turn_counts[-1]
        jinja_scaling = df[df["Turns"] == max_turns]["Jinja Scaling"].values[0]
        tokenize_scaling = df[df["Turns"] == max_turns]["Tokenize Scaling"].values[0]
        
        print(f"      Token count stays ~constant (~{target_tokens:,})")
        print(f"      At {max_turns} turns vs {turn_counts[0]} turn:")
        print(f"        - Turn increase:     {max_turns / turn_counts[0]:.1f}x")
        print(f"        - Jinja scaling:     {jinja_scaling:.2f}x  ← Scales with turns!")
        print(f"        - Tokenize scaling:  {tokenize_scaling:.2f}x  ← Stays ~constant (same tokens)")
        print(f"\n      Total measurements per turn count: {num_samples * num_iterations}")
        
        metadata = {
            "turn_counts": turn_counts, 
            "target_tokens": target_tokens,
            "num_samples": num_samples,
            "num_iterations": num_iterations,
        }
        
        return ExperimentResult("scaling", 2, df, metadata)


def main():
    """CLI entry point for Experiment 2."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Experiment 2: The 'Chat History' Tax (Scaling)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--quick", action="store_true", help="Quick mode with fewer samples")
    parser.add_argument("--turns", nargs="+", type=int, default=None, help="Turn counts to test")
    parser.add_argument("--tokens", type=int, default=None, help="Target token count")
    
    args = parser.parse_args()
    
    # Build config
    config = CONFIG.copy()
    if args.quick:
        config["turn_counts"] = [1, 25, 50, 100]
        config["exp2_num_samples"] = 3
        config["exp2_iterations"] = 15
        config["fixed_total_tokens"] = 50000
    if args.turns:
        config["turn_counts"] = args.turns
    if args.tokens:
        config["fixed_total_tokens"] = args.tokens
    
    # Run experiment
    experiment = Experiment2Scaling(config)
    result = experiment.run()
    
    # Save results
    if not result.data.empty:
        output_path = "exp2_scaling_results.csv"
        result.data.to_csv(output_path, index=False)
        print(f"\n   ✓ Results saved to: {output_path}")
    
    return result


if __name__ == "__main__":
    main()
