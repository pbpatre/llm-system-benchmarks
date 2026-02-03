"""
Configuration for LLM Pre-Processing Benchmark Suite.

This module contains all configurable parameters for the benchmark experiments.
Modify these values to adjust the benchmark behavior for different scenarios.
"""

# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================

CONFIG = {
    # Model Configuration
    "model_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "model_name": "Llama-3.1-8B-Instruct",
    
    # Benchmark Parameters
    "warmup_runs": 3,
    "num_iterations": 10,
    
    # Token/Context Sizes - Production realistic chat (~1000 tokens per request)
    "target_tokens": 1000,  # ~1k tokens for baseline (typical chat)
    "fixed_total_tokens": 100000,  # For experiment 2 - large enough that template overhead is negligible
    
    # Experiment 2: Scaling Test (show Jinja overhead with turns)
    "turn_counts": [1, 10, 25, 50, 75, 100],
    "exp2_num_samples": 5,  # Different conversations per turn count
    "exp2_iterations": 30,  # Iterations per sample (total = samples × iterations)
    
    # Experiment 3: High-Throughput Concurrency Test
    # Production systems process MANY concurrent requests
    "exp3_total_requests": 100000,  # Total requests to simulate (production-like)
    "exp3_gpu_batch_size": 64,  # GPU processes in batches (typical: 32, 64, 128)
    "exp3_tokens_per_request": 1000,  # ~1k tokens per request
    "exp3_turns_per_request": 100,  # Complex chat history (100 turns = lots of Jinja work)
    "exp3_thread_counts": [1, 4, 8, 16, 32, 64],  # Test many concurrency levels
    
    # Experiment 4 & 5: Threading/Multiprocessing Test
    "thread_counts": [1, 2, 4, 8, 16, 32],
    "process_counts": [1, 2, 4, 8, 16],
    "scaling_total_requests": 10000,  # Total requests for scaling tests
    "scaling_batch_size": 64,  # GPU batch size
    "scaling_tokens_per_request": 1000,  # ~1k tokens per request
    "scaling_turns_per_request": 100,  # 100 turns (complex Jinja)
    
    # Output Configuration
    "output_csv": "results.csv",
    "plot_dpi": 150,
}


def get_quick_config() -> dict:
    """
    Return a modified config for quick benchmark runs (~5 min).
    
    Reduces the number of iterations, samples, and requests while
    maintaining statistically meaningful results.
    """
    quick = CONFIG.copy()
    quick.update({
        "warmup_runs": 2,
        "num_iterations": 3,
        "target_tokens": 1000,  # 1k tokens per request
        "fixed_total_tokens": 50000,  # For exp2 - 50k ensures template overhead is negligible
        "turn_counts": [1, 25, 50, 100],
        "exp2_num_samples": 3,  # Fewer samples in quick mode
        "exp2_iterations": 15,  # Fewer iterations in quick mode
        # Exp 3: High-throughput simulation (reduced for quick mode)
        "exp3_total_requests": 10000,  # 10k total requests
        "exp3_gpu_batch_size": 64,  # GPU batch size
        "exp3_tokens_per_request": 1000,  # 1k tokens
        "exp3_turns_per_request": 100,  # 100 turns (complex Jinja)
        "exp3_thread_counts": [1, 4, 8, 16, 32],
        # Exp 4/5: Scaling tests
        "scaling_total_requests": 5000,  # 5k total
        "scaling_batch_size": 64,  # GPU batch size
        "scaling_tokens_per_request": 1000,  # 1k tokens
        "scaling_turns_per_request": 100,  # 100 turns
        "thread_counts": [1, 2, 4, 8, 16],
        "process_counts": [1, 2, 4, 8],
    })
    return quick
