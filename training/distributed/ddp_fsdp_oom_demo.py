"""
DDP vs FSDP OOM Demonstration

This script demonstrates scenarios where:
1. DDP runs out of memory (OOM)
2. FSDP succeeds with the same configuration
3. Activation checkpointing can rescue DDP

GPU Memory Requirements:
  - H100 (80GB):  Use model_size=xl or xxl with batch_size=4-8
  - H200 (140GB): Use model_size=7b or 13b with batch_size=4-8
  
Run 'python check_gpu_info.py' to see your GPU memory capacity first!

Usage for H200 (140GB):
  # Try DDP (should OOM)
  torchrun --nproc-per-node=2 ddp_fsdp_oom_demo.py --mode=ddp --model_size=7b --batch_size=4
  
  # Try FSDP (should succeed)
  torchrun --nproc-per-node=2 ddp_fsdp_oom_demo.py --mode=fsdp --model_size=7b --batch_size=4
  
  # DDP with activation checkpointing (may succeed)
  torchrun --nproc-per-node=2 ddp_fsdp_oom_demo.py --mode=ddp --model_size=7b --batch_size=4 --use_activation_checkpointing
  
  # Find OOM boundary
  torchrun --nproc-per-node=2 ddp_fsdp_oom_demo.py --mode=ddp --model_size=7b --find_oom_limit

Memory Breakdown for Transformers:
  DDP per GPU:
    - Model weights: N params × 2 bytes (bf16) = N × 2
    - Gradients: N params × 2 bytes (bf16) = N × 2
    - Optimizer (Adam): N params × 12 bytes (fp32 master + 2 states) = N × 12
    - Activations: batch × seq × hidden × layers × ~16-32 bytes
    - TOTAL ≈ N × 16 + activations
  
  FSDP per GPU (2 GPUs):
    - Model weights: N/2 params × 2 bytes = N × 1
    - Gradients: N/2 params × 2 bytes = N × 1
    - Optimizer: N/2 params × 12 bytes = N × 6
    - Activations: same as DDP (not sharded)
    - TOTAL ≈ N × 8 + activations
    → 50% memory savings on model/grad/optim!
"""

import os
import sys
import time
import math
import argparse
import functools
from contextlib import nullcontext
from datetime import timedelta

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from transformers import GPT2Config, GPT2LMHeadModel
from dataclasses import dataclass

# Profiling imports (optional)
try:
    from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler
    PROFILER_AVAILABLE = True
except ImportError:
    PROFILER_AVAILABLE = False


# --- Model Size Configurations ---
MODEL_CONFIGS = {
    "tiny": {
        "n_layer": 4,
        "n_embd": 512,
        "n_head": 8,
        "params": "~26M",
    },
    "small": {
        "n_layer": 12,
        "n_embd": 768,
        "n_head": 12,
        "params": "124M",
    },
    "medium": {
        "n_layer": 24,
        "n_embd": 1024,
        "n_head": 16,
        "params": "350M",
    },
    "large": {
        "n_layer": 36,
        "n_embd": 1280,
        "n_head": 20,
        "params": "774M",
    },
    "xl": {
        "n_layer": 48,
        "n_embd": 1600,
        "n_head": 25,
        "params": "1.5B",
    },
    "xxl": {
        "n_layer": 60,
        "n_embd": 2048,
        "n_head": 32,
        "params": "3B",
    },
    "7b": {
        "n_layer": 32,
        "n_embd": 4096,
        "n_head": 32,
        "params": "7B",
    },
    "13b": {
        "n_layer": 40,
        "n_embd": 5120,
        "n_head": 40,
        "params": "13B",
    },
}


@dataclass
class Config:
    # Model
    model_size: str = "large"
    vocab_size: int = 50257
    seq_len: int = 2048
    
    # Training
    batch_size: int = 4  # per GPU
    grad_accum_steps: int = 1
    max_steps: int = 10
    
    # Optimization
    use_activation_checkpointing: bool = False
    use_flash_attention: bool = False  # Requires flash-attn package
    
    # Profiling
    enable_profiling: bool = False
    profile_dir: str = "./output/profiling_traces"
    profile_wait_steps: int = 2
    profile_warmup_steps: int = 2
    profile_active_steps: int = 2
    
    # Hardware
    dtype: torch.dtype = torch.bfloat16
    seed: int = 42


class SyntheticDataset(Dataset):
    """Random tokens for benchmarking."""
    def __init__(self, vocab_size: int, seq_len: int, length: int = 1000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.length = length
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        data = torch.randint(0, self.vocab_size, (self.seq_len + 1,), dtype=torch.long)
        return data[:-1], data[1:]


def setup_distributed():
    """Initialize distributed process group."""
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    if world_size > 1:
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
            dist.init_process_group("nccl", timeout=timedelta(minutes=10))
        else:
            device = torch.device("cpu")
            dist.init_process_group("gloo", timeout=timedelta(minutes=10))
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
    
    return rank, world_size, local_rank, device


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def get_model(config: Config):
    """Create GPT-2 model based on size config."""
    model_cfg = MODEL_CONFIGS[config.model_size]
    
    gpt_config = GPT2Config(
        vocab_size=config.vocab_size,
        n_positions=config.seq_len,
        n_embd=model_cfg["n_embd"],
        n_layer=model_cfg["n_layer"],
        n_head=model_cfg["n_head"],
        use_cache=False,
    )
    
    model = GPT2LMHeadModel(gpt_config)
    return model, model_cfg


def apply_activation_checkpointing(model):
    """
    Apply gradient checkpointing to transformer blocks.
    Trades compute for memory: recomputes activations during backward pass.
    
    Handles FSDP-wrapped, DDP-wrapped, and unwrapped models.
    Uses non-reentrant checkpointing for FSDP compatibility.
    """
    # Unwrap the model to get the actual transformer
    actual_model = model
    
    # Handle FSDP-wrapped models
    if hasattr(model, '_fsdp_wrapped_module'):
        actual_model = model._fsdp_wrapped_module
    # Handle DDP-wrapped models
    elif hasattr(model, 'module'):
        actual_model = model.module
    
    # Apply gradient checkpointing to the actual model
    # IMPORTANT: use_reentrant=False is required for FSDP compatibility
    if hasattr(actual_model, 'transformer'):
        actual_model.transformer.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        return True
    
    return False


def estimate_memory_footprint(config: Config, world_size: int, use_fsdp: bool):
    """
    Estimate memory footprint for DDP vs FSDP.
    
    Memory components:
    1. Model parameters (weights)
    2. Gradients (same shape as weights)
    3. Optimizer states (Adam: master weights + momentum + variance)
    4. Activations (depends on batch size, sequence length, hidden size)
    """
    model_cfg = MODEL_CONFIGS[config.model_size]
    
    # Approximate parameter count (simplified)
    n_layer = model_cfg["n_embd"]
    d_model = model_cfg["n_embd"]
    vocab_size = config.vocab_size
    
    # Transformer blocks: attention + MLP
    # Attention: 4 × d_model × d_model (Q, K, V, O projections)
    # MLP: 2 × d_model × (4 × d_model) = 8 × d_model²
    # Total per layer: 12 × d_model²
    params_per_layer = 12 * d_model * d_model
    params_transformer = model_cfg["n_layer"] * params_per_layer
    
    # Embeddings: vocab_size × d_model + position embeddings
    params_embed = vocab_size * d_model + config.seq_len * d_model
    
    # LM head: d_model × vocab_size
    params_head = d_model * vocab_size
    
    total_params = params_transformer + params_embed + params_head
    
    # Bytes per parameter
    param_bytes = 2 if config.dtype in [torch.float16, torch.bfloat16] else 4
    grad_bytes = 2  # Usually match param dtype
    optim_bytes = 12  # Adam: fp32 master (4) + momentum (4) + variance (4)
    
    if use_fsdp:
        # Sharded across GPUs
        params_per_gpu = total_params / world_size
        model_memory = params_per_gpu * param_bytes
        grad_memory = params_per_gpu * grad_bytes
        optim_memory = params_per_gpu * optim_bytes
    else:
        # Full replication
        model_memory = total_params * param_bytes
        grad_memory = total_params * grad_bytes
        optim_memory = total_params * optim_bytes
    
    # Activations (rough estimate, highly dependent on implementation)
    # For transformer: batch × seq × hidden × num_layers × multiplier
    # Multiplier includes intermediate activations, attention scores, etc.
    activation_multiplier = 32 if not config.use_activation_checkpointing else 4
    activation_memory = (
        config.batch_size * 
        config.seq_len * 
        d_model * 
        model_cfg["n_layer"] * 
        activation_multiplier
    )
    
    total_memory = model_memory + grad_memory + optim_memory + activation_memory
    
    return {
        "total_params": total_params,
        "model_memory_gb": model_memory / 1e9,
        "grad_memory_gb": grad_memory / 1e9,
        "optim_memory_gb": optim_memory / 1e9,
        "activation_memory_gb": activation_memory / 1e9,
        "total_memory_gb": total_memory / 1e9,
    }


def create_profiler(config: Config, rank: int, mode: str, checkpoint_suffix: str = ""):
    """Create PyTorch profiler for communication analysis."""
    if not PROFILER_AVAILABLE:
        print("Warning: torch.profiler not available. Profiling disabled.")
        return None
    
    # Determine experiment name
    exp_name = f"{mode}_{'with' if config.use_activation_checkpointing else 'no'}_checkpoint{checkpoint_suffix}"
    
    # Detect NCCL configuration
    p2p_disabled = os.environ.get("NCCL_P2P_DISABLE", "0") == "1"
    shm_disabled = os.environ.get("NCCL_SHM_DISABLE", "0") == "1"
    
    if p2p_disabled and shm_disabled:
        network_suffix = "_pcie_bottleneck"
    else:
        network_suffix = "_nvlink_baseline"
    
    exp_name = exp_name + network_suffix
    
    trace_dir = os.path.join(config.profile_dir, exp_name)
    os.makedirs(trace_dir, exist_ok=True)
    
    schedule = torch.profiler.schedule(
        wait=config.profile_wait_steps,
        warmup=config.profile_warmup_steps,
        active=config.profile_active_steps,
        repeat=1,
    )
    
    def trace_handler(prof):
        # Save TensorBoard traces
        tensorboard_trace_handler(trace_dir)(prof)
        
        if rank == 0:
            print(f"\nTraces exported to: {trace_dir}")
            
            # Calculate overlap metrics
            try:
                from ..helper.profiler_analyzer import FSDPOverlapProfiler
                analyzer = FSDPOverlapProfiler(prof)
                metrics = analyzer.calculate_overlap()
                metrics.print_summary()
                
                # Export metrics
                metrics_path = os.path.join(trace_dir, "overlap_metrics.json")
                analyzer.export_metrics(metrics_path)
            except Exception as e:
                print(f"Note: Could not calculate overlap metrics: {e}")
    
    return profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule,
        on_trace_ready=trace_handler,
        record_shapes=True,
        with_stack=True,
        profile_memory=True,
    )


def print_memory_estimate(config: Config, world_size: int, use_fsdp: bool, rank: int):
    """Print memory estimation."""
    if rank != 0:
        return
    
    mode = "FSDP" if use_fsdp else "DDP"
    est = estimate_memory_footprint(config, world_size, use_fsdp)
    
    print("\n" + "=" * 70)
    print(f"MEMORY ESTIMATION - {mode}")
    print("=" * 70)
    print(f"Model: {config.model_size} ({MODEL_CONFIGS[config.model_size]['params']} params)")
    print(f"Actual params: {est['total_params']:,}")
    print(f"Batch size per GPU: {config.batch_size}")
    print(f"Sequence length: {config.seq_len}")
    print(f"World size: {world_size}")
    print(f"Activation checkpointing: {config.use_activation_checkpointing}")
    print()
    print(f"Memory breakdown (per GPU):")
    print(f"  Model weights:     {est['model_memory_gb']:>6.2f} GB")
    print(f"  Gradients:         {est['grad_memory_gb']:>6.2f} GB")
    print(f"  Optimizer states:  {est['optim_memory_gb']:>6.2f} GB")
    print(f"  Activations:       {est['activation_memory_gb']:>6.2f} GB")
    print(f"  {'─' * 35}")
    print(f"  TOTAL (estimated): {est['total_memory_gb']:>6.2f} GB")
    print("=" * 70)
    
    if est['total_memory_gb'] > 75:
        print("⚠️  WARNING: Estimated memory exceeds 75GB - may OOM on 80GB GPU!")
    elif est['total_memory_gb'] > 60:
        print("⚠️  CAUTION: Estimated memory is high - monitor closely")
    else:
        print("✓  Estimated memory should fit comfortably")
    print()


def run_training(config: Config, use_fsdp: bool):
    """Run training with either DDP or FSDP."""
    rank, world_size, local_rank, device = setup_distributed()
    is_main = (rank == 0)
    mode = "FSDP" if use_fsdp else "DDP"
    
    # Seed
    torch.manual_seed(config.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed + rank)
    
    if is_main:
        print(f"\n{'=' * 70}")
        print(f"Training with {mode}")
        print(f"{'=' * 70}")
        print(f"Device: {device}")
        print(f"World size: {world_size}")
        print(f"Model size: {config.model_size}")
    
    # Print memory estimate
    print_memory_estimate(config, world_size, use_fsdp, rank)
    
    # Create dataset
    dataset = SyntheticDataset(config.vocab_size, config.seq_len)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, sampler=sampler, pin_memory=True)
    
    # Create model
    if is_main:
        print(f"Creating {config.model_size} model...")
    
    model, model_cfg = get_model(config)
    n_params = sum(p.numel() for p in model.parameters())
    
    if is_main:
        print(f"Model parameters: {n_params / 1e6:.1f}M")
    
    # Move to device
    model = model.to(device)
    
    # Reset memory stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    
    # Wrap in DDP or FSDP
    if use_fsdp:
        auto_wrap_policy = functools.partial(
            size_based_auto_wrap_policy,
            min_num_params=20_000,
        )
        
        mp_policy = None
        if config.dtype in [torch.bfloat16, torch.float16]:
            mp_policy = MixedPrecision(
                param_dtype=config.dtype,
                reduce_dtype=config.dtype,
                buffer_dtype=config.dtype,
            )
        
        model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mp_policy,
            device_id=device,
        )
        
        if is_main:
            print(f"Model wrapped in FSDP")
    else:
        model = DDP(model, device_ids=[local_rank])
        if is_main:
            print(f"Model wrapped in DDP")
    
    # Apply activation checkpointing if requested
    if config.use_activation_checkpointing:
        if is_main:
            print("Applying activation checkpointing...")
        success = apply_activation_checkpointing(model)
        if is_main:
            if success:
                print("  ✓ Activation checkpointing enabled successfully")
            else:
                print("  ✗ Warning: Could not enable activation checkpointing")

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=0.1,
        betas=(0.9, 0.95),
        fused=torch.cuda.is_available(),
    )
    
    # Create profiler if enabled
    profiler = None
    if config.enable_profiling:
        if is_main:
            print(f"\nProfiling enabled")
        profiler = create_profiler(config, rank, mode)
        if profiler is None and is_main:
            print("Warning: Profiler creation failed, continuing without profiling")
    
    # Training loop
    model.train()
    
    if is_main:
        print(f"\nStarting training for {config.max_steps} steps...")
    
    step = 0
    t0 = time.time()
    
    # Profiler context
    profiler_ctx = profiler if profiler else nullcontext()
    
    try:
        with profiler_ctx:
            for epoch in range(100):  # Arbitrary large number
                sampler.set_epoch(epoch)
                
                for batch_idx, (x, y) in enumerate(dataloader):
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    
                    # Training step with profiler markers
                    if config.enable_profiling and profiler:
                        record_fn = record_function
                    else:
                        record_fn = lambda name: nullcontext()
                    
                    with record_fn("## TRAIN_STEP ##"):
                        # Forward
                        with record_fn("FORWARD"):
                            with torch.autocast(device_type=device.type, dtype=config.dtype):
                                outputs = model(x, labels=y)
                                loss = outputs.loss / config.grad_accum_steps
                        
                        # Backward
                        with record_fn("BACKWARD"):
                            loss.backward()
                        
                        # Step
                        if (batch_idx + 1) % config.grad_accum_steps == 0:
                            with record_fn("OPTIMIZER_STEP"):
                                optimizer.step()
                                optimizer.zero_grad(set_to_none=True)
                            
                            step += 1
                            
                            # Profiler step
                            if profiler:
                                profiler.step()
                            
                            # Log
                            if step % 5 == 0 and is_main:
                                if torch.cuda.is_available():
                                    allocated = torch.cuda.memory_allocated() / 1e9
                                    reserved = torch.cuda.memory_reserved() / 1e9
                                    peak = torch.cuda.max_memory_allocated() / 1e9
                                else:
                                    allocated = reserved = peak = 0
                                
                                t1 = time.time()
                                tokens_per_sec = (config.batch_size * config.seq_len * world_size * 5) / (t1 - t0)
                                
                                phase = "normal"
                                if profiler and config.enable_profiling:
                                    if step <= config.profile_wait_steps:
                                        phase = "wait"
                                    elif step <= config.profile_wait_steps + config.profile_warmup_steps:
                                        phase = "warmup"
                                    else:
                                        phase = "active"
                                
                                print(f"Step {step:>3d} ({phase:7s}) | Loss: {loss.item():.4f} | "
                                      f"Mem: {allocated:.1f}GB alloc, {peak:.1f}GB peak | "
                                      f"TPS: {tokens_per_sec:,.0f}")
                                
                                t0 = time.time()
                            
                            if step >= config.max_steps:
                                break
                
                if step >= config.max_steps:
                    break
        
        # Success!
        if is_main:
            if torch.cuda.is_available():
                final_peak = torch.cuda.max_memory_allocated() / 1e9
                final_reserved = torch.cuda.memory_reserved() / 1e9
            else:
                final_peak = final_reserved = 0
            
            print(f"\n{'=' * 70}")
            print(f"✅ SUCCESS - {mode}")
            print(f"{'=' * 70}")
            print(f"Completed {config.max_steps} steps")
            print(f"Peak GPU memory allocated: {final_peak:.2f} GB")
            print(f"GPU memory reserved: {final_reserved:.2f} GB")
            print(f"{'=' * 70}\n")
            
            return {
                "success": True,
                "peak_memory_gb": final_peak,
                "mode": mode,
            }
    
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            if is_main:
                if torch.cuda.is_available():
                    peak = torch.cuda.max_memory_allocated() / 1e9
                else:
                    peak = 0
                
                print(f"\n{'=' * 70}")
                print(f"❌ OOM - {mode}")
                print(f"{'=' * 70}")
                print(f"Out of memory at step {step}")
                print(f"Peak GPU memory before OOM: {peak:.2f} GB")
                print(f"Error: {str(e)}")
                print(f"{'=' * 70}\n")
            
            return {
                "success": False,
                "peak_memory_gb": peak if torch.cuda.is_available() else 0,
                "mode": mode,
                "error": str(e),
            }
        else:
            raise
    
    finally:
        cleanup()


def main():
    parser = argparse.ArgumentParser(description="DDP vs FSDP OOM Demo")
    parser.add_argument("--mode", type=str, choices=["ddp", "fsdp"], default="ddp",
                       help="Training mode: ddp or fsdp")
    parser.add_argument("--model_size", type=str, 
                       choices=list(MODEL_CONFIGS.keys()), 
                       default="large",
                       help="Model size configuration")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size per GPU")
    parser.add_argument("--seq_len", type=int, default=2048,
                       help="Sequence length")
    parser.add_argument("--max_steps", type=int, default=10,
                       help="Number of training steps")
    parser.add_argument("--use_activation_checkpointing", action="store_true",
                       help="Enable activation checkpointing (gradient checkpointing)")
    parser.add_argument("--find_oom_limit", action="store_true",
                       help="Binary search to find OOM boundary")
    parser.add_argument("--profile", action="store_true",
                       help="Enable profiling (communication and compute analysis)")
    
    args = parser.parse_args()
    
    config = Config(
        model_size=args.model_size,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        max_steps=args.max_steps,
        use_activation_checkpointing=args.use_activation_checkpointing,
        enable_profiling=args.profile,
    )
    
    use_fsdp = (args.mode == "fsdp")
    
    if args.find_oom_limit:
        # Binary search for OOM boundary
        rank = int(os.environ.get("RANK", 0))
        if rank == 0:
            print("\n🔍 Finding OOM boundary via binary search...")
            print("This will run multiple experiments with increasing batch sizes.\n")
        
        # Start with small batch, double until OOM
        batch_sizes = [1, 2, 4, 8, 16, 32, 64]
        last_success = None
        
        for bs in batch_sizes:
            config.batch_size = bs
            result = run_training(config, use_fsdp)
            
            if result["success"]:
                last_success = bs
            else:
                break
        
        if rank == 0:
            print(f"\n{'=' * 70}")
            print(f"OOM Boundary Results")
            print(f"{'=' * 70}")
            print(f"Mode: {args.mode.upper()}")
            print(f"Last successful batch size: {last_success}")
            print(f"First OOM batch size: {bs}")
            print(f"{'=' * 70}\n")
    else:
        # Single run
        run_training(config, use_fsdp)


if __name__ == "__main__":
    main()
