"""
Communication Profiling for FSDP/DDP Training

This script profiles distributed training to expose communication bottlenecks.
It uses torch.profiler to capture:
1. NCCL collective operations (all_gather, reduce_scatter, all_reduce)
2. Compute kernels (GEMMs, attention)
3. GPU idle time (communication stalls)

Usage:
  # Run A: NVLink Baseline (high-bandwidth GPU-to-GPU)
  torchrun --nproc_per_node=2 training/profile_distributed.py --mode=fsdp

  # Run B: Simulated Network Bottleneck (forces PCIe path)
  NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 torchrun --nproc_per_node=2 training/profile_distributed.py --mode=fsdp

  # Then analyze:
  tensorboard --logdir=./output/profiling_traces

Author: Systems Blog Series
"""

import os
import sys
import time
import json
import functools
from datetime import timedelta
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Optional

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

# --- Profiler imports ---
from torch.profiler import (
    profile,
    record_function,
    ProfilerActivity,
    tensorboard_trace_handler,
)


@dataclass
class ProfileConfig:
    """Configuration for profiling experiments."""
    
    # Model (smaller for fast profiling)
    batch_size: int = 4
    seq_len: int = 512
    d_model: int = 768
    n_layers: int = 6  # Fewer layers for faster profiling
    n_heads: int = 12
    vocab_size: int = 50257
    
    # Profiler schedule
    # wait=5: Skip first 5 steps (CUDA warmup, JIT compilation)
    # warmup=5: Trace but discard next 5 (stabilization)
    # active=2: Capture these 2 steps for analysis
    # repeat=1: Run once (total: 12 steps)
    wait_steps: int = 5
    warmup_steps: int = 5
    active_steps: int = 2
    total_steps: int = 12  # wait + warmup + active
    
    # Profiler features
    profile_dir: str = "./output/profiling_traces"
    record_shapes: bool = True       # Track tensor shapes (useful for comm analysis)
    with_stack: bool = True          # Python call stacks (heavier but insightful)
    profile_memory: bool = True      # Memory allocations timeline
    with_flops: bool = True          # FLOP estimates for compute kernels
    
    # Hardware
    dtype: torch.dtype = field(
        default_factory=lambda: torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    )
    
    # Experiment tracking
    experiment_name: str = "baseline"  # Will be set based on NCCL env vars


class SyntheticDataset(Dataset):
    """Random tokens for pure GPU profiling (no data loading noise)."""
    
    def __init__(self, vocab_size: int, seq_len: int, length: int = 10000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.length = length
    
    def __len__(self) -> int:
        return self.length
    
    def __getitem__(self, idx):
        data = torch.randint(0, self.vocab_size, (self.seq_len + 1,), dtype=torch.long)
        return data[:-1], data[1:]


def detect_nccl_config() -> str:
    """Detect NCCL configuration for naming the experiment."""
    p2p_disabled = os.environ.get("NCCL_P2P_DISABLE", "0") == "1"
    shm_disabled = os.environ.get("NCCL_SHM_DISABLE", "0") == "1"
    
    if p2p_disabled and shm_disabled:
        return "pcie_bottleneck"
    elif p2p_disabled:
        return "p2p_disabled"
    elif shm_disabled:
        return "shm_disabled"
    else:
        return "nvlink_baseline"


def setup_distributed():
    """Initialize distributed process group with proper device assignment."""
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    
    timeout = timedelta(minutes=5)
    
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        gpu_id = local_rank % num_gpus
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(device)
        
        if world_size > num_gpus:
            dist.init_process_group("gloo", timeout=timeout)
            if rank == 0:
                print(f"WARNING: {world_size} ranks but only {num_gpus} GPU(s). Using GLOO.")
        else:
            dist.init_process_group("nccl", timeout=timeout)
    else:
        device = torch.device("cpu")
        dist.init_process_group("gloo", timeout=timeout)
    
    return rank, world_size, local_rank, device


def get_model(cfg: ProfileConfig) -> nn.Module:
    """Initialize GPT-2 model from scratch."""
    model_config = GPT2Config(
        vocab_size=cfg.vocab_size,
        n_positions=cfg.seq_len,
        n_embd=cfg.d_model,
        n_layer=cfg.n_layers,
        n_head=cfg.n_heads,
        use_cache=False,
    )
    return GPT2LMHeadModel(model_config)


def wrap_model_fsdp(model: nn.Module, device: torch.device, cfg: ProfileConfig) -> FSDP:
    """Wrap model with FSDP for sharded training."""
    auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy,
        min_num_params=20_000,
    )
    
    mp_policy = None
    if cfg.dtype in (torch.bfloat16, torch.float16):
        mp_policy = MixedPrecision(
            param_dtype=cfg.dtype,
            reduce_dtype=cfg.dtype,
            buffer_dtype=cfg.dtype,
        )
    
    return FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mp_policy,
        device_id=device,
    )


def wrap_model_ddp(model: nn.Module, device: torch.device) -> DDP:
    """Wrap model with DDP for replicated training."""
    return DDP(model, device_ids=[device.index])


def create_profiler(cfg: ProfileConfig, rank: int, mode: str) -> profile:
    """
    Create a torch.profiler with optimal settings for communication analysis.
    
    Key design decisions:
    1. schedule: wait/warmup/active pattern avoids cold-start artifacts
    2. record_shapes: Essential for understanding tensor sizes in collectives
    3. with_stack: Adds ~10% overhead but invaluable for tracing comm calls
    4. profile_memory: Shows memory timeline (allocation/deallocation patterns)
    5. with_flops: Estimates compute intensity for overlap analysis
    """
    
    # Create output dir for experiment
    # TensorBoard expects traces directly in logdir/<run_name>/, not in subdirectories
    # The trace filename includes worker info, so we put all ranks in same directory
    trace_dir = os.path.join(
        cfg.profile_dir,
        f"{mode}_{cfg.experiment_name}"
    )
    os.makedirs(trace_dir, exist_ok=True)
    
    schedule = torch.profiler.schedule(
        wait=cfg.wait_steps,
        warmup=cfg.warmup_steps,
        active=cfg.active_steps,
        repeat=1,
    )
    
    # Custom trace handler that also logs summary stats
    def trace_handler(prof):        
        if rank == 0:
            # Print summary stats for quick analysis
            print("\n" + "=" * 70)
            print("PROFILER SUMMARY (Rank 0)")
            print("=" * 70)
            
            # CUDA/device time by kernel type (key_averages() returns FunctionEventAvg: use device_time_total)
            print("\nTop 10 CUDA Kernels by Total Time:")
            sort_by = "self_device_time_total"  # FunctionEventAvg uses this; "cuda_time_total" is FunctionEvent-only
            print(prof.key_averages().table(sort_by=sort_by, row_limit=10))
            
            # Communication operations (look for nccl*)
            print("\nNCCL Operations (Communication):")
            events = prof.key_averages()
            nccl_events = [e for e in events if "nccl" in e.key.lower()]
            def _device_time(e):
                return getattr(e, "device_time_total", None) or getattr(e, "cuda_time_total", 0)
            for e in sorted(nccl_events, key=lambda x: -_device_time(x)):
                t = _device_time(e)
                count = max(e.count, 1)
                print(f"  {e.key}: {t / 1000:.2f}ms (count={e.count}, avg={t / count / 1000:.2f}ms)")
            
        # First, save TensorBoard format (this enables PyTorch Profiler plugin in TensorBoard)
        # This handler exports .pt.trace.json files that TensorBoard's profiler plugin needs
        tb_handler = tensorboard_trace_handler(trace_dir)
        tb_handler(prof)
        
        if rank == 0:
            print(f"TensorBoard traces exported to: {trace_dir}")
            
            # Also export a standalone Chrome trace for chrome://tracing
            # Note: We can't call prof.export_chrome_trace() after tensorboard_trace_handler
            # because the trace has already been exported. The .pt.trace.json files ARE
            # Chrome traces - they can be loaded in chrome://tracing directly!
            
            # Export stacks for flame graph analysis
            if cfg.with_stack:
                stacks_path = os.path.join(trace_dir, "stacks.txt")
                try:
                    prof.export_stacks(stacks_path, "self_cuda_time_total")
                    print(f"Stacks exported: {stacks_path}")
                except:
                    pass  # Stack export sometimes fails, not critical
            
            print(f"\nHow to view traces:")
            print(f"  1. TensorBoard: tensorboard --logdir={cfg.profile_dir}")
            print(f"     Then go to: PyTorch Profiler tab")
            print(f"  2. Chrome trace: chrome://tracing")
            print(f"     Load any *.pt.trace.json file from {trace_dir}")
    
    return profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule,
        on_trace_ready=trace_handler,
        record_shapes=cfg.record_shapes,
        with_stack=cfg.with_stack,
        profile_memory=cfg.profile_memory,
        with_flops=cfg.with_flops,
    )


def train_step_with_markers(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> float:
    """
    Execute one training step with detailed profiler markers.
    
    The record_function markers create named regions in the trace,
    making it easy to identify:
    - Forward pass (where all_gather happens in FSDP)
    - Backward pass (where reduce_scatter happens in FSDP / all_reduce in DDP)
    - Optimizer step
    """
    
    with record_function("## TRAIN_STEP ##"):
        # Zero gradients with memory optimization
        with record_function("ZERO_GRAD"):
            optimizer.zero_grad(set_to_none=True)
        
        # Forward pass
        # FSDP: all_gather happens here (parameters are gathered per FSDP unit)
        with record_function("FORWARD"):
            with torch.autocast(device_type="cuda", dtype=dtype):
                outputs = model(x, labels=y)
                loss = outputs.loss
        
        # Backward pass  
        # FSDP: reduce_scatter happens here (gradients are scattered)
        # DDP: all_reduce happens here (gradients are averaged)
        with record_function("BACKWARD"):
            loss.backward()
        
        # Optimizer step
        with record_function("OPTIMIZER_STEP"):
            optimizer.step()
    
    return loss.item()


def run_profiled_training(mode: str):
    """
    Main profiling loop with comprehensive instrumentation.
    
    Args:
        mode: "fsdp" or "ddp"
    """
    rank, world_size, local_rank, device = setup_distributed()
    cfg = ProfileConfig()
    cfg.experiment_name = detect_nccl_config()
    
    is_main = (rank == 0)
    
    if is_main:
        print("=" * 70)
        print("COMMUNICATION PROFILING EXPERIMENT")
        print("=" * 70)
        print(f"Mode: {mode.upper()}")
        print(f"Experiment: {cfg.experiment_name}")
        print(f"World Size: {world_size}")
        print(f"Device: {device}")
        print(f"Precision: {cfg.dtype}")
        print(f"Model: GPT-2 ({cfg.n_layers} layers, {cfg.d_model} hidden)")
        print(f"Batch Size: {cfg.batch_size}")
        print(f"Sequence Length: {cfg.seq_len}")
        print()
        print("Profiler Schedule:")
        print(f"  wait={cfg.wait_steps} steps (skip cold start)")
        print(f"  warmup={cfg.warmup_steps} steps (stabilize)")
        print(f"  active={cfg.active_steps} steps (capture traces)")
        print(f"  total={cfg.total_steps} steps")
        print()
        print(f"NCCL Environment:")
        print(f"  NCCL_P2P_DISABLE={os.environ.get('NCCL_P2P_DISABLE', 'not set')}")
        print(f"  NCCL_SHM_DISABLE={os.environ.get('NCCL_SHM_DISABLE', 'not set')}")
        print("=" * 70)
    
    # Synchronize before starting
    dist.barrier()
    
    # Data
    dataset = SyntheticDataset(cfg.vocab_size, cfg.seq_len)
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=False
    )
    dataloader = DataLoader(
        dataset, batch_size=cfg.batch_size, sampler=sampler, num_workers=0
    )
    
    # Model
    model = get_model(cfg).to(device)
    
    if mode == "fsdp":
        model = wrap_model_fsdp(model, device, cfg)
        if is_main:
            print(f"Model wrapped with FSDP (sharded across {world_size} GPUs)")
    else:
        model = wrap_model_ddp(model, device)
        if is_main:
            print(f"Model wrapped with DDP (replicated on {world_size} GPUs)")
    
    n_params = sum(p.numel() for p in model.parameters())
    if is_main:
        print(f"Model Parameters: {n_params / 1e6:.1f}M")
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=3e-4, fused=True)
    
    # Create profiler
    profiler = create_profiler(cfg, rank, mode)
    
    # Training loop with profiling
    model.train()
    data_iter = iter(dataloader)
    
    if is_main:
        print("\nStarting profiled training...")
    
    torch.cuda.synchronize()
    t0 = time.time()
    
    with profiler:
        for step in range(cfg.total_steps):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                x, y = next(data_iter)
            
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            
            # Training step with markers
            loss = train_step_with_markers(model, optimizer, x, y, device, cfg.dtype)
            
            # CRITICAL: Tell profiler when step ends
            profiler.step()
            
            if is_main:
                phase = "wait" if step < cfg.wait_steps else \
                        "warmup" if step < cfg.wait_steps + cfg.warmup_steps else "active"
                print(f"Step {step:3d} | Phase: {phase:7s} | Loss: {loss:.4f}")
    
    torch.cuda.synchronize()
    total_time = time.time() - t0
    
    # Analyze overlap metrics (rank 0 only)
    if is_main:
        print(f"\nTotal time: {total_time:.2f}s")
        print(f"Average step time: {total_time / cfg.total_steps * 1000:.1f}ms")
        
        # Calculate overlap efficiency
        try:
            from ..helper.profiler_analyzer import FSDPOverlapProfiler
            analyzer = FSDPOverlapProfiler(profiler)
            metrics = analyzer.calculate_overlap()
            metrics.print_summary()
            
            # Export metrics
            metrics_path = os.path.join(cfg.profile_dir, f"{mode}_{cfg.experiment_name}", "overlap_metrics.json")
            analyzer.export_metrics(metrics_path)
        except Exception as e:
            print(f"\nNote: Could not calculate overlap metrics: {e}")
        
        print(f"\nTraces saved to: {cfg.profile_dir}/{mode}_{cfg.experiment_name}/")
        print("\nTo view traces:")
        print(f"  tensorboard --logdir={cfg.profile_dir}")
        print("  Then go to 'PyTorch Profiler' tab -> 'Trace'")
        print(f"\nTo analyze programmatically:")
        print(f"  python training/analysis/profile_trace_analyzer.py --trace-dir={cfg.profile_dir}")
        print("\nLook for:")
        print("  1. nccl:all_gather kernels (FSDP forward)")
        print("  2. nccl:reduce_scatter kernels (FSDP backward)")
        print("  3. nccl:all_reduce kernels (DDP backward)")
        print("  4. GPU idle gaps before/after communication")
    
    dist.destroy_process_group()


def print_analysis_guide():
    """Print detailed analysis guide for the blog."""
    guide = """
================================================================================
COMMUNICATION PROFILING ANALYSIS GUIDE (For Your Blog)
================================================================================

WHAT TO LOOK FOR IN TENSORBOARD:

1. TRACE VIEW (PyTorch Profiler -> Trace)
   - Each row is a CUDA stream
   - Compute kernels: gemm_*, volta_*, ampere_*, etc.
   - Communication: nccl:all_gather, nccl:reduce_scatter, nccl:all_reduce
   
2. COMPUTE vs COMMUNICATION OVERLAP
   - GOOD (NVLink): Communication kernels "hide" under compute kernels
     The GPU stays busy computing while data transfers in background.
   
   - BAD (PCIe bottleneck): Visible "gaps" where GPU is idle
     The compute kernels finish, then GPU waits for communication.
     These gaps are your "network tax".

3. KEY MEASUREMENTS FOR YOUR BLOG
   Compare between NVLink baseline and PCIe bottleneck:
   
   a) Step Duration
      - Measure wall-clock time for ## TRAIN_STEP ## region
      - Delta = (PCIe step time) - (NVLink step time)
      - This is the physical cost of your network
   
   b) Communication Time
      - Sum all nccl:* kernel durations
      - In NVLink: May be "hidden" (overlapped with compute)
      - In PCIe: Adds serially to total time
   
   c) GPU Utilization
      - Look at the trace timeline
      - Count idle (white/gray) vs active (colored) regions
      - NVLink should show ~95%+ utilization
      - PCIe might show 60-80% (rest is waiting)
   
   d) Bandwidth Saturation
      - Calculate: data_moved / communication_time
      - NVLink: Should hit ~300-600 GB/s
      - PCIe: Limited to ~32-64 GB/s (depends on gen)

4. FSDP-SPECIFIC ANALYSIS
   - all_gather happens during FORWARD (parameters gathered)
   - reduce_scatter happens during BACKWARD (gradients scattered)
   - With multiple FSDP units, you should see interleaved pattern:
     all_gather -> compute -> reduce_scatter -> all_gather -> ...

5. DDP-SPECIFIC ANALYSIS
   - all_reduce happens during BACKWARD (gradients averaged)
   - With gradient bucketing, you see multiple all_reduce calls
   - Communication can overlap with backward of earlier layers

================================================================================
RECOMMENDED BLOG STRUCTURE
================================================================================

1. INTRODUCTION
   - Why communication matters at scale
   - The "memory wall" and "network wall" concepts

2. EXPERIMENTAL SETUP
   - Hardware topology (show nvidia-smi topo -m output)
   - Model configuration (parameters, batch size)
   - Two runs: NVLink baseline vs PCIe simulation

3. PROFILER TRACES (with screenshots)
   - Side-by-side comparison of traces
   - Annotate the key regions (compute, communication, gaps)

4. QUANTITATIVE ANALYSIS
   - Table: Step time, communication time, utilization
   - The "network tax" = how much slower PCIe is

5. FSDP vs DDP COMMUNICATION PATTERNS
   - DDP: all_reduce gradients (2x model size)
   - FSDP: all_gather + reduce_scatter per unit
   - When each is more efficient

6. RECOMMENDATIONS
   - Use NVLink/NVSwitch when possible
   - Overlap strategies (communication hiding)
   - Model/batch size tuning for your network
================================================================================
"""
    print(guide)


if __name__ == "__main__":
    # Parse command line arguments
    if "--analysis-guide" in sys.argv:
        print_analysis_guide()
        sys.exit(0)
    
    # Determine mode
    mode = "fsdp"
    for arg in sys.argv:
        if arg.startswith("--mode="):
            mode = arg.split("=")[1].lower()
    
    if mode not in ("fsdp", "ddp"):
        print(f"Error: mode must be 'fsdp' or 'ddp', got '{mode}'")
        sys.exit(1)
    
    run_profiled_training(mode)
