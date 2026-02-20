# Distributed Training Profiling Suite

A comprehensive suite for profiling and analyzing distributed training performance, focusing on communication patterns, memory management, and bottleneck identification in multi-GPU training scenarios.

## Overview

This suite provides tools to:

- **Profile communication patterns** in DDP vs FSDP training
- **Compare network topologies** (NVLink vs PCIe bottlenecks)
- **Analyze overlap efficiency** between computation and communication
- **Simulate data pipeline bottlenecks** and their impact on training
- **Demonstrate memory optimization** techniques (activation checkpointing, FSDP sharding)

## Directory Structure

```
training/
├── README.md                    # This file
│
├── distributed/                 # Main distributed training scripts
│   ├── profile_distributed.py       # FSDP/DDP profiling with PyTorch Profiler
│   ├── train_distributed_node.py    # Multi-node training runner
│   ├── train_single_node_baseline.py # Single-node baseline for comparison
│   ├── ddp_fsdp_oom_demo.py         # Memory optimization demo (checkpointing)
│   └── run_profiling_experiments.sh # Automated experiment suite runner
│
├── analysis/                    # Analysis and simulation tools
│   ├── profile_trace_analyzer.py    # Parse and analyze PyTorch Profiler traces
│   ├── dataloader_bottleneck_simulation.py  # Simulate dataloader impact
│   ├── padding_tax_simulation.py            # Analyze padding overhead
│   ├── pcie_transfer_bottleneck.py          # Model PCIe bandwidth impact
│   ├── random_vs_seq_dataloader_simulation.py
│   ├── random_vs_seq_python_simulation.py
│   ├── streaming_loader_lib_profiling.py
│   ├── io_benchmark_utils.py
│   └── DATA_PIPELINE_REPORT.md      # Data pipeline analysis report
│
└── helper/                      # Utility modules
    ├── profiler_analyzer.py     # FSDP overlap metrics calculator
    ├── check_gpu_info.py        # GPU topology and capabilities checker
    ├── fsdp_ddp_comparison_demo.py  # Side-by-side DDP/FSDP comparison
    ├── LLM_TRAINING_INTERNALS.md    # Documentation on training internals
    └── __init__.py
```

## Quick Start

### Prerequisites

```bash
# Multi-GPU system (2+ GPUs recommended)
# CUDA toolkit installed
# PyTorch with distributed support

# Check your GPU setup
python training/helper/check_gpu_info.py
```

### Running Profiling Experiments

The easiest way to get started is with the automated experiment runner:

```bash
# Run all experiments (DDP, FSDP, NVLink, PCIe)
bash training/distributed/run_profiling_experiments.sh all

# Run quick comparison (DDP vs FSDP with NVLink only)
bash training/distributed/run_profiling_experiments.sh basic

# Run network comparison (FSDP NVLink vs PCIe)
bash training/distributed/run_profiling_experiments.sh network
```

### Manual Profiling

You can also run individual profiling experiments:

```bash
# Profile FSDP with NVLink
torchrun --nproc_per_node=2 training/distributed/profile_distributed.py --mode=fsdp

# Profile DDP with NVLink
torchrun --nproc_per_node=2 training/distributed/profile_distributed.py --mode=ddp

# Profile FSDP with forced PCIe path (to simulate slow interconnect)
NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 \
    torchrun --nproc_per_node=2 training/distributed/profile_distributed.py --mode=fsdp
```

### Analyzing Results

```bash
# Analyze profiler traces and generate comparison report
python training/analysis/profile_trace_analyzer.py

# View traces in TensorBoard
tensorboard --logdir=./output/profiling_traces

# View Chrome traces (chrome://tracing)
# Load: output/profiling_traces/<experiment>/*.pt.trace.json
```

## Key Scripts

### Distributed Training Scripts

#### `distributed/profile_distributed.py`
Main profiling script for comparing DDP and FSDP communication patterns.

**Usage:**
```bash
torchrun --nproc_per_node=<NUM_GPUS> training/distributed/profile_distributed.py \
    --mode={ddp|fsdp} \
    [--model_size={small|medium|large}] \
    [--batch_size=<size>] \
    [--max_steps=<steps>]
```

**Features:**
- PyTorch Profiler integration with Chrome trace export
- NCCL operation tracking
- Overlap efficiency metrics (FSDP)
- Memory usage profiling

#### `distributed/ddp_fsdp_oom_demo.py`
Demonstrates memory optimization techniques for large models.

**Usage:**
```bash
torchrun --nproc_per_node=2 training/distributed/ddp_fsdp_oom_demo.py \
    --mode={ddp|fsdp} \
    [--use_activation_checkpointing] \
    [--profile] \
    [--model_size={small|medium|large}]
```

**Key comparisons:**
- DDP vs FSDP memory usage
- Impact of activation checkpointing
- Memory-compute tradeoffs

#### `distributed/train_distributed_node.py`
Production-ready multi-node training runner with comprehensive monitoring.

**Features:**
- Multi-node, multi-GPU support
- Automatic checkpointing
- Learning rate scheduling
- W&B integration support
- FSDP with CPU offloading options

#### `distributed/run_profiling_experiments.sh`
Automated experiment suite for systematic comparison.

**Experiment suites:**
- `all`: Complete suite (DDP, FSDP, NVLink, PCIe, checkpointing)
- `basic`: Quick DDP vs FSDP comparison
- `network`: Network topology impact (NVLink vs PCIe)
- `checkpoint`: Activation checkpointing experiments

### Analysis Tools

#### `analysis/profile_trace_analyzer.py`
Parse PyTorch Profiler traces and extract key metrics.

**Metrics extracted:**
- Step timing breakdown (compute vs communication)
- NCCL operation statistics
- GPU utilization estimates
- Communication overhead quantification

**Usage:**
```bash
python training/analysis/profile_trace_analyzer.py \
    [--trace-dir ./output/profiling_traces]
```

#### `analysis/dataloader_bottleneck_simulation.py`
Simulate impact of dataloader bottlenecks on training throughput.

**Scenarios:**
- CPU-bound preprocessing
- I/O bound data loading
- Inefficient prefetching
- Impact of `num_workers` configuration

#### `analysis/padding_tax_simulation.py`
Quantify compute waste from sequence padding in batch training.

**Analysis:**
- FLOPS waste calculation
- Impact of batch composition
- Dynamic batching strategies

### Helper Utilities

#### `helper/profiler_analyzer.py`
FSDP overlap metrics calculator.

**Key metrics:**
- Communication-computation overlap percentage
- All-gather efficiency
- Reduce-scatter efficiency
- Forward/backward pass breakdown

#### `helper/check_gpu_info.py`
GPU topology and capabilities checker.

**Information displayed:**
- GPU models and memory
- NVLink topology
- PCIe configuration
- CUDA/NCCL versions

## Output Directory

All experiments generate outputs in `./output/`:

```
output/
├── profiling_traces/
│   ├── ddp_nvlink_baseline/
│   │   ├── *.pt.trace.json      # Chrome trace
│   │   ├── *.json               # TensorBoard trace
│   │   └── overlap_metrics.json # FSDP metrics (if applicable)
│   ├── fsdp_nvlink_baseline/
│   └── fsdp_pcie_bottleneck/
└── communication_analysis.md     # Generated comparison report
```

## Understanding the Results

### Key Metrics

1. **Step Time**: Total time per training iteration
2. **Compute Time**: Time spent in forward/backward computation
3. **Communication Time**: Time spent in all-reduce/all-gather/reduce-scatter
4. **Overlap Efficiency**: % of communication hidden by computation (FSDP only)
5. **GPU Utilization**: Estimated % of time GPU is actively computing

### Comparing DDP vs FSDP

**DDP (Data Parallel):**
- ✅ Simple, well-understood
- ✅ No memory overhead from sharding
- ❌ All-reduce at end of backward (no overlap)
- ❌ Full model replication (high memory for large models)

**FSDP (Fully Sharded Data Parallel):**
- ✅ Shard parameters across GPUs (lower memory)
- ✅ Overlap all-gather with forward, reduce-scatter with backward
- ✅ Scales to larger models
- ❌ More complex
- ❌ Sensitive to communication bandwidth

### Network Topology Impact

**NVLink:**
- High bandwidth (~300-600 GB/s)
- Low latency
- Best for FSDP communication patterns

**PCIe:**
- Lower bandwidth (~16-32 GB/s)
- Higher latency
- Can bottleneck FSDP severely
- DDP less sensitive (bulk all-reduce at end)

## Common Use Cases

### Use Case 1: Profile Your Training Job

```bash
# Replace with your training script
torchrun --nproc_per_node=8 training/distributed/profile_distributed.py \
    --mode=fsdp \
    --model_size=large \
    --batch_size=4 \
    --max_steps=20
```

### Use Case 2: Debug Communication Bottlenecks

```bash
# Run both network configs
bash training/distributed/run_profiling_experiments.sh network

# Analyze traces
python training/analysis/profile_trace_analyzer.py

# Look for high communication times or low overlap
```

### Use Case 3: Optimize Memory Usage

```bash
# Compare memory footprints
torchrun --nproc_per_node=2 training/distributed/ddp_fsdp_oom_demo.py \
    --mode=ddp --model_size=large

torchrun --nproc_per_node=2 training/distributed/ddp_fsdp_oom_demo.py \
    --mode=fsdp --model_size=large

torchrun --nproc_per_node=2 training/distributed/ddp_fsdp_oom_demo.py \
    --mode=fsdp --use_activation_checkpointing --model_size=large
```

## Troubleshooting

### No GPUs Detected
```bash
# Check GPU visibility
nvidia-smi
python training/helper/check_gpu_info.py
```

### NCCL Errors
```bash
# Enable NCCL debug output
export NCCL_DEBUG=INFO

# Check network connectivity
nvidia-smi topo -m
```

### Out of Memory
```bash
# Try smaller batch size or model
--batch_size=1 --model_size=small

# Enable activation checkpointing
--use_activation_checkpointing

# Use FSDP with CPU offload
# (see train_distributed_node.py)
```

## Advanced Topics

### Multi-Node Training

```bash
# On node 0 (master)
torchrun \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr=<node0_ip> \
    --master_port=29500 \
    training/distributed/train_distributed_node.py

# On node 1
torchrun \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr=<node0_ip> \
    --master_port=29500 \
    training/distributed/train_distributed_node.py
```

### Custom Profiling

See `helper/profiler_analyzer.py` for examples of:
- Custom metric extraction
- Overlap calculation
- Memory timeline analysis

## Related Documentation

- [LLM Training Internals](helper/LLM_TRAINING_INTERNALS.md) - Deep dive into training mechanics
- [Data Pipeline Report](analysis/DATA_PIPELINE_REPORT.md) - Data loading bottlenecks
- [Main README](../README.md) - Full project documentation

## Contributing

When adding new experiments or analysis tools:

1. Follow the existing code structure
2. Add comprehensive docstrings
3. Include usage examples in this README
4. Test with both DDP and FSDP modes

## References

- [PyTorch FSDP Tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- [PyTorch Profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)
