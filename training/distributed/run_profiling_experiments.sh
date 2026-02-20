#!/bin/bash
#
# Run Communication Profiling Experiments
# 
# This script runs comprehensive profiling experiments comparing:
# - DDP vs FSDP
# - NVLink (fast) vs PCIe (slow) interconnect
# - With and without activation checkpointing
#
# Usage:
#   bash training/distributed/run_profiling_experiments.sh [suite]
#
# Suites:
#   all        - Run all experiments (default)
#                • DDP (NVLink)
#                • FSDP (NVLink)  
#                • FSDP (PCIe bottleneck)
#                • DDP + Activation Checkpointing (TODO)
#                • FSDP + Activation Checkpointing (TODO)
#
#   basic      - Run only DDP and FSDP with NVLink
#                • DDP (NVLink)
#                • FSDP (NVLink)
#
#   network    - Run FSDP NVLink vs PCIe comparison
#                • FSDP (NVLink baseline)
#                • FSDP (PCIe bottleneck)
#
#   checkpoint - Run checkpointing comparisons (TODO)
#                • DDP + Checkpointing
#                • FSDP + Checkpointing
#
# Examples:
#   bash training/distributed/run_profiling_experiments.sh            # Run all
#   bash training/distributed/run_profiling_experiments.sh basic      # Quick comparison
#   bash training/distributed/run_profiling_experiments.sh network    # Network analysis
#

set -e  # Exit on error

# Parse arguments
SUITE="${1:-all}"

if [[ "$SUITE" != "all" && "$SUITE" != "basic" && "$SUITE" != "network" && "$SUITE" != "checkpoint" ]]; then
    echo "Error: suite must be 'all', 'basic', 'network', or 'checkpoint', got '$SUITE'"
    exit 1
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Communication Profiling Experiments${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Suite: $SUITE"
echo "Output directory: ./output/profiling_traces/"
echo ""

# Determine which experiments to run
RUN_DDP=false
RUN_FSDP=false
RUN_PCIE=false
RUN_CHECKPOINT=false

case "$SUITE" in
    all)
        RUN_DDP=true
        RUN_FSDP=true
        RUN_PCIE=true
        RUN_CHECKPOINT=true
        ;;
    basic)
        RUN_DDP=true
        RUN_FSDP=true
        ;;
    network)
        RUN_FSDP=true
        RUN_PCIE=true
        ;;
    checkpoint)
        RUN_CHECKPOINT=true
        ;;
esac

echo "Experiments to run:"
[[ "$RUN_DDP" == true ]] && echo "  ✓ DDP (NVLink)"
[[ "$RUN_FSDP" == true ]] && echo "  ✓ FSDP (NVLink)"
[[ "$RUN_PCIE" == true ]] && echo "  ✓ FSDP (PCIe bottleneck)"
[[ "$RUN_CHECKPOINT" == true ]] && echo "  ✓ DDP + Activation Checkpointing"
[[ "$RUN_CHECKPOINT" == true ]] && echo "  ✓ FSDP + Activation Checkpointing"
echo ""

# Check for GPUs
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}Error: nvidia-smi not found. This script requires NVIDIA GPUs.${NC}"
    exit 1
fi

GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo -e "${GREEN}Found $GPU_COUNT GPU(s)${NC}"

if [ "$GPU_COUNT" -lt 2 ]; then
    echo -e "${YELLOW}Warning: Only $GPU_COUNT GPU detected. Results may not be meaningful.${NC}"
    echo -e "${YELLOW}This experiment compares multi-GPU communication patterns.${NC}"
fi

# Show GPU topology
echo ""
echo -e "${BLUE}GPU Topology:${NC}"
nvidia-smi topo -m
echo ""

# Confirm before running
read -p "Continue with profiling? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# Create output directory
mkdir -p output/profiling_traces

EXPERIMENT_NUM=1

# Experiment: DDP (NVLink)
if [[ "$RUN_DDP" == true ]]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Experiment $EXPERIMENT_NUM: DDP (NVLink)${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo "Data Parallel with all_reduce (full model replication)."
    echo "Using high-bandwidth NVLink interconnect."
    echo ""
    
    torchrun --nproc_per_node=2 training/distributed/profile_distributed.py --mode=ddp
    
    echo ""
    echo -e "${GREEN}✓ DDP (NVLink) complete${NC}"
    echo ""
    sleep 2
    EXPERIMENT_NUM=$((EXPERIMENT_NUM + 1))
fi

# Experiment: FSDP (NVLink)
if [[ "$RUN_FSDP" == true ]]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Experiment $EXPERIMENT_NUM: FSDP (NVLink)${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo "Fully Sharded Data Parallel (parameter sharding)."
    echo "Using high-bandwidth NVLink interconnect."
    echo ""
    
    torchrun --nproc_per_node=2 training/distributed/profile_distributed.py --mode=fsdp
    
    echo ""
    echo -e "${GREEN}✓ FSDP (NVLink) complete${NC}"
    echo ""
    sleep 2
    EXPERIMENT_NUM=$((EXPERIMENT_NUM + 1))
fi

# Experiment: FSDP (PCIe bottleneck)
if [[ "$RUN_PCIE" == true ]]; then
    echo ""
    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}Experiment $EXPERIMENT_NUM: FSDP (PCIe)${NC}"
    echo -e "${YELLOW}========================================${NC}"
    echo "FSDP with forced PCIe/CPU communication path."
    echo "Environment: NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1"
    echo ""
    
    NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 \
        torchrun --nproc_per_node=2 training/distributed/profile_distributed.py --mode=fsdp
    
    echo ""
    echo -e "${GREEN}✓ FSDP (PCIe) complete${NC}"
    echo ""
    sleep 2
    EXPERIMENT_NUM=$((EXPERIMENT_NUM + 1))
fi

# Experiment: DDP + Activation Checkpointing
if [[ "$RUN_CHECKPOINT" == true ]]; then
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Experiment $EXPERIMENT_NUM: DDP + Checkpointing${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo "DDP with activation checkpointing (trades compute for memory)."
    echo "Using ddp_fsdp_oom_demo.py with profiling enabled."
    echo ""
    
    torchrun --nproc_per_node=2 training/distributed/ddp_fsdp_oom_demo.py \
        --mode=ddp \
        --use_activation_checkpointing \
        --profile \
        --model_size=small \
        --max_steps=10
    
    echo ""
    echo -e "${GREEN}✓ DDP + Checkpointing complete${NC}"
    echo ""
    sleep 2
    EXPERIMENT_NUM=$((EXPERIMENT_NUM + 1))
fi

# Experiment: FSDP + Activation Checkpointing
if [[ "$RUN_CHECKPOINT" == true ]]; then
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Experiment $EXPERIMENT_NUM: FSDP + Checkpointing${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo "FSDP with activation checkpointing."
    echo "Using ddp_fsdp_oom_demo.py with profiling enabled."
    echo ""
    
    torchrun --nproc_per_node=2 training/distributed/ddp_fsdp_oom_demo.py \
        --mode=fsdp \
        --use_activation_checkpointing \
        --profile \
        --model_size=small \
        --max_steps=10
    
    echo ""
    echo -e "${GREEN}✓ FSDP + Checkpointing complete${NC}"
    echo ""
    sleep 2
    EXPERIMENT_NUM=$((EXPERIMENT_NUM + 1))
fi

# Analysis
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Analyzing Results${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

python training/analysis/profile_trace_analyzer.py

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}All Experiments Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Experiments run:"
ls -1 output/profiling_traces/ | sed 's/^/  - /'
echo ""
echo "Results:"
echo "  - Traces: ./output/profiling_traces/"
echo "  - Analysis: ./output/communication_analysis.md"
echo "  - Overlap metrics: ./output/profiling_traces/*/overlap_metrics.json"
echo ""
echo "Next steps:"
echo ""
echo "  1. View traces in TensorBoard:"
echo "     tensorboard --logdir=./output/profiling_traces"
echo "     Then click: PYTORCH_PROFILER tab -> Select run -> Trace view"
echo ""
echo "  2. Review the comparison table:"
echo "     cat output/communication_analysis.md"
echo ""
echo "  3. View overlap metrics:"
echo "     cat output/profiling_traces/fsdp_nvlink_baseline/overlap_metrics.json"
echo ""
echo "  4. Open Chrome traces (chrome://tracing):"
echo "     Load any: output/profiling_traces/*/*.pt.trace.json"
echo ""
echo "Comparison highlights:"
if [[ -f output/communication_analysis.md ]]; then
    echo ""
    cat output/communication_analysis.md | head -20
fi
echo ""
echo -e "${BLUE}Happy blogging!${NC}"
