"""
Profile Trace Analyzer

Parses torch.profiler traces and extracts key metrics for comparing
communication performance across different network configurations.

This script helps you generate quantitative data for your blog:
- Step timing breakdown (compute vs communication)
- NCCL operation statistics
- GPU utilization estimates
- Network bottleneck quantification

Usage:
    python training/analysis/profile_trace_analyzer.py --trace-dir ./output/profiling_traces/

Author: Systems Blog Series
"""

import os
import sys
import json
import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


@dataclass
class KernelEvent:
    """Represents a single kernel execution event from the trace."""
    name: str
    category: str
    start_us: float
    duration_us: float
    stream: Optional[int] = None
    
    @property
    def end_us(self) -> float:
        return self.start_us + self.duration_us
    
    @property
    def is_nccl(self) -> bool:
        return "nccl" in self.name.lower()
    
    @property
    def is_compute(self) -> bool:
        compute_keywords = ["gemm", "volta", "ampere", "sm80", "sm90", "cutlass", "attention"]
        return any(kw in self.name.lower() for kw in compute_keywords)


@dataclass
class StepMetrics:
    """Metrics for a single training step."""
    step_id: int
    total_duration_ms: float
    compute_time_ms: float
    nccl_time_ms: float
    memory_time_ms: float
    other_time_ms: float
    
    # NCCL breakdown
    all_gather_time_ms: float = 0.0
    all_gather_count: int = 0
    reduce_scatter_time_ms: float = 0.0
    reduce_scatter_count: int = 0
    all_reduce_time_ms: float = 0.0
    all_reduce_count: int = 0
    
    @property
    def communication_fraction(self) -> float:
        """Fraction of step spent in communication."""
        if self.total_duration_ms > 0:
            return self.nccl_time_ms / self.total_duration_ms
        return 0.0
    
    @property
    def compute_fraction(self) -> float:
        """Fraction of step spent in compute."""
        if self.total_duration_ms > 0:
            return self.compute_time_ms / self.total_duration_ms
        return 0.0


@dataclass 
class ExperimentSummary:
    """Summary statistics for an entire experiment."""
    experiment_name: str
    mode: str  # fsdp or ddp
    num_steps: int
    
    avg_step_time_ms: float
    avg_compute_time_ms: float
    avg_nccl_time_ms: float
    
    total_all_gather_ms: float = 0.0
    total_reduce_scatter_ms: float = 0.0
    total_all_reduce_ms: float = 0.0
    
    nccl_op_counts: Dict[str, int] = field(default_factory=dict)
    
    @property
    def gpu_utilization(self) -> float:
        """Estimated GPU utilization (compute / total)."""
        if self.avg_step_time_ms > 0:
            return self.avg_compute_time_ms / self.avg_step_time_ms
        return 0.0
    
    @property
    def communication_overhead(self) -> float:
        """Communication as fraction of step time."""
        if self.avg_step_time_ms > 0:
            return self.avg_nccl_time_ms / self.avg_step_time_ms
        return 0.0


def parse_chrome_trace(trace_path: str) -> List[KernelEvent]:
    """
    Parse a Chrome trace JSON file and extract kernel events.
    
    Chrome trace format: https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU
    """
    events = []
    
    with open(trace_path, 'r') as f:
        try:
            trace_data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error parsing {trace_path}: {e}")
            return events
    
    # Handle both array format and object format
    if isinstance(trace_data, dict):
        trace_events = trace_data.get("traceEvents", [])
    else:
        trace_events = trace_data
    
    for event in trace_events:
        # We only care about duration events ('X') or begin/end events ('B'/'E')
        if event.get("ph") not in ("X", "B", "E"):
            continue
        
        name = event.get("name", "")
        cat = event.get("cat", "")
        ts = event.get("ts", 0)  # timestamp in microseconds
        dur = event.get("dur", 0)  # duration in microseconds
        
        # Skip empty or metadata events
        if not name or name.startswith("__"):
            continue
        
        events.append(KernelEvent(
            name=name,
            category=cat,
            start_us=ts,
            duration_us=dur,
            stream=event.get("args", {}).get("stream", None)
        ))
    
    return events


def find_step_boundaries(events: List[KernelEvent]) -> List[Tuple[float, float]]:
    """
    Find the start/end times of each training step using markers.
    
    Looks for "## TRAIN_STEP ##" record_function markers.
    """
    step_ranges = []
    
    for event in events:
        if "TRAIN_STEP" in event.name:
            step_ranges.append((event.start_us, event.end_us))
    
    return step_ranges


def analyze_step(
    events: List[KernelEvent],
    step_start_us: float,
    step_end_us: float,
    step_id: int
) -> StepMetrics:
    """Analyze a single training step and extract metrics."""
    
    compute_time = 0.0
    nccl_time = 0.0
    memory_time = 0.0
    
    all_gather_time = 0.0
    all_gather_count = 0
    reduce_scatter_time = 0.0
    reduce_scatter_count = 0
    all_reduce_time = 0.0
    all_reduce_count = 0
    
    for event in events:
        # Only consider events within this step
        if event.end_us < step_start_us or event.start_us > step_end_us:
            continue
        
        # Compute time
        if event.is_compute:
            compute_time += event.duration_us
        
        # NCCL time (with breakdown)
        if event.is_nccl:
            nccl_time += event.duration_us
            
            name_lower = event.name.lower()
            if "all_gather" in name_lower or "allgather" in name_lower:
                all_gather_time += event.duration_us
                all_gather_count += 1
            elif "reduce_scatter" in name_lower or "reducescatter" in name_lower:
                reduce_scatter_time += event.duration_us
                reduce_scatter_count += 1
            elif "all_reduce" in name_lower or "allreduce" in name_lower:
                all_reduce_time += event.duration_us
                all_reduce_count += 1
        
        # Memory operations
        if "memcpy" in event.name.lower() or "memset" in event.name.lower():
            memory_time += event.duration_us
    
    total_duration = step_end_us - step_start_us
    other_time = total_duration - compute_time - nccl_time - memory_time
    
    return StepMetrics(
        step_id=step_id,
        total_duration_ms=total_duration / 1000,
        compute_time_ms=compute_time / 1000,
        nccl_time_ms=nccl_time / 1000,
        memory_time_ms=memory_time / 1000,
        other_time_ms=max(0, other_time / 1000),  # Can be negative due to parallelism
        all_gather_time_ms=all_gather_time / 1000,
        all_gather_count=all_gather_count,
        reduce_scatter_time_ms=reduce_scatter_time / 1000,
        reduce_scatter_count=reduce_scatter_count,
        all_reduce_time_ms=all_reduce_time / 1000,
        all_reduce_count=all_reduce_count,
    )


def analyze_experiment(trace_dir: str) -> Optional[ExperimentSummary]:
    """Analyze all traces in an experiment directory."""
    
    # Find trace files (chrome_trace.json or .pt.trace.json)
    trace_files = []
    for root, dirs, files in os.walk(trace_dir):
        for f in files:
            # Look for chrome_trace.json first, then fall back to .pt.trace.json
            if f == "chrome_trace.json" or f.endswith(".pt.trace.json"):
                trace_files.append(os.path.join(root, f))
    
    if not trace_files:
        print(f"No trace files found in {trace_dir}")
        print(f"Expected: chrome_trace.json or *.pt.trace.json")
        return None
    
    # Prefer chrome_trace.json if available
    chrome_traces = [f for f in trace_files if f.endswith("chrome_trace.json")]
    if chrome_traces:
        trace_files = chrome_traces
    
    # Parse experiment name from directory structure
    # Expected: ./output/profiling_traces/fsdp_nvlink_baseline/rank_0/chrome_trace.json
    parts = trace_dir.rstrip("/").split("/")
    experiment_name = parts[-1] if parts else "unknown"
    
    # Infer mode from name
    mode = "fsdp" if "fsdp" in experiment_name.lower() else "ddp"
    
    all_step_metrics: List[StepMetrics] = []
    
    # Analyze first trace file (rank 0 is usually most representative)
    trace_path = sorted(trace_files)[0]
    print(f"Analyzing: {trace_path}")
    
    events = parse_chrome_trace(trace_path)
    if not events:
        print(f"No events found in trace")
        return None
    
    step_boundaries = find_step_boundaries(events)
    if not step_boundaries:
        print(f"No TRAIN_STEP markers found. Using heuristics...")
        # Fallback: analyze all events as one step
        min_ts = min(e.start_us for e in events)
        max_ts = max(e.end_us for e in events)
        step_boundaries = [(min_ts, max_ts)]
    
    print(f"Found {len(step_boundaries)} training steps")
    
    for i, (start, end) in enumerate(step_boundaries):
        metrics = analyze_step(events, start, end, i)
        all_step_metrics.append(metrics)
    
    # Compute summary statistics
    if not all_step_metrics:
        return None
    
    avg_step = sum(m.total_duration_ms for m in all_step_metrics) / len(all_step_metrics)
    avg_compute = sum(m.compute_time_ms for m in all_step_metrics) / len(all_step_metrics)
    avg_nccl = sum(m.nccl_time_ms for m in all_step_metrics) / len(all_step_metrics)
    
    total_all_gather = sum(m.all_gather_time_ms for m in all_step_metrics)
    total_reduce_scatter = sum(m.reduce_scatter_time_ms for m in all_step_metrics)
    total_all_reduce = sum(m.all_reduce_time_ms for m in all_step_metrics)
    
    op_counts = {
        "all_gather": sum(m.all_gather_count for m in all_step_metrics),
        "reduce_scatter": sum(m.reduce_scatter_count for m in all_step_metrics),
        "all_reduce": sum(m.all_reduce_count for m in all_step_metrics),
    }
    
    return ExperimentSummary(
        experiment_name=experiment_name,
        mode=mode,
        num_steps=len(all_step_metrics),
        avg_step_time_ms=avg_step,
        avg_compute_time_ms=avg_compute,
        avg_nccl_time_ms=avg_nccl,
        total_all_gather_ms=total_all_gather,
        total_reduce_scatter_ms=total_reduce_scatter,
        total_all_reduce_ms=total_all_reduce,
        nccl_op_counts=op_counts,
    )


def compare_experiments(summaries: List[ExperimentSummary]) -> None:
    """Generate comparison table for blog."""
    
    if len(summaries) < 2:
        print("Need at least 2 experiments to compare")
        return
    
    print("\n" + "=" * 90)
    print("EXPERIMENT COMPARISON (For Your Blog)")
    print("=" * 90)
    
    # Header
    print(f"\n{'Metric':<35}", end="")
    for s in summaries:
        print(f"{s.experiment_name:>25}", end="")
    print()
    print("-" * 90)
    
    # Step time
    print(f"{'Avg Step Time (ms)':<35}", end="")
    for s in summaries:
        print(f"{s.avg_step_time_ms:>25.2f}", end="")
    print()
    
    # Compute time
    print(f"{'Avg Compute Time (ms)':<35}", end="")
    for s in summaries:
        print(f"{s.avg_compute_time_ms:>25.2f}", end="")
    print()
    
    # NCCL time
    print(f"{'Avg NCCL Time (ms)':<35}", end="")
    for s in summaries:
        print(f"{s.avg_nccl_time_ms:>25.2f}", end="")
    print()
    
    # GPU utilization
    print(f"{'Est. GPU Utilization (%)':<35}", end="")
    for s in summaries:
        print(f"{s.gpu_utilization * 100:>24.1f}%", end="")
    print()
    
    # Communication overhead
    print(f"{'Communication Overhead (%)':<35}", end="")
    for s in summaries:
        print(f"{s.communication_overhead * 100:>24.1f}%", end="")
    print()
    
    print("-" * 90)
    
    # NCCL breakdown
    print(f"\n{'NCCL Operation Breakdown':^90}")
    print("-" * 90)
    
    print(f"{'all_gather Time (ms)':<35}", end="")
    for s in summaries:
        print(f"{s.total_all_gather_ms:>25.2f}", end="")
    print()
    
    print(f"{'reduce_scatter Time (ms)':<35}", end="")
    for s in summaries:
        print(f"{s.total_reduce_scatter_ms:>25.2f}", end="")
    print()
    
    print(f"{'all_reduce Time (ms)':<35}", end="")
    for s in summaries:
        print(f"{s.total_all_reduce_ms:>25.2f}", end="")
    print()
    
    print("-" * 90)
    
    # Calculate network tax
    if len(summaries) >= 2:
        baseline = summaries[0]
        bottleneck = summaries[1]
        
        step_delta = bottleneck.avg_step_time_ms - baseline.avg_step_time_ms
        slowdown = bottleneck.avg_step_time_ms / baseline.avg_step_time_ms
        
        print(f"\n{'NETWORK TAX ANALYSIS':^90}")
        print("-" * 90)
        print(f"Step time increase: {step_delta:.2f} ms ({(slowdown - 1) * 100:.1f}% slower)")
        print(f"Baseline step: {baseline.avg_step_time_ms:.2f} ms")
        print(f"Bottleneck step: {bottleneck.avg_step_time_ms:.2f} ms")
        print(f"\nThis {step_delta:.2f} ms is the 'network tax' - the cost of slower interconnect.")
        print("-" * 90)


def generate_blog_markdown(summaries: List[ExperimentSummary], output_path: str) -> None:
    """Generate markdown table for blog post."""
    
    md = ["## Communication Profiling Results\n"]
    
    # Create table header
    md.append("| Metric | " + " | ".join(s.experiment_name for s in summaries) + " |")
    md.append("|" + "|".join(["---"] * (len(summaries) + 1)) + "|")
    
    # Add rows
    md.append(f"| Avg Step Time (ms) | " + " | ".join(f"{s.avg_step_time_ms:.2f}" for s in summaries) + " |")
    md.append(f"| Avg Compute Time (ms) | " + " | ".join(f"{s.avg_compute_time_ms:.2f}" for s in summaries) + " |")
    md.append(f"| Avg NCCL Time (ms) | " + " | ".join(f"{s.avg_nccl_time_ms:.2f}" for s in summaries) + " |")
    md.append(f"| GPU Utilization | " + " | ".join(f"{s.gpu_utilization * 100:.1f}%" for s in summaries) + " |")
    md.append(f"| Comm Overhead | " + " | ".join(f"{s.communication_overhead * 100:.1f}%" for s in summaries) + " |")
    
    md.append("\n### NCCL Operation Breakdown\n")
    md.append("| Operation | " + " | ".join(s.experiment_name for s in summaries) + " |")
    md.append("|" + "|".join(["---"] * (len(summaries) + 1)) + "|")
    md.append(f"| all_gather (ms) | " + " | ".join(f"{s.total_all_gather_ms:.2f}" for s in summaries) + " |")
    md.append(f"| reduce_scatter (ms) | " + " | ".join(f"{s.total_reduce_scatter_ms:.2f}" for s in summaries) + " |")
    md.append(f"| all_reduce (ms) | " + " | ".join(f"{s.total_all_reduce_ms:.2f}" for s in summaries) + " |")
    
    with open(output_path, 'w') as f:
        f.write("\n".join(md))
    
    print(f"\nMarkdown table written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze torch.profiler traces for communication benchmarking"
    )
    parser.add_argument(
        "--trace-dir", 
        type=str, 
        default="./output/profiling_traces",
        help="Directory containing profiler traces"
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default="./output/communication_analysis.md",
        help="Output path for markdown table"
    )
    
    args = parser.parse_args()
    
    # Find all experiment directories
    experiment_dirs = []
    if os.path.exists(args.trace_dir):
        for item in os.listdir(args.trace_dir):
            full_path = os.path.join(args.trace_dir, item)
            if os.path.isdir(full_path):
                experiment_dirs.append(full_path)
    
    if not experiment_dirs:
        print(f"No experiment directories found in {args.trace_dir}")
        print("\nRun the profiling script first:")
        print("  torchrun --nproc_per_node=2 training/distributed/profile_distributed.py --mode=fsdp")
        sys.exit(1)
    
    print(f"Found {len(experiment_dirs)} experiments:")
    for d in experiment_dirs:
        print(f"  - {d}")
    
    # Analyze each experiment
    summaries = []
    for exp_dir in sorted(experiment_dirs):
        summary = analyze_experiment(exp_dir)
        if summary:
            summaries.append(summary)
            print(f"\nSummary for {summary.experiment_name}:")
            print(f"  Steps: {summary.num_steps}")
            print(f"  Avg step time: {summary.avg_step_time_ms:.2f} ms")
            print(f"  GPU utilization: {summary.gpu_utilization * 100:.1f}%")
    
    # Compare if we have multiple experiments
    if len(summaries) >= 2:
        compare_experiments(summaries)
    
    # Generate markdown if requested
    if args.output_md and summaries:
        generate_blog_markdown(summaries, args.output_md)


if __name__ == "__main__":
    main()
