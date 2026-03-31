"""
Profiler Analysis Utilities

Classes for extracting communication vs compute metrics from PyTorch profiler results.
Used for quantifying network bottlenecks and GPU utilization.
"""

import torch
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json


@dataclass
class OverlapMetrics:
    """Metrics for compute-communication overlap analysis."""
    total_step_time_ms: float
    total_compute_time_ms: float
    total_nccl_time_ms: float
    total_memory_time_ms: float
    
    # NCCL breakdown
    all_gather_time_ms: float
    all_gather_count: int
    reduce_scatter_time_ms: float
    reduce_scatter_count: int
    all_reduce_time_ms: float
    all_reduce_count: int
    
    # Derived metrics
    compute_fraction: float
    nccl_fraction: float
    gpu_utilization: float  # Estimate: compute / total
    
    # Overlap assessment
    overlap_health: str  # "GOOD", "MODERATE", "POOR"
    bottleneck: str  # "COMPUTE", "COMMUNICATION", "BALANCED"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "step_time_ms": self.total_step_time_ms,
            "compute_time_ms": self.total_compute_time_ms,
            "nccl_time_ms": self.total_nccl_time_ms,
            "memory_time_ms": self.total_memory_time_ms,
            "all_gather_time_ms": self.all_gather_time_ms,
            "all_gather_count": self.all_gather_count,
            "reduce_scatter_time_ms": self.reduce_scatter_time_ms,
            "reduce_scatter_count": self.reduce_scatter_count,
            "all_reduce_time_ms": self.all_reduce_time_ms,
            "all_reduce_count": self.all_reduce_count,
            "compute_fraction": self.compute_fraction,
            "nccl_fraction": self.nccl_fraction,
            "gpu_utilization": self.gpu_utilization,
            "overlap_health": self.overlap_health,
            "bottleneck": self.bottleneck,
        }
    
    def print_summary(self):
        """Print human-readable summary."""
        print("\n" + "="*70)
        print("OVERLAP ANALYSIS SUMMARY")
        print("="*70)
        print(f"Step Time:         {self.total_step_time_ms:>10.2f} ms")
        print(f"Compute Time:      {self.total_compute_time_ms:>10.2f} ms ({self.compute_fraction*100:>5.1f}%)")
        print(f"NCCL Time:         {self.total_nccl_time_ms:>10.2f} ms ({self.nccl_fraction*100:>5.1f}%)")
        print(f"Memory Ops:        {self.total_memory_time_ms:>10.2f} ms")
        print("-"*70)
        print(f"GPU Utilization:   {self.gpu_utilization*100:>10.1f}%")
        print(f"Overlap Health:    {self.overlap_health:>10}")
        print(f"Bottleneck:        {self.bottleneck:>10}")
        print("-"*70)
        print("NCCL Breakdown:")
        print(f"  all_gather:      {self.all_gather_time_ms:>10.2f} ms (count={self.all_gather_count})")
        print(f"  reduce_scatter:  {self.reduce_scatter_time_ms:>10.2f} ms (count={self.reduce_scatter_count})")
        print(f"  all_reduce:      {self.all_reduce_time_ms:>10.2f} ms (count={self.all_reduce_count})")
        print("="*70)


class FSDPOverlapProfiler:
    """
    Analyzes compute-communication overlap in FSDP training.
    
    The 'overlap health' is assessed by comparing NCCL time to compute time:
    - GOOD: NCCL time << Compute time (communication hidden by compute)
    - MODERATE: NCCL time ~ Compute time (some overlap)
    - POOR: NCCL time >> Compute time (network bottleneck)
    
    Usage:
        # During profiling:
        with profile(...) as prof:
            # training loop
            prof.step()
        
        analyzer = FSDPOverlapProfiler(prof)
        metrics = analyzer.calculate_overlap()
        metrics.print_summary()
        
        # Or from saved trace:
        analyzer = FSDPOverlapProfiler.from_trace_file("trace.json")
        metrics = analyzer.calculate_overlap()
    """
    
    def __init__(self, profiler_result):
        """
        Initialize from a PyTorch profiler result.
        
        Args:
            profiler_result: torch.profiler.profile result (after profiling)
        """
        self.prof = profiler_result
        self.events = profiler_result.key_averages()
    
    @classmethod
    def from_trace_file(cls, trace_path: str):
        """
        Create analyzer from saved trace file.
        
        Args:
            trace_path: Path to .pt.trace.json file
            
        Returns:
            FSDPOverlapProfiler instance
            
        Note: This is a simplified version. For full analysis of saved traces,
              use the profile_trace_analyzer.py script.
        """
        # This would require parsing the JSON trace file
        # For now, recommend using the existing analyzer
        raise NotImplementedError(
            "Loading from trace file requires using profile_trace_analyzer.py. "
            "Use FSDPOverlapProfiler(prof) during live profiling instead."
        )
    
    def _get_device_time(self, event) -> float:
        """Get device time from event, handling different PyTorch versions."""
        # Try different attribute names for compatibility
        if hasattr(event, 'device_time_total'):
            return event.device_time_total
        elif hasattr(event, 'cuda_time_total'):
            return event.cuda_time_total
        return 0.0
    
    def calculate_overlap(self) -> OverlapMetrics:
        """
        Calculate overlap metrics from profiler events.
        
        Returns:
            OverlapMetrics with detailed analysis
        """
        # Categorize events
        nccl_events = []
        compute_events = []
        memory_events = []
        
        # NCCL breakdown
        all_gather_events = []
        reduce_scatter_events = []
        all_reduce_events = []
        
        for e in self.events:
            key_lower = e.key.lower()
            
            # NCCL operations
            if "nccl" in key_lower:
                nccl_events.append(e)
                
                if "all_gather" in key_lower or "allgather" in key_lower:
                    all_gather_events.append(e)
                elif "reduce_scatter" in key_lower or "reducescatter" in key_lower:
                    reduce_scatter_events.append(e)
                elif "all_reduce" in key_lower or "allreduce" in key_lower:
                    all_reduce_events.append(e)
            
            # Compute operations (matrix multiplications, convolutions, etc.)
            elif any(kw in key_lower for kw in [
                "gemm", "sgemm", "hgemm", "bmm",  # Matrix multiplications
                "conv", "convolution",  # Convolutions
                "fmha", "attention",  # Attention
                "aten::linear", "aten::matmul", "aten::mm",  # Linear layers
            ]):
                compute_events.append(e)
            
            # Memory operations
            elif any(kw in key_lower for kw in ["memcpy", "memset"]):
                memory_events.append(e)
        
        # Calculate times (convert microseconds to milliseconds)
        total_compute = sum(self._get_device_time(e) for e in compute_events) / 1000
        total_nccl = sum(self._get_device_time(e) for e in nccl_events) / 1000
        total_memory = sum(self._get_device_time(e) for e in memory_events) / 1000
        
        # NCCL breakdown
        all_gather_time = sum(self._get_device_time(e) for e in all_gather_events) / 1000
        reduce_scatter_time = sum(self._get_device_time(e) for e in reduce_scatter_events) / 1000
        all_reduce_time = sum(self._get_device_time(e) for e in all_reduce_events) / 1000
        
        # Estimate total step time
        # Note: This is an approximation. True overlap would require timeline analysis.
        # For conservative estimate, assume no overlap: total = compute + nccl + memory
        total_step_time = total_compute + total_nccl + total_memory
        
        # Calculate fractions
        compute_fraction = total_compute / total_step_time if total_step_time > 0 else 0
        nccl_fraction = total_nccl / total_step_time if total_step_time > 0 else 0
        
        # GPU utilization (compute time / total time)
        # High utilization means compute dominates (good)
        # Low utilization means waiting for communication or memory (bad)
        gpu_utilization = compute_fraction
        
        # Assess overlap health
        # Good overlap: NCCL time is "hidden" under compute time
        # Poor overlap: NCCL time dominates, creating visible gaps
        nccl_to_compute_ratio = total_nccl / total_compute if total_compute > 0 else float('inf')
        
        if nccl_to_compute_ratio < 0.3:
            overlap_health = "GOOD"
        elif nccl_to_compute_ratio < 0.8:
            overlap_health = "MODERATE"
        else:
            overlap_health = "POOR"
        
        # Identify bottleneck
        if compute_fraction > 0.7:
            bottleneck = "COMPUTE"
        elif nccl_fraction > 0.5:
            bottleneck = "COMMUNICATION"
        else:
            bottleneck = "BALANCED"
        
        return OverlapMetrics(
            total_step_time_ms=total_step_time,
            total_compute_time_ms=total_compute,
            total_nccl_time_ms=total_nccl,
            total_memory_time_ms=total_memory,
            all_gather_time_ms=all_gather_time,
            all_gather_count=len(all_gather_events),
            reduce_scatter_time_ms=reduce_scatter_time,
            reduce_scatter_count=len(reduce_scatter_events),
            all_reduce_time_ms=all_reduce_time,
            all_reduce_count=len(all_reduce_events),
            compute_fraction=compute_fraction,
            nccl_fraction=nccl_fraction,
            gpu_utilization=gpu_utilization,
            overlap_health=overlap_health,
            bottleneck=bottleneck,
        )
    
    def compare_experiments(self, other: 'FSDPOverlapProfiler') -> Dict:
        """
        Compare two experiments (e.g., NVLink vs PCIe).
        
        Args:
            other: Another FSDPOverlapProfiler instance
            
        Returns:
            Dictionary with comparison metrics
        """
        baseline = self.calculate_overlap()
        comparison = other.calculate_overlap()
        
        step_time_delta = comparison.total_step_time_ms - baseline.total_step_time_ms
        step_time_pct = (step_time_delta / baseline.total_step_time_ms * 100) if baseline.total_step_time_ms > 0 else 0
        
        nccl_time_delta = comparison.total_nccl_time_ms - baseline.total_nccl_time_ms
        nccl_time_pct = (nccl_time_delta / baseline.total_nccl_time_ms * 100) if baseline.total_nccl_time_ms > 0 else 0
        
        return {
            "network_tax_ms": step_time_delta,
            "network_tax_pct": step_time_pct,
            "nccl_increase_ms": nccl_time_delta,
            "nccl_increase_pct": nccl_time_pct,
            "baseline_metrics": baseline.to_dict(),
            "comparison_metrics": comparison.to_dict(),
        }
    
    def export_metrics(self, output_path: str):
        """Export metrics to JSON file."""
        metrics = self.calculate_overlap()
        with open(output_path, 'w') as f:
            json.dump(metrics.to_dict(), f, indent=2)
        print(f"Metrics exported to: {output_path}")


# Example usage in profiling script:
"""
from torch.profiler import profile, ProfilerActivity
from training.helper.profiler_analyzer import FSDPOverlapProfiler

# In your training loop:
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=5, warmup=5, active=2),
) as prof:
    for step in range(12):
        # Training step
        loss = train_step(model, data)
        loss.backward()
        optimizer.step()
        
        prof.step()

# Analyze after profiling
analyzer = FSDPOverlapProfiler(prof)
metrics = analyzer.calculate_overlap()
metrics.print_summary()

# Export for blog
analyzer.export_metrics("./output/overlap_metrics.json")
"""
