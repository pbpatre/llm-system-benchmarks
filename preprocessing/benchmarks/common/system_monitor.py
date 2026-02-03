"""
System Resource Monitor for benchmarking.

Provides non-blocking monitoring of CPU utilization (per-core and total)
and memory usage during benchmark runs. Designed to have minimal overhead
on the benchmarked workload.
"""

import threading
from typing import Optional, List, Dict

import numpy as np
import psutil

from .data_classes import SystemMetrics


class SystemMonitor:
    """
    Non-blocking system resource monitor that runs in a separate thread.
    
    Captures CPU utilization (per-core and total) and memory usage at
    configurable intervals. Designed to have minimal overhead on the
    benchmarked workload.
    
    Usage:
        with SystemMonitor(interval=0.1) as monitor:
            # ... run workload ...
        stats = monitor.get_stats()
    
    For multiprocessing workloads, set capture_children=True to include
    CPU usage from child processes, or use system_wide=True to capture
    total system CPU (recommended for multiprocessing).
    """
    
    def __init__(
        self, 
        interval: float = 0.1, 
        capture_children: bool = False,
        system_wide: bool = False
    ):
        """
        Initialize the system monitor.
        
        Args:
            interval: Sampling interval in seconds (default 0.1s)
            capture_children: If True, include child process CPU in per-process tracking
            system_wide: If True, use system-wide CPU metrics (best for multiprocessing)
        """
        import time
        self._time = time
        
        self.interval = interval
        self.capture_children = capture_children
        self.system_wide = system_wide
        
        self._active = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Collected samples
        self._cpu_samples: List[List[float]] = []  # Per-core samples
        self._cpu_total_samples: List[float] = []  # Total CPU samples
        self._memory_samples: List[float] = []  # Memory in bytes
        
        # Process handle for memory tracking
        self._process = psutil.Process()
    
    def __enter__(self) -> 'SystemMonitor':
        """Start monitoring when entering context."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop monitoring when exiting context."""
        self.stop()
    
    def start(self) -> None:
        """Start the monitoring thread."""
        self._active = True
        self._cpu_samples = []
        self._cpu_total_samples = []
        self._memory_samples = []
        
        # Prime psutil's CPU measurement (first call returns 0)
        psutil.cpu_percent(percpu=True)
        if not self.system_wide:
            self._process.cpu_percent()
        
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
    
    def stop(self) -> None:
        """Stop the monitoring thread."""
        self._active = False
        if self._thread is not None:
            self._thread.join(timeout=self.interval * 2)
            self._thread = None
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop - runs in separate thread."""
        while self._active:
            try:
                # Capture per-core CPU utilization
                per_core = psutil.cpu_percent(percpu=True)
                
                # Calculate total CPU
                if self.system_wide:
                    # System-wide total (best for multiprocessing)
                    cpu_total = sum(per_core) / len(per_core) if per_core else 0
                else:
                    # Process-specific CPU (may exceed 100% on multi-core)
                    cpu_total = self._process.cpu_percent()
                    if self.capture_children:
                        # Add children's CPU usage
                        try:
                            for child in self._process.children(recursive=True):
                                cpu_total += child.cpu_percent()
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                
                # Capture memory usage
                memory = self._process.memory_info().rss
                if self.capture_children:
                    try:
                        for child in self._process.children(recursive=True):
                            memory += child.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                
                # Store samples thread-safely
                with self._lock:
                    self._cpu_samples.append(per_core)
                    self._cpu_total_samples.append(cpu_total)
                    self._memory_samples.append(memory)
                
            except Exception:
                # Silently ignore monitoring errors to not affect benchmark
                pass
            
            self._time.sleep(self.interval)
    
    def get_stats(self) -> SystemMetrics:
        """
        Get aggregated statistics from collected samples.
        
        Returns:
            SystemMetrics with avg/max CPU, peak memory, and per-core utilization
        """
        with self._lock:
            if not self._cpu_samples:
                return SystemMetrics(
                    avg_cpu_total=0.0,
                    max_cpu_total=0.0,
                    peak_memory_mb=0.0,
                    per_core_utilization=[],
                    sample_count=0
                )
            
            # Calculate per-core averages
            num_cores = len(self._cpu_samples[0]) if self._cpu_samples else 0
            per_core_avg = []
            for core_idx in range(num_cores):
                core_samples = [s[core_idx] for s in self._cpu_samples if len(s) > core_idx]
                per_core_avg.append(np.mean(core_samples) if core_samples else 0.0)
            
            return SystemMetrics(
                avg_cpu_total=np.mean(self._cpu_total_samples) if self._cpu_total_samples else 0.0,
                max_cpu_total=np.max(self._cpu_total_samples) if self._cpu_total_samples else 0.0,
                peak_memory_mb=max(self._memory_samples) / (1024 * 1024) if self._memory_samples else 0.0,
                per_core_utilization=per_core_avg,
                sample_count=len(self._cpu_samples)
            )
    
    def get_raw_samples(self) -> Dict[str, List]:
        """Get raw sample data for detailed analysis."""
        with self._lock:
            return {
                "cpu_per_core": list(self._cpu_samples),
                "cpu_total": list(self._cpu_total_samples),
                "memory_bytes": list(self._memory_samples)
            }
