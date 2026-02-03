"""
Data classes for benchmark results and metrics.

These classes provide structured containers for storing and passing
benchmark results between different stages of the experiments.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any

import pandas as pd


@dataclass
class StageLatency:
    """Stores latency breakdown for a single preprocessing request."""
    jinja_ms: float
    tokenize_ms: float
    collate_ms: float
    total_ms: float
    token_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "jinja_ms": self.jinja_ms,
            "tokenize_ms": self.tokenize_ms,
            "collate_ms": self.collate_ms,
            "total_ms": self.total_ms,
            "token_count": self.token_count,
        }


@dataclass
class ConcurrencyMetrics:
    """Stores metrics from concurrency test."""
    total_batch_time_ms: float
    requests_per_second: float
    cpu_user_time_percent: float
    avg_latency_ms: float


@dataclass
class ScalingResult:
    """Stores speedup test results."""
    thread_count: int
    elapsed_time_ms: float
    speedup_factor: float
    is_linear: bool = False


@dataclass 
class ExperimentResult:
    """Container for experiment results."""
    experiment_name: str
    experiment_id: int
    data: pd.DataFrame
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemMetrics:
    """Container for system resource metrics."""
    avg_cpu_total: float  # Average total CPU utilization %
    max_cpu_total: float  # Maximum total CPU utilization %
    peak_memory_mb: float  # Peak memory usage in MB
    per_core_utilization: List[float]  # Per-core average utilization %
    sample_count: int  # Number of samples taken
