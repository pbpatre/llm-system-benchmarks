#!/usr/bin/env python3
"""
BenchmarkSuite: Orchestrates all experiments in the LLM Pre-Processing Benchmark Suite.

This module provides a unified interface to run all experiments, generate visualizations,
and export results. It can be used programmatically or via the CLI runner.
"""

import os
from typing import Dict, Any, List, Optional

import pandas as pd

from .common import CONFIG, ExperimentResult, get_quick_config
from .experiments import (
    Experiment1Baseline,
    Experiment2Scaling,
    Experiment3Concurrency,
    Experiment4Threading,
)
from .visualization import generate_all_plots


class BenchmarkSuite:
    """
    Production-grade benchmark suite for analyzing LLM preprocessing bottlenecks.
    
    This suite proves that Python's Global Interpreter Lock (GIL) and Jinja2 templating
    are major bottlenecks in high-throughput inference servers.
    
    Experiments:
        1. Baseline: Breakdown latency for a single large request
        2. Scaling: Show Jinja slowdown with conversation turns
        3. Concurrency: Show Jinja % increasing with thread count
        4. Threading: Prove Rust releases GIL, Python doesn't
    
    Usage:
        # Run all experiments
        suite = BenchmarkSuite()
        suite.run_all()
        
        # Run specific experiments
        suite = BenchmarkSuite()
        suite.run_experiment(1)  # Just baseline
        suite.run_experiment(3)  # Just concurrency
        suite.generate_plots()
        suite.save_results()
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the benchmark suite.
        
        Args:
            config: Configuration dictionary. Uses default CONFIG if not provided.
        """
        self.config = config or CONFIG
        self.results: List[ExperimentResult] = []
        self.all_results_df: Optional[pd.DataFrame] = None
    
    def run_experiment(self, experiment_id: int) -> Optional[ExperimentResult]:
        """
        Run a specific experiment by ID.
        
        Args:
            experiment_id: Experiment number (1-4)
            
        Returns:
            ExperimentResult or None if invalid ID
        """
        experiment_classes = {
            1: Experiment1Baseline,
            2: Experiment2Scaling,
            3: Experiment3Concurrency,
            4: Experiment4Threading,
        }
        
        if experiment_id not in experiment_classes:
            print(f"Invalid experiment ID: {experiment_id}. Must be 1-4.")
            return None
        
        experiment_class = experiment_classes[experiment_id]
        experiment = experiment_class(self.config)
        result = experiment.run()
        
        # Add to results list, replacing any existing result with same ID
        self.results = [r for r in self.results if r.experiment_id != experiment_id]
        self.results.append(result)
        self.results.sort(key=lambda r: r.experiment_id)
        
        return result
    
    def run_experiments(self, experiment_ids: List[int]) -> List[ExperimentResult]:
        """
        Run multiple specific experiments.
        
        Args:
            experiment_ids: List of experiment numbers to run
            
        Returns:
            List of ExperimentResult objects
        """
        results = []
        for exp_id in experiment_ids:
            result = self.run_experiment(exp_id)
            if result is not None:
                results.append(result)
        return results
    
    def generate_plots(self, output_dir: str = ".") -> List[str]:
        """
        Generate visualization plots for all completed experiments.
        
        Args:
            output_dir: Directory to save plot files
            
        Returns:
            List of paths to generated plot files
        """
        return generate_all_plots(
            self.results, 
            output_dir, 
            dpi=self.config.get("plot_dpi", 150)
        )
    
    def save_results(self, output_path: str = None) -> str:
        """
        Save all results to CSV.
        
        Args:
            output_path: Path to save CSV file
            
        Returns:
            Path to saved file
        """
        output_path = output_path or self.config.get("output_csv", "results.csv")
        
        print(f"\n   Saving results to: {output_path}")
        
        all_rows = []
        
        for result in self.results:
            if result.data.empty:
                continue
            df = result.data.copy()
            df["Experiment"] = result.experiment_name
            df["Experiment_ID"] = result.experiment_id
            all_rows.append(df)
        
        if all_rows:
            combined_df = pd.concat(all_rows, ignore_index=True)
            combined_df.to_csv(output_path, index=False)
            self.all_results_df = combined_df
            print(f"   ✓ Saved {len(combined_df)} rows")
        else:
            print("   ⚠️  No results to save")
            
        return output_path
    
    def run_all(self, output_dir: str = ".") -> None:
        """
        Run all experiments and generate outputs.
        
        Args:
            output_dir: Directory to save plots and CSV
        """
        print("\n" + "="*70)
        print("🚀 LLM PRE-PROCESSING BOTTLENECK BENCHMARK SUITE")
        print("="*70)
        print(f"   Model: {self.config['model_name']}")
        print(f"   Target tokens: {self.config['target_tokens']:,}")
        print(f"   CPU cores: {os.cpu_count()}")
        print("="*70)
        
        # Run all experiments in order
        for exp_id in range(1, 5):
            self.run_experiment(exp_id)
        
        # Generate visualizations
        plot_files = self.generate_plots(output_dir)
        
        # Save results
        csv_path = os.path.join(output_dir, self.config.get("output_csv", "results.csv"))
        self.save_results(csv_path)
        
        # Print summary
        print("\n" + "="*70)
        print("📊 BENCHMARK COMPLETE")
        print("="*70)
        print(f"   Results CSV: {csv_path}")
        print(f"   Generated plots: {len(plot_files)}")
        for pf in plot_files:
            print(f"      - {pf}")
    
    def get_results_dataframe(self) -> Optional[pd.DataFrame]:
        """Get combined results as a pandas DataFrame."""
        return self.all_results_df
    
    def get_experiment_result(self, experiment_id: int) -> Optional[ExperimentResult]:
        """Get result for a specific experiment."""
        return next((r for r in self.results if r.experiment_id == experiment_id), None)
