#!/usr/bin/env python3
"""
Visualization module for LLM Pre-Processing Benchmark Suite.

Generates publication-quality plots for all experiments with consistent styling.
Each plot function can be called independently for blog posts.
"""

import os
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def setup_style():
    """Configure matplotlib and seaborn for consistent styling."""
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 11


def plot_exp1_baseline(
    data: pd.DataFrame, 
    metadata: Dict[str, Any],
    output_path: str = "exp1_baseline_breakdown.png",
    dpi: int = 150
) -> str:
    """
    Generate Experiment 1 visualization: Baseline Breakdown.
    
    Shows horizontal bar chart of preprocessing stage latencies.
    
    Args:
        data: DataFrame with Stage, Mean (ms), etc.
        metadata: Dict with token_count, num_messages, etc.
        output_path: Path to save the plot
        dpi: Resolution for saved image
        
    Returns:
        Path to saved plot file
    """
    setup_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    df = data[data["Stage"] != "TOTAL"]
    colors = ["#e74c3c", "#3498db", "#2ecc71"]
    
    # Calculate total for percentages
    total_ms = df["Mean (ms)"].sum()
    
    bars = ax.barh(df["Stage"], df["Mean (ms)"], color=colors, edgecolor='black')
    
    # Add labels with both ms and percentage
    for bar, val in zip(bars, df["Mean (ms)"]):
        pct = (val / total_ms) * 100
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
               f'{val:.1f} ms ({pct:.1f}%)', va='center', fontweight='bold')
    
    token_count = metadata.get('token_count', 50000)
    ax.set_xlabel("Latency (ms)", fontsize=12)
    ax.set_title(f"Experiment 1: Single Request (~{token_count:,} tokens)",
                fontsize=14, fontweight='bold')
    ax.set_xlim(0, max(df["Mean (ms)"]) * 1.35)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    return output_path


def plot_exp2_scaling(
    data: pd.DataFrame,
    metadata: Dict[str, Any],
    output_path: str = "exp2_scaling_turns.png",
    dpi: int = 150
) -> str:
    """
    Generate Experiment 2 visualization: Scaling with Turns.
    
    Shows two plots:
    1. Raw latencies with error bars (left)
    2. Scaling factors relative to baseline (right)
    
    Args:
        data: DataFrame with Turns, Jinja (ms), Tokenize (ms), etc.
        metadata: Dict with turn_counts, target_tokens, etc.
        output_path: Path to save the plot
        dpi: Resolution for saved image
        
    Returns:
        Path to saved plot file
    """
    setup_style()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    df = data
    
    # Left plot: Raw latencies with error bars
    ax1.errorbar(df["Turns"], df["Jinja (ms)"], yerr=df["Jinja CI95 (ms)"],
                fmt='o-', color="#e74c3c", linewidth=2, markersize=8, 
                capsize=4, capthick=1.5, label="Jinja (±95% CI)")
    ax1.errorbar(df["Turns"], df["Tokenize (ms)"], yerr=df["Tokenize CI95 (ms)"],
                fmt='s-', color="#3498db", linewidth=2, markersize=8,
                capsize=4, capthick=1.5, label="Tokenize (±95% CI)")
    ax1.set_xlabel("Conversation Turns", fontsize=12)
    ax1.set_ylabel("Latency (ms)", fontsize=12)
    ax1.set_title("Latency vs Turns\n(Token count held constant)", fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right plot: Scaling factors
    ax2.plot(df["Turns"], df["Jinja Scaling"], 'o-',
            color="#e74c3c", linewidth=2, markersize=8, label="Jinja Scaling")
    ax2.plot(df["Turns"], df["Tokenize Scaling"], 's-',
            color="#3498db", linewidth=2, markersize=8, label="Tokenize Scaling")
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label="Baseline (1x)")
    ax2.set_xlabel("Conversation Turns", fontsize=12)
    ax2.set_ylabel("Scaling Factor", fontsize=12)
    ax2.set_title("Scaling Factor vs Turns\nJinja ↑ with turns, Tokenize stays ~flat", fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add sample size annotation
    n_samples = df["N Samples"].iloc[0] if "N Samples" in df.columns else "N/A"
    target_tokens = metadata.get("target_tokens", "N/A")
    fig.suptitle(f"Experiment 2: Chat History Tax (~{target_tokens:,} tokens, n={n_samples} measurements/point)", 
                fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    return output_path


def plot_exp3_gpu_wait_time(
    data: pd.DataFrame,
    metadata: Dict[str, Any],
    output_path: str = "exp3_gpu_wait_time.png",
    dpi: int = 150
) -> str:
    """
    Generate Experiment 3 visualization: GPU Wait Time vs Threads.
    
    Args:
        data: DataFrame with Threads, GPU Wait (ms), Jinja (ms), etc.
        metadata: Dict with experiment configuration
        output_path: Path to save the plot
        dpi: Resolution for saved image
        
    Returns:
        Path to saved plot file
    """
    setup_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    df = data
    x = df["Threads"]
    
    ax.plot(x, df["GPU Wait (ms)"], 'o-', color='#9b59b6', linewidth=2, 
           markersize=10, label='GPU Wait Time')
    ax.plot(x, df["Jinja (ms)"], 's--', color='#e74c3c', linewidth=2, 
           markersize=8, label='Jinja Time')
    ax.plot(x, df["Tokenize (ms)"], '^--', color='#3498db', linewidth=2, 
           markersize=8, label='Tokenize Time')
    
    ax.set_xlabel("Thread Count", fontsize=12)
    ax.set_ylabel("Time (ms)", fontsize=12)
    ax.set_title("Experiment 3: GPU Wait Time vs Concurrency\n(Time until batch ready for GPU)",
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(df["Threads"])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    return output_path


def plot_exp3_jinja_overhead(
    data: pd.DataFrame,
    metadata: Dict[str, Any],
    output_path: str = "exp3_jinja_overhead.png",
    dpi: int = 150
) -> str:
    """
    Generate Experiment 3 visualization: Jinja % of GPU Wait Time.
    
    Args:
        data: DataFrame with Threads, Jinja %, etc.
        metadata: Dict with experiment configuration
        output_path: Path to save the plot
        dpi: Resolution for saved image
        
    Returns:
        Path to saved plot file
    """
    setup_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    df = data
    
    bars = ax.bar(df["Threads"].astype(str), df["Jinja %"], 
                 color='#e74c3c', edgecolor='black', alpha=0.8)
    
    for bar, val in zip(bars, df["Jinja %"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
               f'{val:.1f}%', ha='center', fontweight='bold', fontsize=10)
    
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% threshold')
    
    ax.set_xlabel("Thread Count", fontsize=12)
    ax.set_ylabel("Jinja % of GPU Wait Time", fontsize=12)
    ax.set_title("Experiment 3: Jinja Overhead Increases with Concurrency\n⚠️ At high concurrency, Jinja is THE bottleneck!",
                fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(df["Jinja %"]) * 1.2)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    return output_path


def plot_exp4_threading_speedup(
    data: pd.DataFrame,
    metadata: Dict[str, Any],
    output_path: str = "exp4_threading_speedup.png",
    dpi: int = 150
) -> str:
    """
    Generate Experiment 4 visualization: Threading Speedup.
    
    Args:
        data: DataFrame with Threads, Jinja Speedup, Tokenize Speedup
        metadata: Dict with experiment configuration
        output_path: Path to save the plot
        dpi: Resolution for saved image
        
    Returns:
        Path to saved plot file
    """
    setup_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    df = data
    
    ax.plot(df["Threads"], df["Jinja Speedup"], 'o-',
           color="#e74c3c", linewidth=2, markersize=10, label="Jinja (GIL-bound)")
    ax.plot(df["Threads"], df["Tokenize Speedup"], 's-',
           color="#3498db", linewidth=2, markersize=10, label="Tokenize (Rust)")
    ax.plot(df["Threads"], df["Threads"], '--',
           color="#95a5a6", linewidth=1.5, label="Ideal Linear")
    
    ax.set_xlabel("Thread Count", fontsize=12)
    ax.set_ylabel("Speedup Factor", fontsize=12)
    ax.set_title("Experiment 4: Threading Speedup\nRust parallelizes, Python Jinja does NOT",
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(df["Threads"])
    
    # Add annotation for GIL blocking
    max_threads = df["Threads"].max()
    jinja_final = df[df["Threads"] == max_threads]["Jinja Speedup"].values[0]
    ax.annotate(f'GIL Blocked!\n{jinja_final:.1f}x only',
               xy=(max_threads, jinja_final),
               xytext=(max_threads * 0.7, jinja_final + 2),
               fontsize=10, color="#e74c3c",
               arrowprops=dict(arrowstyle='->', color="#e74c3c"))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    return output_path


def generate_all_plots(
    results: List,
    output_dir: str = ".",
    dpi: int = 150
) -> List[str]:
    """
    Generate all visualization plots from experiment results.
    
    Args:
        results: List of ExperimentResult objects
        output_dir: Directory to save plots
        dpi: Resolution for saved images
        
    Returns:
        List of paths to generated plot files
    """
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    plot_files = []
    
    # Get experiment results by ID
    exp1 = next((r for r in results if r.experiment_id == 1), None)
    exp2 = next((r for r in results if r.experiment_id == 2), None)
    exp3 = next((r for r in results if r.experiment_id == 3), None)
    exp4 = next((r for r in results if r.experiment_id == 4), None)
    
    # Plot 1: Baseline Breakdown
    if exp1 is not None and not exp1.data.empty:
        filepath = os.path.join(output_dir, "exp1_baseline_breakdown.png")
        plot_exp1_baseline(exp1.data, exp1.metadata, filepath, dpi)
        plot_files.append(filepath)
        print(f"   ✓ Saved: {filepath}")
    
    # Plot 2: Scaling with Turns
    if exp2 is not None and not exp2.data.empty:
        filepath = os.path.join(output_dir, "exp2_scaling_turns.png")
        plot_exp2_scaling(exp2.data, exp2.metadata, filepath, dpi)
        plot_files.append(filepath)
        print(f"   ✓ Saved: {filepath}")
    
    # Plot 3a: GPU Wait Time
    if exp3 is not None and not exp3.data.empty:
        filepath = os.path.join(output_dir, "exp3_gpu_wait_time.png")
        plot_exp3_gpu_wait_time(exp3.data, exp3.metadata, filepath, dpi)
        plot_files.append(filepath)
        print(f"   ✓ Saved: {filepath}")
        
        # Plot 3b: Jinja Overhead
        filepath = os.path.join(output_dir, "exp3_jinja_overhead.png")
        plot_exp3_jinja_overhead(exp3.data, exp3.metadata, filepath, dpi)
        plot_files.append(filepath)
        print(f"   ✓ Saved: {filepath}")
    
    # Plot 4: Threading Speedup
    if exp4 is not None and not exp4.data.empty:
        filepath = os.path.join(output_dir, "exp4_threading_speedup.png")
        plot_exp4_threading_speedup(exp4.data, exp4.metadata, filepath, dpi)
        plot_files.append(filepath)
        print(f"   ✓ Saved: {filepath}")
    
    return plot_files
