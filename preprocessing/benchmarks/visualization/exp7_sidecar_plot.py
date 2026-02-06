"""
Cross-Experiment Visualization: Sidecar Results (exp7)

Generates a publication-ready figure showing:

Phase 1: Sidecar-only comparison (Long Context, 50 Users)
  - Latency and Throughput: Monolith vs Sidecar

Phase 2: Cross-Experiment Comparison (100 Turns, matches exp5/exp6)
  - Low Concurrency (20): vLLM vs SGLang (Radix) vs Sidecar
  - High Concurrency (400): vLLM vs SGLang (Radix) vs Sidecar

Usage:
    uv run python -m preprocessing.benchmarks.visualization.exp7_sidecar_plot
"""

import os
import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# DATA FROM EXPERIMENT RESULTS
# Update these values with your actual results
# =============================================================================

# --- PHASE 1: Sidecar-only (50 Users, Long Context ~2500 tokens) ---
P1_LABELS = ["Monolith", "Sidecar"]
P1_LATENCY = [379.14, 212.16]
P1_THROUGHPUT = [118.44, 178.58]

# --- PHASE 2: Cross-Experiment (100 Turns, matches exp5/exp6) ---

# Low Concurrency (20 users)
P2_LOW_LABELS = ["vLLM", "SGLang\n(Radix)", "Sidecar"]
P2_LOW_LATENCY = [150.17, 67.21, 58.83]      # exp5, exp6 (shared), exp7
P2_LOW_THROUGHPUT = [133.75, 245.84, 308.53]

# High Concurrency (400 users)
P2_HIGH_LABELS = ["vLLM", "SGLang\n(Radix)", "Sidecar"]
P2_HIGH_LATENCY = [2745.41, 1780.42, 1006.04]    # exp5, exp6 (shared), exp7
P2_HIGH_THROUGHPUT = [133.99, 227.52, 292.01]

# Output directory
OUTPUT_DIR = "output"


def plot_sidecar_results(output_dir=OUTPUT_DIR):
    """Generate the final cross-experiment visualization."""
    os.makedirs(output_dir, exist_ok=True)

    fig = plt.figure(figsize=(16, 14))

    # Create grid: 3 rows, 2 columns
    # Row 1: Phase 1 (Sidecar only)
    # Row 2: Phase 2 Low Concurrency (Cross-experiment)
    # Row 3: Phase 2 High Concurrency (Cross-experiment)

    # Colors
    colors_p1 = ['#EF5350', '#42A5F5']  # Red=Monolith, Blue=Sidecar
    colors_p2 = ['#FF7043', '#AB47BC', '#42A5F5']  # Orange=vLLM, Purple=SGLang, Blue=Sidecar

    # =========================================================================
    # ROW 1: PHASE 1 - Sidecar Only (Long Context)
    # =========================================================================

    # Phase 1 Latency
    ax1 = fig.add_subplot(3, 2, 1)
    x1 = np.arange(len(P1_LABELS))
    bars1 = ax1.bar(x1, P1_LATENCY, color=colors_p1, width=0.5)
    ax1.set_title(
        "Phase 1: Latency\n(50 Users, Long Context ~2500 tokens)",
        fontsize=11, fontweight='bold'
    )
    ax1.set_xticks(x1)
    ax1.set_xticklabels(P1_LABELS)
    ax1.set_ylabel("Latency (ms) ↓ Lower is Better", fontweight='bold')
    ax1.bar_label(bars1, fmt='%.0f', padding=3)

    # Annotation
    diff = P1_LATENCY[0] - P1_LATENCY[1]
    speedup = P1_LATENCY[0] / P1_LATENCY[1]
    ax1.text(
        0.5, max(P1_LATENCY) * 0.6,
        f'CPU Tax:\n{int(diff)}ms\n({speedup:.1f}x)',
        ha='center', fontsize=10, fontweight='bold',
        bbox=dict(boxstyle="round", fc="lightyellow", ec="orange", alpha=0.8)
    )

    # Phase 1 Throughput
    ax2 = fig.add_subplot(3, 2, 2)
    bars2 = ax2.bar(x1, P1_THROUGHPUT, color=colors_p1, width=0.5)
    ax2.set_title(
        "Phase 1: Throughput\n(50 Users, Long Context ~2500 tokens)",
        fontsize=11, fontweight='bold'
    )
    ax2.set_xticks(x1)
    ax2.set_xticklabels(P1_LABELS)
    ax2.set_ylabel("Throughput (req/s) ↑ Higher is Better", fontweight='bold')
    ax2.bar_label(bars2, fmt='%.0f', padding=3)

    speedup_thr = P1_THROUGHPUT[1] / P1_THROUGHPUT[0]
    ax2.text(
        0.5, max(P1_THROUGHPUT) * 0.5,
        f'{speedup_thr:.1f}x\nhigher',
        ha='center', fontsize=11, fontweight='bold', color='#2E7D32'
    )

    # =========================================================================
    # ROW 2: PHASE 2 LOW CONCURRENCY - Cross Experiment
    # =========================================================================

    x2 = np.arange(len(P2_LOW_LABELS))

    # Low Concurrency Latency
    ax3 = fig.add_subplot(3, 2, 3)
    bars3 = ax3.bar(x2, P2_LOW_LATENCY, color=colors_p2, width=0.5)
    ax3.set_title(
        "Phase 2: Latency @ Low Concurrency (20 Users)\nCross-Experiment: vLLM vs SGLang vs Sidecar",
        fontsize=11, fontweight='bold'
    )
    ax3.set_xticks(x2)
    ax3.set_xticklabels(P2_LOW_LABELS)
    ax3.set_ylabel("Latency (ms) ↓ Lower is Better", fontweight='bold')
    ax3.bar_label(bars3, fmt='%.0f', padding=3)

    # Winner annotation
    winner_idx = np.argmin(P2_LOW_LATENCY)
    ax3.annotate(
        '✓ Best',
        xy=(winner_idx, P2_LOW_LATENCY[winner_idx]),
        xytext=(winner_idx, P2_LOW_LATENCY[winner_idx] + 30),
        ha='center', fontsize=9, color='green', fontweight='bold'
    )

    # Low Concurrency Throughput
    ax4 = fig.add_subplot(3, 2, 4)
    bars4 = ax4.bar(x2, P2_LOW_THROUGHPUT, color=colors_p2, width=0.5)
    ax4.set_title(
        "Phase 2: Throughput @ Low Concurrency (20 Users)\nCross-Experiment: vLLM vs SGLang vs Sidecar",
        fontsize=11, fontweight='bold'
    )
    ax4.set_xticks(x2)
    ax4.set_xticklabels(P2_LOW_LABELS)
    ax4.set_ylabel("Throughput (req/s) ↑ Higher is Better", fontweight='bold')
    ax4.bar_label(bars4, fmt='%.0f', padding=3)

    # Winner annotation
    winner_idx = np.argmax(P2_LOW_THROUGHPUT)
    ax4.annotate(
        '✓ Best',
        xy=(winner_idx, P2_LOW_THROUGHPUT[winner_idx]),
        xytext=(winner_idx, P2_LOW_THROUGHPUT[winner_idx] + 20),
        ha='center', fontsize=9, color='green', fontweight='bold'
    )

    # =========================================================================
    # ROW 3: PHASE 2 HIGH CONCURRENCY - Cross Experiment
    # =========================================================================

    # High Concurrency Latency
    ax5 = fig.add_subplot(3, 2, 5)
    bars5 = ax5.bar(x2, P2_HIGH_LATENCY, color=colors_p2, width=0.5)
    ax5.set_title(
        "Phase 2: Latency @ High Concurrency (400 Users)\nCross-Experiment: vLLM vs SGLang vs Sidecar",
        fontsize=11, fontweight='bold'
    )
    ax5.set_xticks(x2)
    ax5.set_xticklabels(P2_HIGH_LABELS)
    ax5.set_ylabel("Latency (ms) ↓ Lower is Better", fontweight='bold')
    ax5.bar_label(bars5, fmt='%.0f', padding=3)

    # Python ceiling line
    ax5.axhline(y=1000, color='gray', linestyle='--', alpha=0.7)
    ax5.text(
        2.3, 1100, "Python HTTP Floor",
        ha='center', fontsize=8, color='gray', style='italic'
    )

    # Savings annotation
    saved_vs_vllm = P2_HIGH_LATENCY[0] - P2_HIGH_LATENCY[2]
    ax5.annotate(
        f'Sidecar saves\n{int(saved_vs_vllm)}ms\nvs vLLM',
        xy=(2, P2_HIGH_LATENCY[2]),
        xytext=(1.2, P2_HIGH_LATENCY[2] + 600),
        arrowprops=dict(arrowstyle='->', color='green'),
        ha='center', fontsize=9, fontweight='bold', color='#2E7D32'
    )

    # High Concurrency Throughput
    ax6 = fig.add_subplot(3, 2, 6)
    bars6 = ax6.bar(x2, P2_HIGH_THROUGHPUT, color=colors_p2, width=0.5)
    ax6.set_title(
        "Phase 2: Throughput @ High Concurrency (400 Users)\nCross-Experiment: vLLM vs SGLang vs Sidecar",
        fontsize=11, fontweight='bold'
    )
    ax6.set_xticks(x2)
    ax6.set_xticklabels(P2_HIGH_LABELS)
    ax6.set_ylabel("Throughput (req/s) ↑ Higher is Better", fontweight='bold')
    ax6.bar_label(bars6, fmt='%.0f', padding=3)

    # Winner annotation
    winner_idx = np.argmax(P2_HIGH_THROUGHPUT)
    speedup_vs_vllm = P2_HIGH_THROUGHPUT[2] / P2_HIGH_THROUGHPUT[0]
    ax6.text(
        1, max(P2_HIGH_THROUGHPUT) * 0.4,
        f'Sidecar:\n{speedup_vs_vllm:.1f}x\nvs vLLM',
        ha='center', fontsize=10, fontweight='bold', color='#2E7D32'
    )

    # =========================================================================
    # LEGEND AND TITLE
    # =========================================================================

    from matplotlib.patches import Patch

    # Phase 1 legend
    legend1 = [
        Patch(facecolor='#EF5350', label='Monolith (Server processing)'),
        Patch(facecolor='#42A5F5', label='Sidecar (Pre-tokenized)')
    ]

    # Phase 2 legend
    legend2 = [
        Patch(facecolor='#FF7043', label='vLLM (exp5)'),
        Patch(facecolor='#AB47BC', label='SGLang + Radix Cache (exp6)'),
        Patch(facecolor='#42A5F5', label='Sidecar (exp7)')
    ]

    fig.legend(
        handles=legend2, loc='upper center',
        ncol=3, fontsize=10, frameon=True,
        bbox_to_anchor=(0.5, 0.98)
    )

    # Main title
    fig.suptitle(
        "Experiment 7: Sidecar Pre-Tokenization - Cross-Experiment Comparison",
        fontsize=14, fontweight='bold', y=1.01
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save
    output_path = os.path.join(output_dir, "exp7_sidecar_results.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Generated {output_path}")

    return output_path


if __name__ == "__main__":
    plot_sidecar_results()
