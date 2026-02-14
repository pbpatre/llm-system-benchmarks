import argparse
import os

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer

# Configuration
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B" # or "bert-base-uncased" for speed
CONTEXT_LEN = 4096
BATCH_SIZE = 32

def get_dataset():
    print(f"📚 Loading Alpaca dataset...")
    # Using a small slice for speed
    ds = load_dataset("tatsu-lab/alpaca", split="train[:2000]")
    return ds

def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def simulate_naive_batching(dataset, tokenizer):
    """
    Simulates standard PyTorch DataLoader behavior:
    1. Grab BATCH_SIZE items.
    2. Pad them all to CONTEXT_LEN (or max in batch, but we enforce fixed context here for the 'Tax' demo).
    """
    print("🐢 Simulating Naive Batching...")
    batch_grid = np.zeros((BATCH_SIZE, CONTEXT_LEN))
    
    # Take the first N samples
    samples = [f"{x['instruction']} {x['input']} {x['output']}" for x in dataset]
    samples = samples[:BATCH_SIZE]
    
    for i, text in enumerate(samples):
        # Tokenize without padding first to get raw length
        tokens = tokenizer(text, truncation=True, max_length=CONTEXT_LEN)["input_ids"]
        length = len(tokens)
        
        # Fill the grid: 1 = Real Data, 0 = Padding
        batch_grid[i, :length] = 1 
        # The rest remains 0 (Padding)
        
    return batch_grid

def simulate_packed_batching(dataset, tokenizer):
    """
    Simulates 'Packed' behavior (e.g., Torchtune/MosaicML):
    1. Tokenize all samples (keeping them whole).
    2. Greedily bin-pack complete samples into rows of CONTEXT_LEN.
    3. Leftover space at the end of a row becomes padding.
    """
    print("🐇 Simulating Packed Batching...")
    batch_grid = np.zeros((BATCH_SIZE, CONTEXT_LEN))
    
    samples = [f"{x['instruction']} {x['input']} {x['output']}" for x in dataset]
    
    # 1. Tokenize all samples and get their lengths (keep samples whole)
    sample_lengths = []
    for text in samples:
        tokens = tokenizer(text, truncation=True, max_length=CONTEXT_LEN)["input_ids"]
        # +1 for the EOS separator between packed samples
        sample_lengths.append(len(tokens) + 1)
    
    # 2. Greedily pack complete samples into each row
    sample_idx = 0
    for row in range(BATCH_SIZE):
        filled = 0
        while sample_idx < len(sample_lengths):
            length = sample_lengths[sample_idx]
            if filled + length <= CONTEXT_LEN:
                batch_grid[row, filled:filled + length] = 1
                filled += length
                sample_idx += 1
            else:
                break  # This sample doesn't fit; move to next row
        # Remaining slots in this row stay 0 (padding)
        
    return batch_grid

def plot_heatmap(naive_grid, packed_grid):
    print("🎨 Generating Heatmap...")
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Color map: Red (0/Pad) to Green (1/Data)
    cmap = sns.color_palette("RdYlGn", as_cmap=True)

    # Plot Naive
    sns.heatmap(naive_grid, ax=axes[0], cmap=cmap, vmin=0, vmax=1, cbar=False, xticklabels=False, yticklabels=False)
    naive_efficiency = np.mean(naive_grid) * 100
    axes[0].set_title(f"Naive Batching\nEfficiency: {naive_efficiency:.1f}% (The 'Tax')", fontsize=16)
    axes[0].set_xlabel("Context Length (4096)", fontsize=12)
    axes[0].set_ylabel(f"Batch Size ({BATCH_SIZE})", fontsize=12)

    # Plot Packed
    sns.heatmap(packed_grid, ax=axes[1], cmap=cmap, vmin=0, vmax=1, cbar=False, xticklabels=False, yticklabels=False)
    packed_efficiency = np.mean(packed_grid) * 100
    axes[1].set_title(f"Sequence Packing\nEfficiency: {packed_efficiency:.1f}%", fontsize=16)
    axes[1].set_xlabel("Context Length (4096)", fontsize=12)

    plt.tight_layout()
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "padding_tax_heatmap.png")
    plt.savefig(output_file)
    print(f"✅ Saved visualization to {output_file}")

if __name__ == "__main__":
    ds = get_dataset()
    tok = get_tokenizer()
    
    naive = simulate_naive_batching(ds, tok)
    packed = simulate_packed_batching(ds, tok)
    
    plot_heatmap(naive, packed)