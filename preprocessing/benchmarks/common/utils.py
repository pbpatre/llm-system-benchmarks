"""
Utility Functions for LLM Benchmarks.

Contains helper functions for batch collation and other common operations
used across experiments.
"""

import time
from typing import List, Tuple, Optional

import torch


def collate_batch(
    input_ids_list: List[List[int]], 
    pad_token_id: int,
    max_length: Optional[int] = None,
    alignment: int = 8
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Collate a list of input_ids into a padded batch tensor.
    
    Production kernels (TensorRT/vLLM) require tensors to be aligned to 8 or 16 bytes
    for optimal memory access patterns. This adds a small but realistic calculation
    overhead to the collation step.
    
    Args:
        input_ids_list: List of token ID sequences
        pad_token_id: Token ID to use for padding
        max_length: Optional explicit max length (aligned up if provided)
        alignment: Byte alignment boundary (default: 8 for production realism)
    
    Returns:
        Tuple of (batch_input_ids, attention_mask, collate_time_ms)
    """
    start = time.perf_counter()
    
    tensors = [torch.tensor(ids, dtype=torch.long) for ids in input_ids_list]
    
    # Determine raw max length from data
    raw_max_length = max(t.shape[0] for t in tensors)
    
    # Apply alignment: pad to next multiple of alignment boundary
    # Formula: (max_length + alignment - 1) // alignment * alignment
    # This mimics production kernels (TensorRT/vLLM) that require aligned tensors
    if max_length is None:
        aligned_length = (raw_max_length + alignment - 1) // alignment * alignment
    else:
        aligned_length = (max_length + alignment - 1) // alignment * alignment
    
    # Pad sequence to aligned length
    batch_input_ids = torch.nn.utils.rnn.pad_sequence(
        tensors, 
        batch_first=True, 
        padding_value=pad_token_id
    )
    
    # If aligned_length is larger than what pad_sequence produced, pad further
    current_length = batch_input_ids.shape[1]
    if aligned_length > current_length:
        padding_needed = aligned_length - current_length
        batch_input_ids = torch.nn.functional.pad(
            batch_input_ids,
            (0, padding_needed),
            value=pad_token_id
        )
    
    attention_mask = (batch_input_ids != pad_token_id).long()
    
    collate_ms = (time.perf_counter() - start) * 1000
    
    return batch_input_ids, attention_mask, collate_ms
