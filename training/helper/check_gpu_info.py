"""
Quick script to check GPU information and memory capacity.
"""
import torch

if torch.cuda.is_available():
    print("=" * 70)
    print("GPU Information")
    print("=" * 70)
    
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs: {num_gpus}\n")
    
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        total_memory_gb = props.total_memory / (1024**3)
        
        print(f"GPU {i}: {props.name}")
        print(f"  Total Memory: {total_memory_gb:.2f} GB")
        print(f"  Compute Capability: {props.major}.{props.minor}")
        print(f"  Multi-Processors: {props.multi_processor_count}")
        print()
        
        # Check current allocation
        torch.cuda.set_device(i)
        allocated = torch.cuda.memory_allocated(i) / (1024**3)
        reserved = torch.cuda.memory_reserved(i) / (1024**3)
        free = (props.total_memory - torch.cuda.memory_reserved(i)) / (1024**3)
        
        print(f"  Current State:")
        print(f"    Allocated: {allocated:.2f} GB")
        print(f"    Reserved:  {reserved:.2f} GB")
        print(f"    Free:      {free:.2f} GB")
        print()
    
    print("=" * 70)
else:
    print("No CUDA GPUs available")
