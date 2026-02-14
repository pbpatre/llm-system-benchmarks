import torch
from torch.utils.data import Dataset, DataLoader
import time
import argparse

# Configuration
# We simulate a "heavy" transformation to mimic real tokenization/augmentation
class HeavyDataset(Dataset):
    def __init__(self, size=10000, sleep_time=0.01):
        self.size = size
        self.sleep_time = sleep_time
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        # Simulate "Heavy" CPU work (Tokenization, Augmentation, JSON parsing)
        # 10ms per sample is realistic for complex tokenization
        time.sleep(self.sleep_time) 
        return torch.randn(4096) # Dummy tensor

def profile_loader(num_workers, batch_size, cpu_delay):
    print(f"\n🐢 Profiling DataLoader with {num_workers} workers (CPU Delay: {cpu_delay*1000:.1f}ms)...")
    
    dataset = HeavyDataset(size=500, sleep_time=cpu_delay)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    
    # Metrics
    data_times = []
    compute_times = []
    
    # Simulate the GPU Training Loop
    gpu_sim_time = 0.05  # Simulate a fast forward/backward pass (50ms)
    
    start_time = time.time()
    iter_loader = iter(loader)
    
    # Warmup
    _ = next(iter_loader)
    
    t0 = time.time()
    for i in range(20): # Profile 20 steps
        try:
            # 1. MEASURE DATA LOADING (The "Wait")
            batch = next(iter_loader) 
            t1 = time.time()
            data_times.append(t1 - t0)
            
            # 2. MEASURE COMPUTE (The "Work")
            time.sleep(gpu_sim_time) 
            t2 = time.time()
            compute_times.append(t2 - t1)
            
            t0 = time.time()
            
        except StopIteration:
            break
            
    avg_data = sum(data_times) / len(data_times)
    avg_compute = sum(compute_times) / len(compute_times)
    
    print(f"   Avg Data Wait: {avg_data*1000:.2f} ms")
    print(f"   Avg Compute:   {avg_compute*1000:.2f} ms")
    
    if avg_data > 0.01:
        print(f"   🚨 STARVATION DETECTED! GPU is idle for {avg_data*1000:.1f}ms per step.")
    else:
        print(f"   ✅ GPU is fed correctly.")

if __name__ == "__main__":
    # Scenario 1: Single Process (The Default Mistake)
    # Why it fails: The main process has to do EVERYTHING (Load -> Tokenize -> Train)
    profile_loader(num_workers=0, batch_size=32, cpu_delay=0.005)
    
    # Scenario 2: Multi-Process (The "Standard" Fix)
    # Why it helps: Workers prep data in background.
    # Why it might still fail: 'cpu_delay' simulates heavy tokenization. 
    # If (delay * batch_size) / workers > gpu_time, we still starve.
    profile_loader(num_workers=4, batch_size=32, cpu_delay=0.005)