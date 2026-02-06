import asyncio
import time
import numpy as np
import pandas as pd
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from transformers import AutoTokenizer
import pynvml

# --- EXPERIMENT NARRATIVE ---
# EXPERIMENT 5: vLLM Baseline - The Latency Wall
#
# SERIES CONTEXT:
#   exp5 (this): vLLM baseline - proves latency wall exists under load
#   exp6: SGLang with Radix Cache - proves GPU caching doesn't solve CPU bottleneck
#   exp7: Sidecar approach - proves pre-tokenization helps, reveals Python ceiling
#
# Goal: Characterize the "Throughput vs. Latency" trade-off in vLLM's architecture.
#
# vLLM is designed for MAXIMUM THROUGHPUT. It groups requests to keep the GPU
# 100% busy. This is excellent for batch processing and heavy loads.
#
# However, this architecture creates a "Latency Wall" at high concurrency.
# Because the Python Event Loop (CPU) manages the queue, a massive backlog
# of requests creates Head-of-Line blocking. The GPU is happy (busy), but
# the User is unhappy (waiting for the queue to drain).
#
# This experiment quantifies that trade-off and establishes the baseline
# for comparison with SGLang (exp6) and Sidecar (exp7) approaches.

# --- CONFIGURATION ---
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
TOTAL_REQUESTS = 1000

# WORKLOAD:
# 100 turns of history. This represents a complex RAG/Agent workload that
# requires significant CPU Preprocessing (Jinja2) before GPU Inference.
TURNS = 100
WORDS = 1

# SCENARIOS:
SCENARIOS = {
    # Scenario A: "Interactive Mode"
    # 20 concurrent users. The CPU queue is short.
    # Expectation: Low Latency, Moderate Throughput.
    "A_Interactive": 20,

    # Scenario B: "Saturation Mode"
    # 400 concurrent users. The queue is full.
    # Expectation: Max GPU Util (Success), but High Latency (Trade-off).
    "B_Saturation":  400
}

class GPUMonitor:
    def __init__(self, interval=0.05):
        self.interval = interval
        self.running = False
        self.util_samples = []
        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.available = True
        except Exception:
            self.available = False

    async def start(self):
        if not self.available: return
        self.running = True
        while self.running:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                self.util_samples.append(util.gpu)
            except Exception:
                pass
            await asyncio.sleep(self.interval)

    def stop(self):
        self.running = False
        valid = [x for x in self.util_samples if x > 0]
        return np.mean(valid) if valid else 0.0

def generate_complex_prompt(unique_id):
    # Generates a structure-heavy prompt (high CPU cost)
    msgs = [{"role": "system", "content": "System"}]
    for i in range(TURNS):
        msgs.append({"role": "user", "content": f"turn_{unique_id}_{i}"})
    return msgs

async def run_scenario(name, concurrency, engine, tokenizer):
    print(f"\n🚀 SCENARIO: {name} (Concurrency={concurrency})")
    print(f"   Objective: Measure Latency impact when Queue Depth is {concurrency}")

    gpu_mon = GPUMonitor()
    monitor_task = asyncio.create_task(gpu_mon.start())

    sampling_params = SamplingParams(max_tokens=1, ignore_eos=True)
    sem = asyncio.Semaphore(concurrency)
    results = []

    start_time = time.perf_counter()

    async def worker(i):
        async with sem:
            msgs = generate_complex_prompt(i)
            prompt = tokenizer.apply_chat_template(msgs, tokenize=False)

            t0 = time.perf_counter()
            try:
                gen = engine.generate(prompt, sampling_params, request_id=f"{name}_{i}")
                async for _ in gen: break
                # Record Time-to-First-Token (User Wait Time)
                results.append(time.perf_counter() - t0)
            except Exception:
                pass

    await asyncio.gather(*[worker(i) for i in range(TOTAL_REQUESTS)])
    duration = time.perf_counter() - start_time

    gpu_mon.stop()
    await monitor_task

    avg_gpu = np.mean(gpu_mon.util_samples) if gpu_mon.util_samples else 0
    throughput = TOTAL_REQUESTS / duration
    p50 = np.percentile(results, 50) * 1000 if results else 0

    return {
        "Scenario": name,
        "Concurrency": concurrency,
        "Throughput (req/s)": throughput,
        "Avg GPU Util %": avg_gpu,
        "P50 Latency (ms)": p50
    }

async def main():
    print(f"Initializing vLLM ({MODEL_ID})...")
    # vLLM Config: Optimized for Throughput
    engine_args = AsyncEngineArgs(
        model=MODEL_ID,
        tensor_parallel_size=1,
        dtype="half",
        gpu_memory_utilization=0.6,
        enforce_eager=True,
        enable_prefix_caching=False
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print("Warming up...")
    await asyncio.sleep(2)

    metrics_a = await run_scenario("A_Interactive", SCENARIOS["A_Interactive"], engine, tokenizer)

    print("Draining queue...")
    await asyncio.sleep(5)

    metrics_b = await run_scenario("B_Saturation", SCENARIOS["B_Saturation"], engine, tokenizer)

    df = pd.DataFrame([metrics_a, metrics_b])
    print("\n" + "="*80)
    print("📊 FINAL RESULTS: THE ARCHITECTURAL TRADE-OFF")
    print("="*80)
    print(df.round(2).to_string(index=False))

    print("-" * 80)

    # --- INTERPRETATION LOGIC ---
    latency_increase = metrics_b['P50 Latency (ms)'] / metrics_a['P50 Latency (ms)']
    gpu_util_b = metrics_b['Avg GPU Util %']

    if gpu_util_b > 90 and latency_increase > 5:
        print("✅ HYPOTHESIS CONFIRMED: vLLM is working as designed.")
        print(f"   1. Throughput Success: GPU Util remained high ({gpu_util_b:.1f}%) under load.")
        print(f"   2. Latency Trade-off: User wait time increased by {latency_increase:.1f}x.")
        print("   Conclusion: CPU bottleneck didn't starve the GPU, but it blocked the user.")
        print("   This highlights where SGLang (Radix) or Sidecars fit in: eliminating the queue.")
    else:
        print("⚠️  Results Inconclusive. Check hardware limits.")

if __name__ == "__main__":
    asyncio.run(main())
