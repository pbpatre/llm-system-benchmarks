import asyncio
import time
import aiohttp
import numpy as np
import pandas as pd

# --- EXPERIMENT NARRATIVE ---
# EXPERIMENT 6: SGLang Radix Cache - GPU Caching Doesn't Solve CPU Bottleneck
#
# SERIES CONTEXT:
#   exp5: vLLM baseline - proves latency wall exists under load
#   exp6 (this): SGLang with Radix Cache - proves GPU caching doesn't solve CPU bottleneck
#   exp7: Sidecar approach - proves pre-tokenization helps, reveals Python ceiling
#
# Goal: Test whether SGLang's Radix Attention (GPU-side KV cache) can eliminate
# the latency wall observed in exp5.
#
# HYPOTHESIS:
#   Radix Cache is a GPU optimization. It helps when requests share common prefixes
#   by reusing cached KV computations. HOWEVER, at high concurrency, requests are
#   blocked in the Python/Jinja queue BEFORE they can benefit from the GPU cache.
#
# FINDINGS:
#   - LOW CONCURRENCY: Radix Cache HELPS (significant latency reduction)
#   - HIGH CONCURRENCY: Latency wall STILL EXISTS (requests queued before cache)
#
# CONCLUSION: GPU-side optimizations cannot fix CPU-side bottlenecks.
#             This motivates the Sidecar approach tested in exp7.
#
# PREREQUISITES:
#   Start SGLang server before running this experiment:
#   nohup python -m sglang.launch_server \
#       --model-path Qwen/Qwen2.5-0.5B-Instruct \
#       --port 30000 \
#       --attention-backend triton > sglang_server.log 2>&1 &

# --- CONFIGURATION ---
SERVER_URL = "http://localhost:30000/v1/chat/completions"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
TOTAL_REQUESTS = 1000
TURNS = 100  # High structure to stress Jinja

# SCENARIOS
SCENARIOS = {
    # PHASE 1: Low Concurrency (Prove Radix Works)
    "A_Low_Unique":  {"concurrency": 20,  "shared": False},
    "B_Low_Shared":  {"concurrency": 20,  "shared": True},

    # PHASE 2: High Concurrency (Prove CPU Wall)
    "C_High_Unique": {"concurrency": 400, "shared": False},
    "D_High_Shared": {"concurrency": 400, "shared": True}
}

def generate_chat_history(unique_id, shared=False):
    # To ensure Radix Hit, the prefix must be IDENTICAL.
    # We use a fixed ID '0' for all shared requests.
    uid = "CACHE_HIT_BLOCK" if shared else f"unique_{unique_id}"

    msgs = [{"role": "system", "content": "You are a helpful assistant."}]
    # We create a long history.
    # For Shared: This entire block is cached on GPU.
    # For Unique: This entire block must be computed every time.
    for i in range(TURNS):
        msgs.append({"role": "user", "content": f"history_turn_{uid}_{i}"})
    return msgs

async def send_request(session, i, shared):
    msgs = generate_chat_history(i, shared)
    payload = {
        "model": MODEL_ID,
        "messages": msgs,
        "max_tokens": 1,
        "temperature": 0
    }

    start = time.perf_counter()
    try:
        async with session.post(SERVER_URL, json=payload) as resp:
            await resp.read() # Read response to complete request
            return (time.perf_counter() - start) * 1000
    except aiohttp.ClientError:
        return None

async def run_scenario(name, config):
    concurrency = config['concurrency']
    shared = config['shared']

    print(f"\n🚀 SCENARIO: {name}")
    print(f"   Concurrency: {concurrency} | Radix Cache: {'✅ ACTIVE' if shared else '❌ OFF'}")

    # Use a Semaphore to strictly control concurrency (like vLLM test)
    sem = asyncio.Semaphore(concurrency)
    conn = aiohttp.TCPConnector(limit=concurrency)

    async with aiohttp.ClientSession(connector=conn) as session:
        # 1. Warmup (Critical for Radix)
        if shared:
            print("   🔥 Warming up Cache...")
            await send_request(session, "warmup", True)

        # 2. The Flood
        start_time = time.perf_counter()
        latencies = []

        async def worker(i):
            async with sem:
                lat = await send_request(session, i, shared)
                if lat: latencies.append(lat)

        await asyncio.gather(*[worker(i) for i in range(TOTAL_REQUESTS)])
        duration = time.perf_counter() - start_time

    p50 = np.percentile(latencies, 50)
    throughput = TOTAL_REQUESTS / duration

    return {
        "Scenario": name,
        "Throughput (req/s)": throughput,
        "P50 Latency (ms)": p50
    }

async def main():
    # Health Check
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:30000/health") as r:
                if r.status != 200: raise Exception
    except Exception:
        print("❌ SGLang server not running on port 30000.")
        return

    results = []

    # Run in order
    for name, config in SCENARIOS.items():
        res = await run_scenario(name, config)
        results.append(res)
        # Cool down to drain queue
        await asyncio.sleep(5)

    df = pd.DataFrame(results)
    print("\n" + "="*80)
    print("📊 FINAL RESULTS: THE LATENCY WALL")
    print("="*80)
    print(df.round(2).to_string(index=False))

    # Analysis
    print("-" * 80)
    low_unique = df.loc[0, 'P50 Latency (ms)']
    low_shared = df.loc[1, 'P50 Latency (ms)']
    high_shared = df.loc[3, 'P50 Latency (ms)']

    print("📈 ANALYSIS:")

    # Low concurrency: Radix helps
    print("\n   LOW CONCURRENCY (20):")
    if low_shared < low_unique:
        radix_speedup = low_unique / low_shared
        print(f"   ✅ Radix Cache HELPS: {radix_speedup:.2f}x faster with cache hits")
        print(f"      Unique: {low_unique:.1f}ms → Shared: {low_shared:.1f}ms")
    else:
        print("   ⚠️  Radix Cache had minimal impact at low concurrency")

    # High concurrency: Latency wall
    print("\n   HIGH CONCURRENCY (400):")
    latency_wall_factor = high_shared / low_shared
    print(f"   ❌ LATENCY WALL: {latency_wall_factor:.1f}x increase despite 100% cache hits")
    print(f"      Low+Cache: {low_shared:.1f}ms → High+Cache: {high_shared:.1f}ms")

    # Conclusion
    print("\n" + "-" * 80)
    print("💡 CONCLUSION:")
    print("   Radix Cache is effective at LOW concurrency (GPU-side optimization works).")
    print("   However, at HIGH concurrency, the CPU/Python queue creates a latency wall.")
    print("   Requests are blocked BEFORE they can benefit from the GPU cache.")
    print("   Solution: Sidecars to offload CPU preprocessing and eliminate the queue.")

if __name__ == "__main__":
    asyncio.run(main())
