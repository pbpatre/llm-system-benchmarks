"""
PCIe Transfer Bottleneck Benchmark
===================================
Measures CPU <-> GPU transfer bandwidth to surface bottlenecks in
your training data pipeline.

Tests:
  - Pinned vs pageable memory  (DMA vs staged copy)
  - Host->Device (H2D) and Device->Host (D2H)
  - Multiple tensor sizes      (latency-bound vs throughput-bound)
  - All 4 combinations of (pin_memory, non_blocking) to show
    that the real win is CPU/GPU overlap, not raw bandwidth

Key takeaway: DataLoader(..., pin_memory=True) gives you the
pinned speedup for free — always enable it.
"""

import torch
import time
import sys
import os
import statistics

# ── Configuration ────────────────────────────────────────────────

WARMUP_ITERS = 5
BENCH_ITERS = 30

# Decimal bytes so results compare directly with PCIe spec sheets (GB/s).
TRANSFER_SIZES = [
    ("1 MB",    1_000_000),
    ("16 MB",  16_000_000),
    ("256 MB", 256_000_000),
    ("1 GB",   1_000_000_000),
]

# Theoretical unidirectional peaks after 128b/130b encoding overhead
PCIE_PEAKS = {
    "3.0 x16": 15.75,
    "4.0 x16": 31.51,
    "5.0 x16": 63.02,
}

# Peak pinned memory the benchmark will use (largest size, pinned src + dst)
_MAX_PINNED_BYTES = TRANSFER_SIZES[-1][1] * 2


# ── Environment detection ────────────────────────────────────────

def _read_cgroup_file(path):
    """Try to read a cgroup file, return stripped content or None."""
    try:
        with open(path) as f:
            return f.read().strip()
    except (FileNotFoundError, PermissionError):
        return None


def _detect_container_env():
    """Detect cgroup memory/CPU limits and Kubernetes pod context.

    Returns a dict with:
      in_container  (bool)
      mem_limit_gb  (float | None) — cgroup memory limit in GB
      cpu_quota     (float | None) — effective CPU cores allowed
      warnings      (list[str])    — issues that could skew results
    """
    info = {"in_container": False, "mem_limit_gb": None, "cpu_quota": None, "warnings": []}

    # ── Detect container ──
    if os.path.exists("/.dockerenv") or os.path.exists("/run/.containerenv"):
        info["in_container"] = True
    cgroup = _read_cgroup_file("/proc/1/cgroup") or ""
    if any(kw in cgroup for kw in ("kubepods", "docker", "containerd")):
        info["in_container"] = True

    if not info["in_container"]:
        return info

    # ── cgroup v2 (unified) ──
    mem_max = _read_cgroup_file("/sys/fs/cgroup/memory.max")
    if mem_max and mem_max != "max":
        info["mem_limit_gb"] = int(mem_max) / 1e9

    cpu_max = _read_cgroup_file("/sys/fs/cgroup/cpu.max")
    if cpu_max and cpu_max != "max":
        parts = cpu_max.split()
        if len(parts) == 2 and parts[0] != "max":
            info["cpu_quota"] = int(parts[0]) / int(parts[1])

    # ── cgroup v1 fallback ──
    if info["mem_limit_gb"] is None:
        v1_mem = _read_cgroup_file("/sys/fs/cgroup/memory/memory.limit_in_bytes")
        if v1_mem:
            limit = int(v1_mem)
            # Kernel reports a huge number when unlimited
            if limit < 2**62:
                info["mem_limit_gb"] = limit / 1e9

    if info["cpu_quota"] is None:
        quota = _read_cgroup_file("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
        period = _read_cgroup_file("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
        if quota and period and int(quota) > 0:
            info["cpu_quota"] = int(quota) / int(period)

    # ── Warnings ──
    if info["mem_limit_gb"] is not None:
        pinned_gb = _MAX_PINNED_BYTES / 1e9
        if pinned_gb > info["mem_limit_gb"] * 0.5:
            info["warnings"].append(
                f"Pinned memory ({pinned_gb:.1f} GB) is >{50}% of cgroup memory limit "
                f"({info['mem_limit_gb']:.1f} GB). OOM kill risk — consider smaller "
                f"TRANSFER_SIZES or raising the pod memory limit."
            )

    if info["cpu_quota"] is not None and info["cpu_quota"] < 2.0:
        info["warnings"].append(
            f"CPU quota is only {info['cpu_quota']:.1f} cores. CFS throttling during "
            f"transfers will inflate wall-clock times (makes PCIe look slower than it is)."
        )

    return info


# ── Helpers ──────────────────────────────────────────────────────

def print_system_info():
    """Print GPU / CUDA details and container environment for reproducibility."""
    props = torch.cuda.get_device_properties(0)
    print(f"GPU:     {props.name}")
    print(f"VRAM:    {props.total_memory / 1e9:.1f} GB")
    print(f"CUDA:    {torch.version.cuda}")
    print(f"PyTorch: {torch.__version__}")
    print()

    # ── Container / pod awareness ──
    env = _detect_container_env()
    if env["in_container"]:
        print("Environment: container (Kubernetes / Docker detected)")
        if env["mem_limit_gb"] is not None:
            print(f"  cgroup memory limit: {env['mem_limit_gb']:.1f} GB")
        else:
            print("  cgroup memory limit: unlimited")
        if env["cpu_quota"] is not None:
            print(f"  cgroup CPU quota:    {env['cpu_quota']:.1f} cores")
        else:
            print(f"  cgroup CPU quota:    unlimited ({os.cpu_count()} visible)")
        print()

        if env["warnings"]:
            for w in env["warnings"]:
                print(f"  WARNING: {w}")
            print()
    else:
        print(f"Environment: bare metal / VM ({os.cpu_count()} CPUs visible)")
        print()

    print("Reference — theoretical PCIe peaks (unidirectional):")
    for gen, bw in PCIE_PEAKS.items():
        print(f"  PCIe {gen}:  {bw:.2f} GB/s")
    print()


def _measure(transfer_fn, data_gb, iters):
    """Run transfer_fn `iters` times, return list of GB/s measurements.

    Uses perf_counter + synchronize barriers to capture the FULL
    wall-clock cost — including any CPU-side staging copies that
    CUDA events would miss for pageable memory.
    """
    speeds = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        transfer_fn()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        if elapsed > 0:
            speeds.append(data_gb / elapsed)
    return speeds


def fmt(speeds):
    """Format a list of GB/s values into a compact summary line."""
    mu = statistics.mean(speeds)
    sd = statistics.stdev(speeds) if len(speeds) > 1 else 0.0
    return f"{mu:7.2f} +/- {sd:4.2f} GB/s  (min {min(speeds):.2f}, max {max(speeds):.2f})"


# ── Benchmarks ───────────────────────────────────────────────────

def benchmark_h2d(size_bytes, pinned):
    """Host -> Device transfer. Returns list of per-iteration GB/s.

    Always uses non_blocking=False so the timing captures pure transfer
    bandwidth, not CPU queuing speed. This keeps the pinned-vs-pageable
    comparison fair.
    """
    n = size_bytes // 4  # float32 = 4 bytes per element
    data_gb = size_bytes / 1e9

    src = torch.randn(n, dtype=torch.float32)
    if pinned:
        src = src.pin_memory()

    def transfer():
        src.to("cuda", non_blocking=False)

    # Warmup — first calls initialise CUDA allocator, page tables, etc.
    _measure(transfer, data_gb, WARMUP_ITERS)
    speeds = _measure(transfer, data_gb, BENCH_ITERS)

    del src
    torch.cuda.empty_cache()
    return speeds


def benchmark_d2h(size_bytes, pinned_dst):
    """Device -> Host transfer.

    When pinned_dst=True, copies into a pre-allocated pinned CPU tensor.
    This is the fast path used by checkpointing and DeepSpeed ZeRO-Offload.
    When False, uses the default .to("cpu") which goes through a pageable
    staging buffer.
    """
    n = size_bytes // 4
    data_gb = size_bytes / 1e9

    src = torch.randn(n, dtype=torch.float32, device="cuda")

    if pinned_dst:
        dst = torch.empty(n, dtype=torch.float32).pin_memory()

        def transfer():
            dst.copy_(src)
    else:
        def transfer():
            src.to("cpu")

    _measure(transfer, data_gb, WARMUP_ITERS)
    speeds = _measure(transfer, data_gb, BENCH_ITERS)

    del src
    if pinned_dst:
        del dst
    torch.cuda.empty_cache()
    return speeds


# ── Overlap experiment ────────────────────────────────────────────

# For the overlap test, use sizes that represent realistic training batches
# plus one large size to see the full effect.
OVERLAP_SIZES = [
    ("16 MB",  16_000_000),
    ("256 MB", 256_000_000),
]

# All 4 combinations of (pinned, non_blocking)
_COMBOS = [
    (False, False, "pageable + blocking    "),
    (False, True,  "pageable + non_blocking"),   # silently falls back to blocking
    (True,  False, "pinned   + blocking    "),
    (True,  True,  "pinned   + non_blocking"),   # the real async DMA path
]


def _calibrate_cpu_work(target_ms=5.0):
    """Find how many iterations of CPU work take ~target_ms.

    We use a torch CPU matrix multiply as the workload because it's
    representative of real preprocessing (tokenisation, augmentation)
    and not trivially optimised away by the compiler.
    """
    work_tensor = torch.randn(256, 256)
    # Binary search for the right iteration count
    iters = 100
    for _ in range(10):
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = work_tensor @ work_tensor
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if abs(elapsed_ms - target_ms) < 0.5:
            break
        iters = max(1, int(iters * target_ms / elapsed_ms))
    return work_tensor, iters


def _do_cpu_work(work_tensor, iters):
    """Execute a fixed amount of CPU work (matrix multiplies)."""
    for _ in range(iters):
        _ = work_tensor @ work_tensor


def benchmark_overlap(size_bytes):
    """Test all 4 (pinned, non_blocking) combinations and PROVE whether
    the CPU is truly free during DMA by doing real CPU work.

    Method:
      1. Measure baseline CPU work time (no transfer happening).
      2. For each combo, start a transfer, then immediately do the
         SAME CPU work, then synchronize.
      3. If total_time ≈ max(transfer_time, cpu_work_time), the CPU
         and DMA ran in TRUE parallel — the overlap is real.
         If total_time ≈ transfer_time + cpu_work_time, they ran
         sequentially — the CPU was NOT free.

    This avoids relying on synchronize() spin behaviour and actually
    measures physical CPU availability.
    """
    n = size_bytes // 4
    data_gb = size_bytes / 1e9

    # Calibrate CPU work to ~5ms (well within the DMA window for 256MB)
    work_tensor, work_iters = _calibrate_cpu_work(target_ms=5.0)

    # Measure baseline CPU work time (no transfer)
    baseline_times = []
    for _ in range(BENCH_ITERS):
        t0 = time.perf_counter()
        _do_cpu_work(work_tensor, work_iters)
        baseline_times.append((time.perf_counter() - t0) * 1000)
    cpu_work_ms = statistics.mean(baseline_times)
    print(f"    CPU work baseline: {cpu_work_ms:.2f} ms "
          f"({work_iters} matmuls of 256x256)\n")

    for pinned, non_blocking, label in _COMBOS:
        src = torch.randn(n, dtype=torch.float32)
        if pinned:
            src = src.pin_memory()

        # Warmup
        for _ in range(WARMUP_ITERS):
            _ = src.to("cuda", non_blocking=non_blocking)
            torch.cuda.synchronize()

        # Measure transfer-only time (no CPU work)
        transfer_only_times = []
        for _ in range(BENCH_ITERS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = src.to("cuda", non_blocking=non_blocking)
            torch.cuda.synchronize()
            transfer_only_times.append((time.perf_counter() - t0) * 1000)

        # Measure transfer + CPU work done together
        overlap_times = []
        for _ in range(BENCH_ITERS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = src.to("cuda", non_blocking=non_blocking)
            _do_cpu_work(work_tensor, work_iters)   # do CPU work RIGHT AFTER .to()
            torch.cuda.synchronize()
            overlap_times.append((time.perf_counter() - t0) * 1000)

        xfer_ms = statistics.mean(transfer_only_times)
        both_ms = statistics.mean(overlap_times)

        # If truly parallel:  both ≈ max(xfer, cpu_work)
        # If sequential:      both ≈ xfer + cpu_work
        parallel_expected = max(xfer_ms, cpu_work_ms)
        sequential_expected = xfer_ms + cpu_work_ms
        # How close to perfect overlap? 1.0 = fully parallel, 0.0 = fully sequential
        if sequential_expected - parallel_expected > 0.01:
            overlap_ratio = (sequential_expected - both_ms) / (sequential_expected - parallel_expected)
            overlap_ratio = max(0.0, min(1.0, overlap_ratio))
        else:
            overlap_ratio = 0.0

        verdict = "PARALLEL (CPU is truly free)" if overlap_ratio > 0.8 else \
                  "PARTIAL overlap" if overlap_ratio > 0.3 else \
                  "SEQUENTIAL (CPU is blocked)"

        print(f"    {label}:")
        print(f"      Transfer only:     {xfer_ms:8.2f} ms")
        print(f"      CPU work only:     {cpu_work_ms:8.2f} ms")
        print(f"      Both together:     {both_ms:8.2f} ms")
        print(f"      If sequential:     {sequential_expected:8.2f} ms")
        print(f"      If parallel:       {parallel_expected:8.2f} ms")
        print(f"      Overlap score:     {overlap_ratio:.0%}  → {verdict}")
        print()

        del src
        torch.cuda.empty_cache()


def run_overlap_suite():
    """Run the 4-combination overlap experiment for key tensor sizes."""
    W = 65
    print("=" * W)
    print("  H2D OVERLAP — proving CPU is truly free during DMA")
    print("=" * W)
    print()
    print("  Method: start a DMA transfer, immediately do CPU matrix multiplies,")
    print("  then check if total time ≈ max(transfer, cpu_work) [parallel]")
    print("  or total time ≈ transfer + cpu_work [sequential].")
    print()

    for label, size in OVERLAP_SIZES:
        print(f"  ── {label} {'─' * (W - len(label) - 6)}")
        print()
        benchmark_overlap(size)

    print("  KEY: only pinned + non_blocking achieves true parallelism.")
    print("  The CPU physically runs your code while DMA hardware transfers data.")
    print()


# ── Suite runner ─────────────────────────────────────────────────

def run_direction(title, bench_fn):
    """Benchmark pinned vs pageable for every tensor size in one direction."""
    W = 65
    print("=" * W)
    print(f"  {title}")
    print("=" * W)

    for label, size in TRANSFER_SIZES:
        sp_base = bench_fn(size, False)
        sp_pin  = bench_fn(size, True)
        speedup = statistics.mean(sp_pin) / statistics.mean(sp_base)

        print(f"\n  {label}:")
        print(f"    Pageable:  {fmt(sp_base)}")
        print(f"    Pinned:    {fmt(sp_pin)}")
        print(f"    Speedup:   {speedup:.2f}x")

    print()


# ── Main ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("No CUDA GPU found — cannot benchmark PCIe transfers.")
        sys.exit(1)

    print()
    print_system_info()

    run_direction(
        "HOST -> DEVICE (H2D) — DataLoader / batch transfer path",
        benchmark_h2d,
    )
    run_direction(
        "DEVICE -> HOST (D2H) — Checkpointing / gradient offload path",
        benchmark_d2h,
    )

    run_overlap_suite()

    # ── Actionable takeaways ─────────────────────────────────────
    print("=" * 65)
    print("  WHAT THIS MEANS FOR TRAINING")
    print("=" * 65)
    print("""
  1. Always set  DataLoader(..., pin_memory=True).
     It gives you the H2D pinned speedup for free.

  2. Small transfers (<= 16 MB) are latency-bound, not bandwidth-bound.
     Batch size and packing efficiency matter more than raw PCIe gen.

  3. If peak bandwidth is well below the PCIe theoretical max, check:
       - GPU is in the correct slot (x16, not x4 or x8)
       - IOMMU / ACS is not throttling DMA
       - NUMA topology — CPU and GPU should be on the same socket

  4. Running in a Kubernetes pod / container?
       - Request a GPU: resources.limits.nvidia.com/gpu: 1
       - Use Guaranteed QoS (requests == limits) to avoid noisy neighbours
       - Set memory limit high enough for pinned allocations (they are
         page-locked and count against cgroup limits, cannot be swapped)
       - Watch for CPU CFS throttling — it inflates wall-clock times and
         makes PCIe look slower than it actually is
""")
