import argparse
import os
import random
import statistics
import time

from io_benchmark_utils import (
    DEFAULT_DATA_DIR,
    DEFAULT_NUM_SAMPLES,
    DEFAULT_SAMPLE_SIZE_KB,
    cleanup_data,
    purge_os_cache,
    setup_data,
)

DEFAULT_NUM_RUNS = 3


def benchmark_random_access(data_dir, num_samples, sample_size_bytes):
    """Read individual files in shuffled order (open -> read -> close per sample)."""
    small_dir = os.path.join(data_dir, "small_files")
    indices = list(range(num_samples))
    random.shuffle(indices)

    bytes_read = 0
    start = time.perf_counter()

    for idx in indices:
        with open(os.path.join(small_dir, f"{idx}.bin"), "rb") as f:
            data = f.read()
            bytes_read += len(data)

    elapsed = time.perf_counter() - start
    mb_per_sec = (bytes_read / (1024 * 1024)) / elapsed
    return elapsed, mb_per_sec


def benchmark_sequential_access(data_dir, num_samples, sample_size_bytes):
    """Stream-read one contiguous file in fixed-size chunks."""
    giant_path = os.path.join(data_dir, "giant_shard.bin")
    bytes_read = 0
    start = time.perf_counter()

    with open(giant_path, "rb") as f:
        while True:
            data = f.read(sample_size_bytes)
            if not data:
                break
            bytes_read += len(data)

    elapsed = time.perf_counter() - start
    mb_per_sec = (bytes_read / (1024 * 1024)) / elapsed
    return elapsed, mb_per_sec


def run_benchmarks(data_dir, num_samples, sample_size_bytes, num_runs, skip_purge):
    """Run both benchmarks multiple times, purging cache between each run."""
    random_speeds = []
    sequential_speeds = []

    can_purge = False
    if not skip_purge:
        print("🔄 Checking if OS page cache can be purged...")
        can_purge = purge_os_cache()
        if can_purge:
            print("   ✅ Cache purge available — benchmarks will hit real storage.\n")
        else:
            print("   ⚠️  Cache purge unavailable (needs sudo without password).")
            print("   Results will reflect cached I/O, not real disk speed.")
            print("   To enable: run with `sudo -v` first, or use `--skip-purge`.\n")

    for run in range(1, num_runs + 1):
        print(f"━━━ Run {run}/{num_runs} ━━━")

        # --- Random Access ---
        if can_purge:
            purge_os_cache()
        _, rand_speed = benchmark_random_access(data_dir, num_samples, sample_size_bytes)
        random_speeds.append(rand_speed)
        print(f"   🐢 Random Access:     {rand_speed:>10.2f} MB/s")

        # --- Sequential Access ---
        if can_purge:
            purge_os_cache()
        _, seq_speed = benchmark_sequential_access(data_dir, num_samples, sample_size_bytes)
        sequential_speeds.append(seq_speed)
        print(f"   🐇 Sequential Access: {seq_speed:>10.2f} MB/s")

    return random_speeds, sequential_speeds


def print_report(random_speeds, sequential_speeds, num_samples, sample_size_bytes):
    """Print final summary with mean, std, and speedup."""
    rand_mean = statistics.mean(random_speeds)
    seq_mean = statistics.mean(sequential_speeds)
    rand_std = statistics.stdev(random_speeds) if len(random_speeds) > 1 else 0.0
    seq_std = statistics.stdev(sequential_speeds) if len(sequential_speeds) > 1 else 0.0
    speedup = seq_mean / rand_mean if rand_mean > 0 else float("inf")

    total_mb = (num_samples * sample_size_bytes) / (1024 * 1024)

    print(f"\n{'=' * 50}")
    print("  RANDOM vs SEQUENTIAL I/O — SUMMARY")
    print(f"{'=' * 50}")
    print(f"  Samples:    {num_samples:,}")
    print(f"  Sample Size: {sample_size_bytes // 1024}KB")
    print(f"  Total Data:  {total_mb:.0f}MB")
    print(f"  Runs:        {len(random_speeds)}")
    print(f"{'-' * 50}")
    print(f"  🐢 Random Access:     {rand_mean:>8.2f} ± {rand_std:.2f} MB/s")
    print(f"  🐇 Sequential Access: {seq_mean:>8.2f} ± {seq_std:.2f} MB/s")
    print(f"{'-' * 50}")
    print(f"  ⚡ Speedup: {speedup:.1f}x")
    print(f"{'=' * 50}")

    if speedup < 2:
        print("\n  ℹ️  Low speedup? Data likely served from OS page cache.")
        print("     Try running with sudo to enable cache purging.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark random vs sequential file I/O patterns (pure Python)"
    )
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR,
                        help=f"Data directory (default: {DEFAULT_DATA_DIR})")
    parser.add_argument("--samples", type=int, default=DEFAULT_NUM_SAMPLES,
                        help=f"Number of data samples (default: {DEFAULT_NUM_SAMPLES})")
    parser.add_argument("--sample-size-kb", type=int, default=DEFAULT_SAMPLE_SIZE_KB,
                        help=f"Size of each sample in KB (default: {DEFAULT_SAMPLE_SIZE_KB})")
    parser.add_argument("--runs", type=int, default=DEFAULT_NUM_RUNS,
                        help=f"Number of benchmark iterations (default: {DEFAULT_NUM_RUNS})")
    parser.add_argument("--skip-purge", action="store_true",
                        help="Skip OS page cache purging")
    parser.add_argument("--keep-data", action="store_true",
                        help="Don't delete benchmark data after finishing")
    args = parser.parse_args()

    sample_size_bytes = args.sample_size_kb * 1024

    setup_data(args.data_dir, args.samples, sample_size_bytes)

    rand_speeds, seq_speeds = run_benchmarks(
        args.data_dir, args.samples, sample_size_bytes, args.runs, args.skip_purge
    )

    print_report(rand_speeds, seq_speeds, args.samples, sample_size_bytes)

    if not args.keep_data:
        cleanup_data(args.data_dir)
