import argparse
import os
import statistics
import time

import torch
import torch.utils.data
from io_benchmark_utils import (
    DEFAULT_DATA_DIR,
    DEFAULT_NUM_SAMPLES,
    DEFAULT_SAMPLE_SIZE_KB,
    cleanup_data,
    count_samples,
    data_exists,
    detect_sample_size,
    setup_data,
)
from torch.utils.data import DataLoader, Dataset

# Defaults
DEFAULT_BATCH_SIZE = 64
DEFAULT_NUM_WORKERS = 4
DEFAULT_NUM_RUNS = 3
DEFAULT_WARMUP_BATCHES = 5


class SmallFilesDataset(Dataset):
    """Each sample opens, reads, and closes an individual file (random access)."""

    def __init__(self, data_dir):
        small_dir = os.path.join(data_dir, "small_files")
        self.files = sorted(
            os.path.join(small_dir, f)
            for f in os.listdir(small_dir)
            if f.endswith(".bin")
        )
        self.sample_size = os.path.getsize(self.files[0])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # REAL I/O: Open -> Read -> Close (per sample)
        with open(self.files[idx], "rb") as f:
            data = f.read()
        return torch.tensor(len(data), dtype=torch.long)


class GiantFileDataset(Dataset):
    """
    Reads samples from one contiguous file via seek+read.
    Each worker keeps its own persistent file handle (opened in worker_init_fn)
    to avoid per-sample open/close overhead.
    """

    def __init__(self, file_path, sample_size, num_samples):
        self.file_path = file_path
        self.sample_size = sample_size
        self.num_samples = num_samples
        self._fh = None

    def _open(self):
        if self._fh is None:
            self._fh = open(self.file_path, "rb")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        self._open()
        self._fh.seek(idx * self.sample_size)
        data = self._fh.read(self.sample_size)
        return torch.tensor(len(data), dtype=torch.long)


def giant_file_worker_init(worker_id):
    """
    Called once per DataLoader worker process.
    Opens a fresh file handle so each worker has its own (no conflicts).
    """
    dataset = torch.utils.data.get_worker_info().dataset
    dataset._fh = open(dataset.file_path, "rb")


def benchmark_loader(loader, label, num_runs, warmup_batches):
    """Run the loader multiple times, report mean +/- std throughput."""
    speeds = []

    for run in range(1, num_runs + 1):
        total_bytes = 0
        iterator = iter(loader)

        # Warmup — let workers spin up, caches settle
        for _ in range(warmup_batches):
            try:
                next(iterator)
            except StopIteration:
                break

        # Timed region
        start = time.perf_counter()
        for batch in iterator:
            total_bytes += batch.sum().item()
        elapsed = time.perf_counter() - start

        if elapsed > 0 and total_bytes > 0:
            mb_per_sec = (total_bytes / (1024 * 1024)) / elapsed
            speeds.append(mb_per_sec)

    if not speeds:
        print(f"   {label}")
        print("     ⚠️  No valid measurements")
        return 0.0

    mean_speed = statistics.mean(speeds)
    std_speed = statistics.stdev(speeds) if len(speeds) > 1 else 0.0
    print(f"   {label}")
    print(f"     Speed: {mean_speed:>8.2f} ± {std_speed:.2f} MB/s "
          f"({num_runs} runs, {warmup_batches} warmup batches)")
    return mean_speed


def run_scenario(data_dir, giant_file_path, sample_size_bytes, num_samples,
                 num_workers, batch_size, num_runs, warmup_batches):
    """Run random vs sequential benchmark for a given worker count."""
    worker_label = f"num_workers={num_workers}"
    if num_workers == 0:
        worker_label += " (main process only — isolates I/O pattern)"
    else:
        worker_label += " (parallel prefetch — masks I/O pattern)"

    print(f"{'─' * 55}")
    print(f"  Scenario: {worker_label}")
    print(f"{'─' * 55}")

    # --- Small Files (Random Access) ---
    print("\n  🐢 Random Access — many small files, shuffled:")
    ds_small = SmallFilesDataset(data_dir)
    loader_small = DataLoader(
        ds_small,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )
    rand_speed = benchmark_loader(
        loader_small, "Small Files (open/read/close per sample)",
        num_runs, warmup_batches
    )

    # --- Giant File (Sequential Access) ---
    print("\n  🐇 Sequential Access — one contiguous shard, seek+read:")
    ds_giant = GiantFileDataset(giant_file_path, sample_size_bytes, num_samples)
    worker_init = giant_file_worker_init if num_workers > 0 else None
    loader_giant = DataLoader(
        ds_giant,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        worker_init_fn=worker_init,
    )
    seq_speed = benchmark_loader(
        loader_giant, "Giant Shard (persistent handle, seek+read)",
        num_runs, warmup_batches
    )

    speedup = seq_speed / rand_speed if rand_speed > 0 else float("inf")
    print(f"\n  ⚡ Speedup: {speedup:.1f}x  (shard vs small files, {worker_label})")
    return rand_speed, seq_speed, speedup


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark random vs sequential I/O through PyTorch DataLoaders"
    )
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR,
                        help=f"Data directory (default: {DEFAULT_DATA_DIR})")
    parser.add_argument("--samples", type=int, default=DEFAULT_NUM_SAMPLES,
                        help=f"Number of data samples (default: {DEFAULT_NUM_SAMPLES})")
    parser.add_argument("--sample-size-kb", type=int, default=DEFAULT_SAMPLE_SIZE_KB,
                        help=f"Size of each sample in KB (default: {DEFAULT_SAMPLE_SIZE_KB})")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Batch size (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS,
                        help=f"DataLoader workers (default: {DEFAULT_NUM_WORKERS})")
    parser.add_argument("--runs", type=int, default=DEFAULT_NUM_RUNS,
                        help=f"Benchmark iterations (default: {DEFAULT_NUM_RUNS})")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP_BATCHES,
                        help=f"Warmup batches per run (default: {DEFAULT_WARMUP_BATCHES})")
    parser.add_argument("--keep-data", action="store_true",
                        help="Don't delete benchmark data after finishing")
    args = parser.parse_args()

    sample_size_bytes = args.sample_size_kb * 1024

    # Generate data if it doesn't already exist
    if data_exists(args.data_dir):
        print(f"📁 Using existing data in {args.data_dir}")
        sample_size_bytes = detect_sample_size(args.data_dir)
        num_samples = count_samples(args.data_dir)
        print(f"   {num_samples:,} samples, {sample_size_bytes // 1024}KB each\n")
    else:
        num_samples = args.samples
        setup_data(args.data_dir, num_samples, sample_size_bytes)

    giant_file_path = os.path.join(args.data_dir, "giant_shard.bin")

    print("📊 DataLoader I/O Benchmark")
    print(f"   Samples: {num_samples:,}  |  Sample size: {sample_size_bytes // 1024}KB  "
          f"|  Batch: {args.batch_size}")
    print()

    # ── Scenario 1: num_workers=0 ──
    # Main process does all I/O synchronously.
    # This isolates the I/O access pattern difference.
    _, _, speedup_sync = run_scenario(
        args.data_dir, giant_file_path, sample_size_bytes, num_samples,
        num_workers=0, batch_size=args.batch_size,
        num_runs=args.runs, warmup_batches=args.warmup
    )
    print()

    # ── Scenario 2: num_workers=N ──
    # Workers prefetch in background. I/O overlaps with main loop.
    # This shows how parallelism can mask the I/O pattern difference.
    _, _, speedup_async = run_scenario(
        args.data_dir, giant_file_path, sample_size_bytes, num_samples,
        num_workers=args.num_workers, batch_size=args.batch_size,
        num_runs=args.runs, warmup_batches=args.warmup
    )
    print()

    # ── Final Summary ──
    print(f"{'═' * 55}")
    print("  SUMMARY")
    print(f"{'═' * 55}")
    print(f"  num_workers=0:  {speedup_sync:.1f}x speedup  "
          "(raw I/O pattern difference)")
    print(f"  num_workers={args.num_workers}:  {speedup_async:.1f}x speedup  "
          "(with parallel prefetch)")
    print(f"{'─' * 55}")
    if speedup_sync > 2 and speedup_async < 2:
        print("  Lesson: Workers mask the I/O difference by prefetching.")
        print("  But on cold storage (HDD/network), the pattern still matters!")
    elif speedup_sync < 2:
        print("  Both scenarios show similar speed — data is likely cached in RAM.")
        print("  The difference appears on cold storage (HDD, NFS, S3).")
    print(f"{'═' * 55}")

    if not args.keep_data:
        cleanup_data(args.data_dir)


if __name__ == "__main__":
    main()
