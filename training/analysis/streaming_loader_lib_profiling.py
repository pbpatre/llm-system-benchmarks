"""
Streaming DataLoader Profiling
==============================
Compares standard ImageFolder vs MosaicML Streaming under
torch.profiler to surface data-loading bottlenecks in a
realistic training loop (ResNet-50 on synthetic JPEG images).

Key design: we purge the OS page cache before EACH scenario so that
every run starts from cold storage.  This makes the I/O access pattern
the dominant factor — ImageFolder's random small-file opens vs
Streaming's sequential shard reads.

Scenarios:
  1. ImageFolder,  num_workers=0  — cold, synchronous (raw I/O cost visible)
  2. Streaming,    num_workers=0  — cold, synchronous (sequential reads)
  3. ImageFolder,  num_workers=4  — cold, parallel prefetch
  4. Streaming,    num_workers=4  — cold, parallel prefetch + sequential reads

Outputs:
  - TensorBoard trace files (view with: tensorboard --logdir=./output/log/profiler)
  - Per-step timing summary printed to stdout for quick comparison
"""

import io
import os
import shutil
import statistics
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from streaming import MDSWriter, StreamingDataset
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from io_benchmark_utils import purge_os_cache

# ── Configuration ────────────────────────────────────────────────

DATA_ROOT = "./output/profile_data"
RAW_DIR = f"{DATA_ROOT}/raw_images"
MDS_DIR = f"{DATA_ROOT}/mds_shards"
LOG_DIR = "./output/log/profiler"

BATCH_SIZE = 64
IMG_SIZE = 224
NUM_SAMPLES = 100_000       # ~1562 batches — stresses inode/dentry cache
WARMUP_STEPS = 3            # full forward/backward warmup (cuDNN, allocator)
PROFILE_WARMUP_STEPS = 3    # profiler-internal warmup (tracing overhead)
PROFILE_ACTIVE_STEPS = 20   # more steps for tighter confidence intervals

# Light augmentation — just decode + resize + normalise.  Keeps CPU work
# minimal so the I/O access pattern difference isn't hidden by augmentation.
LIGHT_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ── Data generation ──────────────────────────────────────────────

def generate_data():
    """Create synthetic JPEG images to force real decode + I/O workload."""
    if os.path.exists(RAW_DIR):
        n_existing = len([f for f in os.listdir(f"{RAW_DIR}/class_0") if f.endswith(".jpg")])
        if n_existing >= NUM_SAMPLES:
            print(f"Using existing {n_existing} images in {RAW_DIR}\n")
            return
        print(f"Found {n_existing} images but need {NUM_SAMPLES}, regenerating...")
        shutil.rmtree(RAW_DIR)

    print(f"Generating {NUM_SAMPLES:,} JPEG images ({IMG_SIZE}x{IMG_SIZE})...")
    os.makedirs(f"{RAW_DIR}/class_0", exist_ok=True)

    t0 = time.perf_counter()
    for i in range(NUM_SAMPLES):
        img = np.random.randint(0, 255, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        img = Image.fromarray(img)
        img.save(f"{RAW_DIR}/class_0/img_{i}.jpg", quality=80)
        if (i + 1) % 5000 == 0:
            elapsed = time.perf_counter() - t0
            rate = (i + 1) / elapsed
            remaining = (NUM_SAMPLES - i - 1) / rate
            print(f"  {i + 1:>6,}/{NUM_SAMPLES:,}  ({rate:.0f} img/s, ~{remaining:.0f}s left)")
    print(f"Done in {time.perf_counter() - t0:.0f}s.\n")


def convert_to_mds():
    """Convert raw JPEGs into MosaicML Streaming shards."""
    if os.path.exists(MDS_DIR):
        print(f"Using existing MDS shards in {MDS_DIR}\n")
        return

    print("Converting to MosaicML Streaming shards...")
    t0 = time.perf_counter()
    dataset = ImageFolder(RAW_DIR)

    columns = {"data": "jpeg", "label": "int"}
    with MDSWriter(out=MDS_DIR, columns=columns, compression=None) as out:
        for i, (_img, label) in enumerate(dataset):
            with open(dataset.samples[i][0], "rb") as f:
                img_bytes = f.read()
            out.write({"data": img_bytes, "label": label})
            if (i + 1) % 5000 == 0:
                print(f"  {i + 1:>6,}/{len(dataset):,}")
    print(f"Done in {time.perf_counter() - t0:.0f}s.\n")


# ── MDS Dataset wrapper ─────────────────────────────────────────

class StreamingImageDataset(StreamingDataset):
    """MosaicML Streaming dataset that decodes JPEGs on the fly."""

    def __init__(self, local, transform, batch_size):
        super().__init__(local=local, batch_size=batch_size)
        self.transform = transform

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        img = sample["data"]
        # MDS with column type "jpeg" auto-decodes to a PIL Image.
        # If raw bytes are returned instead, fall back to manual decode.
        if not isinstance(img, Image.Image):
            img = Image.open(io.BytesIO(img))
        return self.transform(img), sample["label"]


# ── Profiling loop ───────────────────────────────────────────────

def train_and_profile(loader, model, label, purge_cache=False):
    """Run a training loop under torch.profiler with clear phase markers.

    Markers visible in TensorBoard trace:
      DATALOADER_WAIT  — time spent waiting for the next batch
      H2D_TRANSFER     — CPU -> GPU transfer (should be fast with pin_memory)
      FORWARD          — model forward pass
      BACKWARD         — loss.backward()
      OPTIMIZER        — optimizer.step()
    """
    print(f"\n  Profiling: {label}")
    device = torch.device("cuda")

    # Fresh model state each scenario so comparisons are fair
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # ── Full warmup (settles cuDNN autotuner, JIT, allocator) ────
    print(f"    Warming up ({WARMUP_STEPS} full train steps)...")
    model.train()
    iter_loader = iter(loader)
    for _ in range(WARMUP_STEPS):
        inputs, labels_batch = next(iter_loader)
        inputs = inputs.to(device, non_blocking=True)
        labels_batch = labels_batch.to(device, non_blocking=True)

        optimizer.zero_grad()
        loss = criterion(model(inputs), labels_batch)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()

    # ── Purge cache AFTER warmup, right before profiled steps ────
    # This ensures the profiled region starts from cold storage while
    # CUDA/cuDNN are already warmed up.
    if purge_cache:
        if purge_os_cache():
            print("    OS page cache purged — profiling from cold storage.")
        else:
            print("    WARNING: Could not purge OS cache.")

    # ── Profiled run ─────────────────────────────────────────────
    total_steps = PROFILE_WARMUP_STEPS + PROFILE_ACTIVE_STEPS
    print(f"    Capturing trace ({PROFILE_WARMUP_STEPS} warmup + "
          f"{PROFILE_ACTIVE_STEPS} active steps)...")

    # Per-step timing for stdout summary
    data_times = []
    transfer_times = []
    compute_times = []

    trace_dir = f"{LOG_DIR}/{label}"
    if os.path.exists(trace_dir):
        shutil.rmtree(trace_dir)
    os.makedirs(trace_dir, exist_ok=True)

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=0,
            warmup=PROFILE_WARMUP_STEPS,
            active=PROFILE_ACTIVE_STEPS,
            repeat=1,
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_dir),
        record_shapes=True,
        with_stack=True,
    ) as prof:

        for step in range(total_steps):
            try:
                # Phase 1: Data loading (CPU workers -> main process)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.profiler.record_function("DATALOADER_WAIT"):
                    inputs, labels_batch = next(iter_loader)
                t1 = time.perf_counter()

                # Phase 2: H2D transfer (pinned memory -> GPU via DMA)
                with torch.profiler.record_function("H2D_TRANSFER"):
                    inputs = inputs.to(device, non_blocking=True)
                    labels_batch = labels_batch.to(device, non_blocking=True)
                    torch.cuda.synchronize()
                t2 = time.perf_counter()

                # Phase 3: Compute (forward + backward + optimizer)
                with torch.profiler.record_function("FORWARD"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels_batch)

                with torch.profiler.record_function("BACKWARD"):
                    optimizer.zero_grad()
                    loss.backward()

                with torch.profiler.record_function("OPTIMIZER"):
                    optimizer.step()

                torch.cuda.synchronize()
                t3 = time.perf_counter()

                # Only record active steps (skip profiler warmup)
                if step >= PROFILE_WARMUP_STEPS:
                    data_times.append(t1 - t0)
                    transfer_times.append(t2 - t1)
                    compute_times.append(t3 - t2)

                prof.step()

            except StopIteration:
                print(f"    WARNING: DataLoader exhausted at step {step}/{total_steps}")
                break

    # ── Stdout summary ───────────────────────────────────────────
    if data_times:
        def _fmt(vals):
            mu = statistics.mean(vals) * 1000
            sd = statistics.stdev(vals) * 1000 if len(vals) > 1 else 0
            return f"{mu:7.2f} +/- {sd:5.2f} ms"

        total = [d + t + c for d, t, c in zip(data_times, transfer_times, compute_times)]
        data_pct = statistics.mean(data_times) / statistics.mean(total) * 100

        print(f"\n    {label} — per-step timing (active steps only):")
        print(f"      DataLoader wait: {_fmt(data_times)}  ({data_pct:.0f}%)")
        print(f"      H2D transfer:    {_fmt(transfer_times)}")
        print(f"      Compute:         {_fmt(compute_times)}")
        print(f"      Total step:      {_fmt(total)}")
        throughput = BATCH_SIZE / statistics.mean(total)
        print(f"      Throughput:      {throughput:7.1f} img/s")

    return data_times, compute_times


def cleanup():
    """Remove generated data."""
    for d in [DATA_ROOT, LOG_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d)
    print("Cleaned up generated data.")


def _run_scenario(ds_name, ds_factory, num_workers, transform, purge_cache):
    """Run a single profiling scenario and return timing data."""
    label = f"{ds_name}_w{num_workers}"
    loader = DataLoader(
        ds_factory(transform),
        batch_size=BATCH_SIZE,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=(num_workers == 0 and ds_name == "imagefolder"),
        persistent_workers=(num_workers > 0),
    )
    model = models.resnet50()
    data_times, compute_times = train_and_profile(
        loader, model, label, purge_cache=purge_cache,
    )
    del model, loader
    torch.cuda.empty_cache()
    return label, data_times, compute_times


# ── Main ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("No CUDA GPU found — cannot profile training loop.")
        import sys
        sys.exit(1)

    generate_data()
    convert_to_mds()

    # Check if cache purging works
    can_purge = purge_os_cache()
    if can_purge:
        print("OS page cache purge: AVAILABLE")
        print("Each scenario will start from cold storage.\n")
    else:
        print("OS page cache purge: UNAVAILABLE (need root)")
        print("Results will reflect cached I/O — expect similar")
        print("performance between ImageFolder and Streaming.\n")

    # Dataset factories (deferred so we can pass different transforms)
    def make_imagefolder(t):
        return ImageFolder(RAW_DIR, transform=t)

    def make_streaming(t):
        return StreamingImageDataset(local=MDS_DIR, transform=t, batch_size=BATCH_SIZE)

    results = []

    # ── Part 1: num_workers=0 (synchronous, cold storage) ────────
    W = 65
    print("=" * W)
    print("  PART 1: Synchronous loading (num_workers=0)")
    print("  Main thread does all I/O + decode.  Cache purged before")
    print("  each run so we measure REAL disk reads, not page cache.")
    print("=" * W)

    results.append(_run_scenario(
        "imagefolder", make_imagefolder, 0, LIGHT_TRANSFORM, can_purge))
    results.append(_run_scenario(
        "streaming", make_streaming, 0, LIGHT_TRANSFORM, can_purge))

    # ── Part 2: num_workers=4 (parallel prefetch, cold storage) ──
    print()
    print("=" * W)
    print("  PART 2: Parallel prefetch (num_workers=4)")
    print("  Workers prep batches in background, still cold storage.")
    print("  Question: can 4 workers keep up when hitting real disk?")
    print("=" * W)

    results.append(_run_scenario(
        "imagefolder", make_imagefolder, 4, LIGHT_TRANSFORM, can_purge))
    results.append(_run_scenario(
        "streaming", make_streaming, 4, LIGHT_TRANSFORM, can_purge))

    # ── Comparison table ─────────────────────────────────────────
    print()
    print("=" * W)
    print("  SUMMARY — DataLoader wait as % of step time")
    print("=" * W)
    print(f"  {'Scenario':<25} {'DataLoader':>12} {'Compute':>12} {'Data %':>8}")
    print(f"  {'-' * 25} {'-' * 12} {'-' * 12} {'-' * 8}")

    for label, dtimes, ctimes in results:
        if dtimes and ctimes:
            d_ms = statistics.mean(dtimes) * 1000
            c_ms = statistics.mean(ctimes) * 1000
            pct = d_ms / (d_ms + c_ms) * 100
            print(f"  {label:<25} {d_ms:>9.1f} ms {c_ms:>9.1f} ms {pct:>7.1f}%")

    # ── Speedup analysis ─────────────────────────────────────────
    print()
    pairs = [
        ("num_workers=0", results[0], results[1]),
        ("num_workers=4", results[2], results[3]),
    ]
    for desc, (lbl_if, dt_if, _), (lbl_st, dt_st, _) in pairs:
        if dt_if and dt_st:
            if_ms = statistics.mean(dt_if) * 1000
            st_ms = statistics.mean(dt_st) * 1000
            speedup = if_ms / st_ms if st_ms > 0 else 0
            winner = "Streaming" if speedup > 1 else "ImageFolder"
            print(f"  {desc}: DataLoader speedup = {speedup:.2f}x ({winner} faster)")

    print()
    print("  WHAT TO LOOK FOR:")
    print("    With cold storage, Streaming should show lower DataLoader wait")
    print("    because it reads from a few large sequential shards rather than")
    print("    opening/seeking/closing 30K individual JPEG files at random.")
    print()
    print(f"  Trace files saved. View with:")
    print(f"    tensorboard --logdir={LOG_DIR}")
