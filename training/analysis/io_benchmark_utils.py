"""
Shared utilities for I/O benchmark scripts.
Handles data generation and OS cache purging.
"""

import os
import platform
import shutil
import subprocess

DEFAULT_DATA_DIR = "./output/io_benchmark_data"
DEFAULT_NUM_SAMPLES = 10000
DEFAULT_SAMPLE_SIZE_KB = 16  # 16KB per sample → ~160MB total


def setup_data(data_dir, num_samples, sample_size_bytes):
    """Generate test data: many small files + one large contiguous file."""
    total_mb = (num_samples * sample_size_bytes) / (1024 * 1024)

    if os.path.exists(data_dir):
        print(f"🧹 Cleaning up {data_dir}...")
        shutil.rmtree(data_dir)

    small_files_dir = os.path.join(data_dir, "small_files")
    os.makedirs(small_files_dir, exist_ok=True)

    print(f"🎲 Generating {num_samples} files "
          f"({sample_size_bytes // 1024}KB each, ~{total_mb:.0f}MB total)...")

    for i in range(num_samples):
        with open(os.path.join(small_files_dir, f"{i}.bin"), "wb") as f:
            f.write(os.urandom(sample_size_bytes))

    print(f"📦 Creating 1 contiguous shard (~{total_mb:.0f}MB)...")
    giant_path = os.path.join(data_dir, "giant_shard.bin")
    with open(giant_path, "wb") as f_out:
        for i in range(num_samples):
            with open(os.path.join(small_files_dir, f"{i}.bin"), "rb") as f_in:
                f_out.write(f_in.read())

    os.sync()
    print("✅ Data generation complete.\n")


def cleanup_data(data_dir):
    """Remove benchmark data directory."""
    if os.path.exists(data_dir):
        print(f"\n🧹 Cleaning up {data_dir}...")
        shutil.rmtree(data_dir)
        print("Done.")


def data_exists(data_dir):
    """Check if benchmark data has already been generated."""
    small_dir = os.path.join(data_dir, "small_files")
    giant_path = os.path.join(data_dir, "giant_shard.bin")
    return os.path.isdir(small_dir) and os.path.exists(giant_path)


def detect_sample_size(data_dir):
    """Auto-detect sample size from the first small file."""
    small_dir = os.path.join(data_dir, "small_files")
    for name in sorted(os.listdir(small_dir)):
        if name.endswith(".bin"):
            return os.path.getsize(os.path.join(small_dir, name))
    raise FileNotFoundError(f"No .bin files found in {small_dir}")


def count_samples(data_dir):
    """Count the number of .bin sample files."""
    small_dir = os.path.join(data_dir, "small_files")
    return len([f for f in os.listdir(small_dir) if f.endswith(".bin")])


def purge_os_cache():
    """
    Attempt to flush the OS page cache so benchmarks hit real storage.
    - macOS: `sudo purge`  (tries both non-interactive and interactive)
    - Linux: `sync && echo 3 > /proc/sys/vm/drop_caches`
    Returns True if successful, False otherwise.
    """
    system = platform.system()
    try:
        if system == "Darwin":
            result = subprocess.run(
                ["sudo", "-n", "/usr/sbin/purge"],
                capture_output=True, timeout=10
            )
            if result.returncode == 0:
                return True
            result = subprocess.run(
                ["sudo", "/usr/sbin/purge"],
                timeout=30
            )
            return result.returncode == 0
        elif system == "Linux":
            subprocess.run(["sync"], check=True, capture_output=True)
            result = subprocess.run(
                ["sudo", "-n", "sh", "-c",
                 "echo 3 > /proc/sys/vm/drop_caches"],
                capture_output=True, timeout=10
            )
            if result.returncode == 0:
                return True
            result = subprocess.run(
                ["sudo", "sh", "-c",
                 "echo 3 > /proc/sys/vm/drop_caches"],
                timeout=30
            )
            return result.returncode == 0
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
            FileNotFoundError):
        pass
    return False
