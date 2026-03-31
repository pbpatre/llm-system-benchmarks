import os
import time
import math
import functools
from contextlib import nullcontext
from datetime import timedelta
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, FullStateDictConfig, MixedPrecision
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from transformers import GPT2Config, GPT2LMHeadModel, AutoTokenizer
from torch.utils.tensorboard import SummaryWriter
from dataclasses import dataclass
from typing import Tuple


# --- Configuration ---
@dataclass
class TrainingConfig:
    # Model Architecture
    batch_size: int = 8           # Micro-batch per GPU per forward pass
    seq_len: int = 1024
    d_model: int = 768            # GPT-2 Small size
    n_layers: int = 12
    n_heads: int = 12
    vocab_size: int = 50257

    # Data
    dataset_name: str = "wikitext"               # HuggingFace dataset name
    dataset_config: str = "wikitext-103-raw-v1"  # Dataset subset/config
    tokenizer_name: str = "gpt2"                 # Tokenizer (must match vocab_size)
    use_synthetic: bool = False                   # True = random tokens (GPU-only benchmarking)

    # Optimization
    lr: float = 3e-4
    min_lr: float = 3e-5          # Cosine decay floor (10% of peak)
    weight_decay: float = 0.1     # Standard for transformer pretraining
    max_steps: int = 100
    warmup_steps: int = 10
    grad_accum_steps: int = 4     # Effective batch per GPU = 8 * 4 = 32
    max_grad_norm: float = 1.0    # Gradient clipping threshold

    # Hardware
    dtype: torch.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    use_compile: bool = False     # torch.compile for kernel fusion speedup
    num_workers: int = 2          # DataLoader workers per rank

    # Logging & Checkpointing
    log_every: int = 10
    checkpoint_dir: str = "checkpoints/distributed"
    log_dir: str = "runs/distributed"
    seed: int = 42


def set_seed(seed: int, rank: int = 0):
    """Reproducibility: seed all RNGs. Offset by rank so each GPU sees different data order."""
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --- 1. Learning Rate Schedule (Warmup + Cosine Decay) ---
def get_lr_scheduler(optimizer: torch.optim.Optimizer, cfg: TrainingConfig) -> LambdaLR:
    """Linear warmup for `warmup_steps`, then cosine decay to `min_lr`."""
    def lr_lambda(step: int) -> float:
        if step < cfg.warmup_steps:
            return step / max(1, cfg.warmup_steps)
        decay_steps = cfg.max_steps - cfg.warmup_steps
        progress = (step - cfg.warmup_steps) / max(1, decay_steps)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        min_ratio = cfg.min_lr / cfg.lr
        return min_ratio + (1.0 - min_ratio) * cosine_decay
    return LambdaLR(optimizer, lr_lambda)


# --- 2. Datasets ---

class SyntheticTextDataset(Dataset):
    """Random tokens for GPU-only benchmarking (no disk/CPU overhead)."""
    def __init__(self, vocab_size: int, seq_len: int, length: int = 10000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        data = torch.randint(0, self.vocab_size, (self.seq_len + 1,), dtype=torch.long)
        return data[:-1], data[1:]


class HuggingFaceTextDataset(Dataset):
    """
    Loads a HuggingFace dataset, tokenizes all text, concatenates into one
    contiguous token stream, and serves fixed-length chunks.
    Documents are separated by EOS tokens so the model learns document boundaries.
    """
    def __init__(self, dataset_name: str, dataset_config: str, tokenizer_name: str,
                 seq_len: int, split: str = "train"):
        from datasets import load_dataset

        print(f"Loading dataset: {dataset_name}/{dataset_config} [{split}]...")
        raw_dataset = load_dataset(dataset_name, dataset_config, split=split)

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        eos_token = tokenizer.eos_token_id

        print(f"Tokenizing with {tokenizer_name}...")
        def tokenize_fn(examples):
            out = tokenizer(examples["text"], add_special_tokens=False)
            for ids in out["input_ids"]:
                ids.append(eos_token)
            return out

        tokenized = raw_dataset.map(
            tokenize_fn, batched=True,
            remove_columns=raw_dataset.column_names, num_proc=4, desc="Tokenizing",
        )

        print("Concatenating tokens...")
        all_ids = []
        for example in tokenized:
            all_ids.extend(example["input_ids"])

        self.tokens = torch.tensor(all_ids, dtype=torch.long)
        self.seq_len = seq_len
        self.n_sequences = (len(self.tokens) - 1) // seq_len
        self.tokens = self.tokens[: self.n_sequences * seq_len + 1]

        print(f"Dataset ready: {len(self.tokens):,} tokens -> {self.n_sequences:,} sequences of {seq_len}")

    def __len__(self) -> int:
        return self.n_sequences

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.seq_len
        chunk = self.tokens[start : start + self.seq_len + 1]
        return chunk[:-1], chunk[1:]


# --- 3. Model ---
def get_model(cfg: TrainingConfig) -> nn.Module:
    """Initializes a GPT-2 model from scratch (no pretrained weights)."""
    model_config = GPT2Config(
        vocab_size=cfg.vocab_size,
        n_positions=cfg.seq_len,
        n_embd=cfg.d_model,
        n_layer=cfg.n_layers,
        n_head=cfg.n_heads,
        use_cache=False,
    )
    return GPT2LMHeadModel(model_config)


# --- 4. Distributed Setup ---
def setup_distributed():
    """
    Initialize the distributed process group.
    torchrun sets RANK, WORLD_SIZE, LOCAL_RANK automatically.
    In a real cluster, each LOCAL_RANK maps to a distinct GPU via NCCL.
    On a single-GPU simulation, we fall back to GLOO because NCCL >=2.27
    rejects multiple ranks on the same physical GPU.
    """
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    timeout = timedelta(minutes=5)

    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        gpu_id = local_rank % num_gpus
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(device)

        if world_size > num_gpus:
            # Simulation mode: more ranks than GPUs.
            # NCCL rejects duplicate GPUs, so use GLOO (CPU-based collectives).
            # Data stays on the GPU for compute; only gradient sync goes through CPU.
            dist.init_process_group("gloo", timeout=timeout)
            if rank == 0:
                print(f"WARNING: {world_size} ranks but only {num_gpus} GPU(s). "
                      f"Using GLOO backend (simulation mode — not for benchmarking).")
        else:
            # Real multi-GPU: each rank gets its own GPU, use NCCL for fast GPU-direct comms.
            dist.init_process_group("nccl", timeout=timeout)
    else:
        device = torch.device("cpu")
        dist.init_process_group("gloo", timeout=timeout)

    return rank, world_size, local_rank, device


def cleanup():
    dist.destroy_process_group()


def get_dist_dataloader(dataset, batch_size, rank, world_size, num_workers: int = 2):
    """
    Shards data across ranks so each GPU trains on different samples.
    Rank 0 sees indices [0, 2, 4, ...], Rank 1 sees [1, 3, 5, ...], etc.
    """
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank,
        shuffle=True, drop_last=True,
    )
    loader = DataLoader(
        dataset, batch_size=batch_size,
        pin_memory=True, num_workers=num_workers,
        persistent_workers=num_workers > 0,
        sampler=sampler,
    )
    return loader, sampler


def save_distributed_checkpoint(model, optimizer, scheduler, step, config, rank, use_fsdp):
    """
    Save checkpoint. FSDP shards state across ranks — must gather to rank 0 first.
    DDP has the full model on every rank, so only rank 0 saves.
    """
    if use_fsdp:
        full_state_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_state_cfg):
            model_state = model.state_dict()
            optim_state = FSDP.optim_state_dict(model, optimizer)
            if rank == 0:
                os.makedirs(config.checkpoint_dir, exist_ok=True)
                path = os.path.join(config.checkpoint_dir, f"step_{step}_fsdp.pt")
                torch.save({
                    "step": step,
                    "model_state_dict": model_state,
                    "optimizer_state_dict": optim_state,
                    "scheduler_state_dict": scheduler.state_dict(),
                    "config": config,
                }, path)
                print(f"  [Rank 0] FSDP checkpoint saved: {path}")
    else:
        if rank == 0:
            os.makedirs(config.checkpoint_dir, exist_ok=True)
            path = os.path.join(config.checkpoint_dir, f"step_{step}_ddp.pt")
            torch.save({
                "step": step,
                "model_state_dict": model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "config": config,
            }, path)
            print(f"  [Rank 0] DDP checkpoint saved: {path}")


# --- 5. The Experiment Runner ---
def run_experiment(use_fsdp: bool):
    rank, world_size, local_rank, device = setup_distributed()
    config = TrainingConfig()
    mode_name = "FSDP" if use_fsdp else "DDP"

    # Seed offset by rank so each GPU sees different random data
    set_seed(config.seed, rank)

    # Only rank 0 prints status and writes TensorBoard
    is_main = (rank == 0)

    if is_main:
        print(f"Mode: {mode_name} | World Size: {world_size}")
        print(f"Device: {device} | Precision: {config.dtype}")
        print(f"Effective batch: {config.batch_size} * {config.grad_accum_steps} * {world_size} "
              f"= {config.batch_size * config.grad_accum_steps * world_size} per step")

    # --- 1. Data ---
    if config.use_synthetic:
        if is_main:
            print("Using synthetic (random token) dataset")
        dataset = SyntheticTextDataset(config.vocab_size, config.seq_len)
    else:
        # Only rank 0 downloads/tokenizes, others wait
        if is_main:
            dataset = HuggingFaceTextDataset(
                dataset_name=config.dataset_name,
                dataset_config=config.dataset_config,
                tokenizer_name=config.tokenizer_name,
                seq_len=config.seq_len,
            )
        dist.barrier()  # Wait for rank 0 to finish downloading
        if not is_main:
            dataset = HuggingFaceTextDataset(
                dataset_name=config.dataset_name,
                dataset_config=config.dataset_config,
                tokenizer_name=config.tokenizer_name,
                seq_len=config.seq_len,
            )

    dataloader, sampler = get_dist_dataloader(
        dataset, config.batch_size, rank, world_size, num_workers=config.num_workers,
    )

    # --- 2. Model ---
    model = get_model(config).to(device)
    gpu_id = device.index

    if config.use_compile:
        model = torch.compile(model)

    if use_fsdp:
        # FSDP: shards parameters, gradients, and optimizer states across ranks.
        # Each GPU holds only 1/N of the model — dramatically reduces per-GPU memory.
        # auto_wrap_policy decides where to shard (each transformer block becomes a unit).
        auto_wrap_policy = functools.partial(
            size_based_auto_wrap_policy, min_num_params=20_000,
        )
        mp_policy = None
        if config.dtype == torch.bfloat16:
            mp_policy = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            )
        elif config.dtype == torch.float16:
            mp_policy = MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16,
            )
        model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mp_policy,
            device_id=device,
        )
        if is_main:
            print(f"[Rank {rank}] Model wrapped in FSDP (sharded across {world_size} GPUs)")
            if mp_policy:
                print(f"  FSDP MixedPrecision: param={mp_policy.param_dtype}, reduce={mp_policy.reduce_dtype}")
    else:
        # DDP: replicates full model on every GPU.
        # After backward(), gradients are all-reduced (averaged) across ranks.
        # Memory: each GPU holds full model + full optimizer states.
        model = DDP(model, device_ids=[gpu_id])
        if is_main:
            print(f"[Rank {rank}] Model wrapped in DDP (replicated on {world_size} GPUs)")

    n_params = sum(p.numel() for p in model.parameters())
    if is_main:
        print(f"Model Parameters: {n_params / 1e6:.1f}M")

    # --- 3. Optimizer + Scheduler ---
    optimizer = AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
        fused=torch.cuda.is_available(),
    )
    scheduler = get_lr_scheduler(optimizer, config)

    # --- 4. Training Loop ---
    writer = SummaryWriter(log_dir=f"{config.log_dir}_{mode_name.lower()}") if is_main else None
    model.train()

    step = 0
    total_tokens = 0
    accum_loss = 0.0
    # Each step: batch_size * seq_len * grad_accum_steps tokens PER GPU
    # Across all GPUs: multiply by world_size for global throughput
    tokens_per_step_local = config.batch_size * config.seq_len * config.grad_accum_steps
    tokens_per_step_global = tokens_per_step_local * world_size
    epoch = 0

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()

    if is_main:
        print("Starting Training Loop...")

    while step < config.max_steps:
        # DistributedSampler must know the epoch for proper shuffling
        sampler.set_epoch(epoch)

        for batch_idx, (x, y) in enumerate(dataloader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # Skip gradient all-reduce on intermediate accumulation steps.
            # Only the final micro-batch in the window needs to synchronize.
            is_accumulating = (batch_idx + 1) % config.grad_accum_steps != 0
            sync_context = model.no_sync() if is_accumulating else nullcontext()

            # A. Forward Pass
            with sync_context:
                with torch.autocast(device_type=device.type, dtype=config.dtype):
                    outputs = model(x, labels=y)
                    loss = outputs.loss / config.grad_accum_steps

                # B. Backward Pass (all-reduce only fires on the final micro-batch)
                loss.backward()

            accum_loss += loss.detach()

            # C. Optimizer Step — when accumulation window completes
            if (batch_idx + 1) % config.grad_accum_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                step += 1
                total_tokens += tokens_per_step_global

                # All ranks participate in the memory all_reduce every log interval,
                # then only rank 0 prints and writes to TensorBoard.
                if step % config.log_every == 0:
                    if torch.cuda.is_available():
                        local_mem = torch.cuda.max_memory_allocated()
                        mem_tensor = torch.tensor([local_mem], device=device, dtype=torch.long)
                        dist.all_reduce(mem_tensor, op=dist.ReduceOp.MAX)
                        peak_mem_gb = mem_tensor.item() / (1024 ** 3)

                    if is_main:
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        t1 = time.time()
                        dt = t1 - t0

                        tokens_in_window = tokens_per_step_global * config.log_every
                        tps = tokens_in_window / dt
                        true_loss = accum_loss.item()
                        current_lr = scheduler.get_last_lr()[0]

                        print(
                            f"Step {step:>4d} | "
                            f"Loss: {true_loss:.4f} | "
                            f"LR: {current_lr:.2e} | "
                            f"Grad Norm: {grad_norm:.2f} | "
                            f"TPS: {tps:,.0f} | "
                            f"Tokens: {total_tokens:,}"
                        )

                        writer.add_scalar("Train/Loss", true_loss, step)
                        writer.add_scalar("Train/TokensPerSec", tps, step)
                        writer.add_scalar("Train/LearningRate", current_lr, step)
                        writer.add_scalar("Train/GradNorm", grad_norm, step)

                        if torch.cuda.is_available():
                            local_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
                            writer.add_scalar("System/Peak_GPU_Memory_Allocated_GB", peak_mem_gb, step)
                            writer.add_scalar("System/GPU_Memory_Reserved_GB_Rank0", local_reserved, step)

                        t0 = time.time()

                accum_loss = 0.0

                # Checkpoint + exit
                if step >= config.max_steps:
                    save_distributed_checkpoint(model, optimizer, scheduler, step, config, rank, use_fsdp)
                    break

        epoch += 1

    if writer:
        writer.close()

    # All ranks must participate in the final all_reduce before rank 0 prints.
    if torch.cuda.is_available():
        local_peak = torch.cuda.max_memory_allocated()
        peak_tensor = torch.tensor([local_peak], device=device, dtype=torch.long)
        dist.all_reduce(peak_tensor, op=dist.ReduceOp.MAX)

    if is_main:
        print(f"\n--- Training Complete ({mode_name}) ---")
        print(f"  World size: {world_size}")
        print(f"  Steps completed: {step}")
        print(f"  Total tokens processed (all GPUs): {total_tokens:,}")
        if torch.cuda.is_available():
            print(f"  Peak GPU memory allocated (max across ranks): {peak_tensor.item() / (1024**3):.2f} GB")
            print(f"  GPU memory reserved (rank 0): {torch.cuda.memory_reserved() / (1024**3):.2f} GB")

    cleanup()


if __name__ == "__main__":
    import sys

    # Usage: in 
    #        torchrun --nproc-per-node=2 train_distributed_node.py --mode=ddp
    use_fsdp = "--mode=fsdp" in sys.argv
    run_experiment(use_fsdp=use_fsdp)
