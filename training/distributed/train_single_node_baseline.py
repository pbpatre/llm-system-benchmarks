import os
import time
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import GPT2Config, GPT2LMHeadModel, AutoTokenizer
from torch.utils.tensorboard import SummaryWriter
from dataclasses import dataclass
from typing import Tuple

# --- Configuration ---
@dataclass
class TrainingConfig:
    # Model Architecture
    batch_size: int = 8           # Micro-batch per forward pass (fits in VRAM)
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
    grad_accum_steps: int = 4     # Effective batch = 8 * 4 = 32
    max_grad_norm: float = 1.0    # Gradient clipping threshold

    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    use_compile: bool = False     # torch.compile for kernel fusion speedup
    num_workers: int = 4          # DataLoader workers

    # Logging & Checkpointing
    log_every: int = 10           # Log metrics every N steps
    checkpoint_dir: str = "checkpoints/day1_baseline"
    log_dir: str = "runs/day1_baseline"
    seed: int = 42


def set_seed(seed: int):
    """Reproducibility: seed all RNGs so runs are deterministic."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def configure_tf32():
    """Enable TF32 on Ampere+ GPUs for ~2-3x faster matmuls with negligible accuracy impact."""
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


# --- 1. Learning Rate Schedule (Warmup + Cosine Decay) ---
def get_lr_scheduler(optimizer: torch.optim.Optimizer, cfg: TrainingConfig) -> LambdaLR:
    """
    Linear warmup for `warmup_steps`, then cosine decay to `min_lr`.
    Without warmup, large early gradients destabilize transformer training.
    Without decay, the optimizer overshoots near convergence.
    """
    def lr_lambda(step: int) -> float:
        # Warmup phase: linearly ramp from 0 to peak lr
        if step < cfg.warmup_steps:
            return step / max(1, cfg.warmup_steps)
        # Decay phase: cosine anneal from peak lr down to min_lr
        decay_steps = cfg.max_steps - cfg.warmup_steps
        progress = (step - cfg.warmup_steps) / max(1, decay_steps)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        # Scale so we decay to min_lr/lr, not to 0
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

    This is the standard pretraining format: every token position is real data.
    No padding, no waste — exactly what sequence packing achieves, but simpler.
    Documents are separated by EOS tokens so the model learns document boundaries.
    """
    def __init__(self, dataset_name: str, dataset_config: str, tokenizer_name: str,
                 seq_len: int, split: str = "train"):
        from datasets import load_dataset

        print(f"Loading dataset: {dataset_name}/{dataset_config} [{split}]...")
        raw_dataset = load_dataset(dataset_name, dataset_config, split=split)

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        eos_token = tokenizer.eos_token_id

        # Tokenize in batches (much faster than one-by-one)
        print(f"Tokenizing with {tokenizer_name}...")
        def tokenize_fn(examples):
            # Tokenize text, append EOS to mark document boundaries
            out = tokenizer(examples["text"], add_special_tokens=False)
            # Append EOS after each document so model learns to separate them
            for ids in out["input_ids"]:
                ids.append(eos_token)
            return out

        tokenized = raw_dataset.map(
            tokenize_fn,
            batched=True,
            remove_columns=raw_dataset.column_names,
            num_proc=4,
            desc="Tokenizing",
        )

        # Concatenate all tokens into one flat array
        # This is the "shard" approach from the pipeline analysis — one contiguous read
        print("Concatenating tokens...")
        all_ids = []
        for example in tokenized:
            all_ids.extend(example["input_ids"])

        self.tokens = torch.tensor(all_ids, dtype=torch.long)
        self.seq_len = seq_len

        # Number of complete (seq_len + 1) chunks (the +1 is for the shifted target)
        self.n_sequences = (len(self.tokens) - 1) // seq_len
        # Trim to exact multiple so every chunk is full
        self.tokens = self.tokens[: self.n_sequences * seq_len + 1]

        print(f"Dataset ready: {len(self.tokens):,} tokens → {self.n_sequences:,} sequences of {seq_len}")

    def __len__(self) -> int:
        return self.n_sequences

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.seq_len
        chunk = self.tokens[start : start + self.seq_len + 1]
        return chunk[:-1], chunk[1:]


# --- 3. The Model (Memory Allocator) ---
def get_model(cfg: TrainingConfig) -> nn.Module:
    """
    Initializes a GPT-2 style model from scratch.
    System Goal: Allocate parameters in VRAM.
    """
    model_config = GPT2Config(
        vocab_size=cfg.vocab_size,
        n_positions=cfg.seq_len,
        n_embd=cfg.d_model,
        n_layer=cfg.n_layers,
        n_head=cfg.n_heads,
        use_cache=False,  # Disable KV cache for training (saves VRAM)
    )
    model = GPT2LMHeadModel(model_config)
    return model.to(cfg.device)


def save_checkpoint(model, optimizer, scheduler, step, cfg: TrainingConfig):
    """Save model + optimizer + scheduler state for resumable training."""
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    path = os.path.join(cfg.checkpoint_dir, f"step_{step}.pt")
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": cfg,
    }, path)
    print(f"  Checkpoint saved: {path}")


# --- Main ---
def main():
    config = TrainingConfig()

    set_seed(config.seed)
    configure_tf32()

    print(f"Device: {config.device}")
    print(f"Precision: {config.dtype}")
    print(f"Compile: {config.use_compile}")
    if torch.cuda.is_available() and torch.backends.cuda.matmul.allow_tf32:
        print(f"TF32:   enabled (Ampere+ GPU detected)")

    # 1. Init Data
    if config.use_synthetic:
        print("Using synthetic (random token) dataset")
        dataset = SyntheticTextDataset(config.vocab_size, config.seq_len)
    else:
        dataset = HuggingFaceTextDataset(
            dataset_name=config.dataset_name,
            dataset_config=config.dataset_config,
            tokenizer_name=config.tokenizer_name,
            seq_len=config.seq_len,
        )
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=not config.use_synthetic,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=config.num_workers > 0,
    )

    # 2. Init Model
    model = get_model(config)
    if config.use_compile:
        model = torch.compile(model)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {n_params / 1e6:.1f}M")

    # 3. Init Optimizer
    # AdamW maintains 2 states (moment, variance) per parameter.
    # Memory footprint: 1x params + 2x optimizer states = 3x total.
    optimizer = AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),      # Standard for LLM pretraining (GPT-3, LLaMA)
        fused=config.device == "cuda",  # Fused CUDA kernel: fewer memory passes
    )

    # 4. Init LR Scheduler (warmup + cosine decay)
    scheduler = get_lr_scheduler(optimizer, config)

    # --- Training Loop ---
    writer = SummaryWriter(log_dir=config.log_dir)
    model.train()

    step = 0
    total_tokens = 0
    accum_loss = 0.0
    tokens_per_step = config.batch_size * config.seq_len * config.grad_accum_steps

    if config.device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()

    print("Starting Training Loop...")

    while step < config.max_steps:
        for batch_idx, (x, y) in enumerate(dataloader):
            x = x.to(config.device, non_blocking=True)
            y = y.to(config.device, non_blocking=True)

            # A. Forward Pass
            with torch.autocast(device_type=config.device, dtype=config.dtype):
                outputs = model(x, labels=y)
                loss = outputs.loss / config.grad_accum_steps

            # B. Backward Pass
            loss.backward()

            accum_loss += loss.detach()

            # C. Optimizer Step — update weights when accumulation window completes
            if (batch_idx + 1) % config.grad_accum_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                step += 1
                total_tokens += tokens_per_step

                # --- Instrumentation ---
                if step % config.log_every == 0:
                    if config.device == "cuda":
                        torch.cuda.synchronize()
                    t1 = time.time()
                    dt = t1 - t0

                    tokens_in_window = tokens_per_step * config.log_every
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

                    if config.device == "cuda":
                        mem_alloc = torch.cuda.max_memory_allocated() / (1024 ** 3)
                        mem_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
                        writer.add_scalar("System/GPU_Memory_Allocated_GB", mem_alloc, step)
                        writer.add_scalar("System/GPU_Memory_Reserved_GB", mem_reserved, step)

                    t0 = time.time()

                accum_loss = 0.0

                if step >= config.max_steps:
                    save_checkpoint(model, optimizer, scheduler, step, config)
                    break

    writer.close()

    # --- Final Summary ---
    print("\n--- Training Complete ---")
    print(f"  Steps completed: {step}")
    print(f"  Total tokens processed: {total_tokens:,}")
    if config.device == "cuda":
        print(f"  Peak GPU memory allocated: {torch.cuda.max_memory_allocated() / (1024**3):.2f} GB")
        print(f"  GPU memory reserved: {torch.cuda.memory_reserved() / (1024**3):.2f} GB")


if __name__ == "__main__":
    main()
