"""
Data utilities: tokenize FineWeb-Edu once to disk; serve sequential batches.

The core design constraint: ALL arms and ALL seeds read the SAME token stream
in the SAME order from the SAME pre-tokenized file. This is enforced by:
  1. tokenize_and_cache() writes exactly once; subsequent calls are no-ops.
  2. Every SequentialReader starts at position 0.
  3. No shuffling anywhere in the pipeline.
"""

import os
import math
import numpy as np
import torch
from typing import Protocol


class DataCfg(Protocol):
    data_cache:       str
    micro_batch_size: int
    seq_len:          int
    eval_batches:     int
    vocab_size:       int
    tokens_total:     int


def tokenize_and_cache(cfg: DataCfg) -> None:
    """
    Stream FineWeb-Edu, tokenize with GPT-2, write uint16 binary to disk.
    No-op if the cache file already exists.
    """
    if os.path.exists(cfg.data_cache):
        n = os.path.getsize(cfg.data_cache) // 2
        print(f"[data] cache found: {cfg.data_cache} ({n:,} tokens). Skipping.")
        return

    print(f"[data] tokenizing FineWeb-Edu -> {cfg.data_cache} ...")
    import tiktoken
    from datasets import load_dataset

    enc    = tiktoken.get_encoding("gpt2")
    eot    = enc.eot_token
    target = cfg.tokens_total + 25_000_000   # extra to cover val split

    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )

    written  = 0
    milestone = 200_000_000
    with open(cfg.data_cache, "wb") as f:
        for doc in ds:
            tokens = enc.encode_ordinary(doc["text"])
            tokens.append(eot)
            arr = np.array(tokens, dtype=np.uint16)
            f.write(arr.tobytes())
            written += len(tokens)
            if written >= milestone:
                print(f"  {written / 1e9:.2f}B / {target / 1e9:.2f}B tokens ...")
                milestone += 200_000_000
            if written >= target:
                break

    print(f"[data] done: {written:,} tokens -> {cfg.data_cache}")


class TokenDataset:
    """
    Memory-mapped token file. First ~0.5% is held out as a fixed validation
    shard; the rest is training data. All arms share the same memmap; each
    arm/seed gets its own SequentialReader starting at position 0.
    """
    VAL_RATIO = 0.005

    def __init__(self, cfg: DataCfg):
        data    = np.memmap(cfg.data_cache, dtype=np.uint16, mode="r")
        min_val = cfg.micro_batch_size * (cfg.seq_len + 1) * cfg.eval_batches
        n_val   = max(min_val, int(len(data) * self.VAL_RATIO))
        # Align to stride so sequential reads don't straddle the split boundary
        stride  = cfg.micro_batch_size * cfg.seq_len
        n_val   = ((n_val + stride - 1) // stride) * stride
        self.val   = data[:n_val]
        self.train = data[n_val:]
        self.cfg   = cfg
        print(f"[data] train: {len(self.train):,} tokens | val: {len(self.val):,} tokens")

    def train_reader(self) -> "SequentialReader":
        return SequentialReader(self.train, self.cfg.micro_batch_size, self.cfg.seq_len)

    def val_reader(self) -> "SequentialReader":
        return SequentialReader(self.val, self.cfg.micro_batch_size, self.cfg.seq_len)


class SequentialReader:
    """Yields (x, y) batches in order. Wraps to position 0 at end of data."""

    def __init__(self, data: np.ndarray, batch_size: int, seq_len: int):
        self.data       = data
        self.batch_size = batch_size
        self.seq_len    = seq_len
        self.pos        = 0

    def next_batch(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        B, T   = self.batch_size, self.seq_len
        needed = B * T + 1
        if self.pos + needed > len(self.data):
            self.pos = 0
        chunk  = self.data[self.pos : self.pos + needed].astype(np.int64)
        self.pos += B * T
        x = torch.from_numpy(chunk[:-1].reshape(B, T)).to(device)
        y = torch.from_numpy(chunk[1: ].reshape(B, T)).to(device)
        return x, y


def compute_token_buckets(train_data: np.ndarray, vocab_size: int) -> np.ndarray:
    """
    Assign each vocab token to a frequency bucket:
      0 = top-1K most frequent
      1 = 1K - 10K
      2 = 10K+ (rare)
    Computed from the first 50M training tokens. Returns int8 array (vocab_size,).
    """
    n_sample = min(50_000_000, len(train_data))
    counts   = np.bincount(train_data[:n_sample].astype(np.int64), minlength=vocab_size)
    order    = np.argsort(-counts)
    buckets  = np.full(vocab_size, 2, dtype=np.int8)
    buckets[order[:1_000]]       = 0
    buckets[order[1_000:10_000]] = 1
    return buckets
