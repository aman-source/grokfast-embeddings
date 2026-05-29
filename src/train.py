#!/usr/bin/env python3
"""
Unified training script for the Grokfast-embeddings experiment.

Usage:
  python src/train.py --config configs/stage1.yaml
  python src/train.py --config configs/stage2.yaml --seeds 0 --tag gpu0
  python src/train.py --config configs/stage2.yaml --report-only

The config YAML specifies everything: architecture, training hyperparameters,
arm definitions, and stage (1 = four-arm sequential; 2 = multi-seed parallel).
CLI flags can override seeds, tag, device, and operational modes.

Fairness guarantee: within any seed, baseline and grokfast_emb start from
IDENTICAL model weights (same torch.manual_seed(seed) init). No optimizer
state, EMA buffer, or data-reader position is shared across seeds or arms.
"""

import argparse
import contextlib
import json
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

# Adjust path so 'src/' is importable when running from repo root
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent))

from src.model import LlamaDecoder
from src.data  import tokenize_and_cache, TokenDataset, compute_token_buckets
from src.grokfast import apply_grokfast
from src.report import report_kill_criteria


# ============================================================================
# LOGGING -- tee stdout+stderr to timestamped file
# ============================================================================

class _Tee:
    def __init__(self, stream, log_file):
        self._stream = stream
        self._file   = log_file

    def write(self, data: str) -> None:
        self._stream.write(data)
        self._file.write(data)

    def flush(self) -> None:
        self._stream.flush()
        self._file.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def _setup_logging(results_dir: str, tag: str = "") -> None:
    os.makedirs(results_dir, exist_ok=True)
    stamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix   = f"_{tag}" if tag else ""
    log_path = os.path.join(results_dir, f"run_{stamp}{suffix}.log")
    log_file = open(log_path, "w", encoding="utf-8", buffering=1)
    sys.stdout = _Tee(sys.stdout, log_file)
    sys.stderr = _Tee(sys.stderr, log_file)
    print(f"[log] {log_path}")


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    # -- Stage --
    stage: int = 1       # 1 = four-arm sequential; 2 = multi-seed two-arm

    # -- Architecture --
    n_layers:   int   = 8
    d_model:    int   = 512
    n_heads:    int   = 8
    d_ff:       int   = 1408
    seq_len:    int   = 1024
    vocab_size: int   = 50257

    # -- Training --
    lr:            float = 3e-4
    lr_min:        float = 3e-5
    beta1:         float = 0.9
    beta2:         float = 0.95
    weight_decay:  float = 0.1
    grad_clip:     float = 1.0
    batch_size:    int   = 64
    grad_accum:    int   = 1
    tokens_total:  int   = 500_000_000
    warmup_steps:  int   = 300
    eval_interval: int   = 250
    eval_batches:  int   = 50

    # -- Grokfast defaults (overridden per-arm if needed) --
    gf_alpha: float = 0.98
    gf_lamb:  float = 2.0

    # -- Experiment --
    seed:  int  = 42            # stage 1: single seed
    seeds: list = field(default_factory=lambda: [0, 1])  # stage 2: seed list
    arms:  list = field(default_factory=list)             # list of arm dicts

    # -- Paths --
    data_cache:  str = "fineweb_edu_tokens.bin"
    results_dir: str = "results/stage1"
    device:      str = "cuda"

    # -- Stage 2 multi-GPU --
    tag:         str  = ""
    report_only: bool = False

    # -- Smoke-test / override --
    smoke_test:     bool = False
    override_steps: int  = 0

    @property
    def micro_batch_size(self) -> int:
        return self.batch_size // self.grad_accum

    @property
    def tokens_per_step(self) -> int:
        return self.batch_size * self.seq_len

    @property
    def total_steps(self) -> int:
        if self.override_steps > 0:
            return self.override_steps
        return self.tokens_total // self.tokens_per_step


def load_config(config_path: str) -> Config:
    """Load a YAML config file and return a Config dataclass."""
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    cfg = Config()
    for k, v in raw.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Grokfast-embeddings training")
    parser.add_argument("--config",       type=str, required=True,
                        help="Path to YAML config (e.g. configs/stage2.yaml)")
    parser.add_argument("--smoke-test",   action="store_true")
    parser.add_argument("--seeds",        type=int, nargs="+", default=None,
                        help="Stage 2: which seeds to run on this GPU (e.g. --seeds 0 1)")
    parser.add_argument("--tag",          type=str, default=None,
                        help="Namespace tag for log and summary JSON (e.g. gpu0)")
    parser.add_argument("--report-only",  action="store_true",
                        help="Skip training; read all per-run JSONs and print final verdict.")
    parser.add_argument("--device",       type=str, default=None)
    parser.add_argument("--results-dir",  type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Apply CLI overrides
    if args.smoke_test:    cfg.smoke_test  = True
    if args.report_only:   cfg.report_only = True
    if args.seeds:         cfg.seeds       = args.seeds
    if args.tag is not None:        cfg.tag         = args.tag
    if args.device is not None:     cfg.device      = args.device
    if args.results_dir is not None: cfg.results_dir = args.results_dir

    if cfg.smoke_test:
        _apply_smoke_overrides(cfg)

    return cfg


def _apply_smoke_overrides(cfg: Config) -> None:
    cfg.n_layers       = 2
    cfg.d_model        = 64
    cfg.n_heads        = 4
    cfg.d_ff           = 128
    cfg.seq_len        = 16
    cfg.batch_size     = 2
    cfg.grad_accum     = 1
    cfg.override_steps = 4
    cfg.tokens_total   = 100_000
    cfg.warmup_steps   = 1
    cfg.eval_interval  = 2
    cfg.eval_batches   = 5
    cfg.device         = "cpu"
    cfg.data_cache     = f"smoke_tokens_stage{cfg.stage}.bin"
    cfg.results_dir    = f"results/smoke_stage{cfg.stage}"
    if cfg.stage == 2 and (not cfg.seeds or cfg.seeds == [0, 1]):
        cfg.seeds = [0, 1]


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

def _fmt_time(seconds: float) -> str:
    s = int(seconds)
    if s < 60:   return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:   return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    if h < 24:   return f"{h}h{m:02d}m"
    d, h = divmod(h, 24)
    return f"{d}d{h:02d}h{m:02d}m"


def cosine_lr(
    step: int, total_steps: int, warmup_steps: int, lr: float, lr_min: float
) -> float:
    if step < warmup_steps:
        return lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return lr_min + 0.5 * (lr - lr_min) * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(
    model:         nn.Module,
    val_reader,
    token_buckets: np.ndarray,
    device:        torch.device,
    n_batches:     int,
    vocab_size:    int,
) -> tuple[float, list[float]]:
    """Returns (val_loss, [perp_top1k, perp_1k10k, perp_rare])."""
    model.eval()
    val_reader.pos = 0

    ctx = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if device.type == "cuda" else contextlib.nullcontext()
    )

    total_loss   = 0.0
    bucket_nll   = [0.0, 0.0, 0.0]
    bucket_count = [0,   0,   0  ]

    for _ in range(n_batches):
        x, y = val_reader.next_batch(device)
        with ctx:
            logits       = model(x)
            loss_per_tok = F.cross_entropy(
                logits.view(-1, vocab_size), y.view(-1), reduction="none"
            )
        total_loss += loss_per_tok.mean().item()
        y_cpu   = y.view(-1).cpu().numpy()
        lpt_cpu = loss_per_tok.float().cpu().numpy()
        for b in range(3):
            mask = token_buckets[y_cpu] == b
            if mask.any():
                bucket_nll[b]   += float(lpt_cpu[mask].mean())
                bucket_count[b] += 1

    val_loss    = total_loss / n_batches
    bucket_perp = [math.exp(bucket_nll[b] / max(1, bucket_count[b])) for b in range(3)]
    model.train()
    return val_loss, bucket_perp


def save_results(results: dict, results_dir: str, run_id: str) -> None:
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, f"{run_id}.json"), "w") as f:
        json.dump(results, f, indent=2)


def _run_id(arm_name: str, seed: int | None) -> str:
    """Canonical filename stem for a run."""
    if seed is None:
        return arm_name                     # stage 1: just the arm name
    return f"{arm_name}_seed{seed}"         # stage 2: arm + seed


# ============================================================================
# PER-RUN TRAINING LOOP
# ============================================================================

def train_run(
    arm_cfg:          dict,         # {name, beta1, grokfast, emb_only, gf_lamb (opt)}
    seed:             int | None,   # None for stage 1 (seed lives in cfg)
    cfg:              Config,
    model_init_state: dict,
    dataset:          TokenDataset,
    token_buckets:    np.ndarray,
    device:           torch.device,
    run_idx:          int   = 0,
    total_runs:       int   = 1,
    exp_start_time:   float = 0.0,
) -> dict:
    """
    Train one (arm, seed) run from the saved initial weights.

    Fairness guarantee: model_init_state was produced by torch.manual_seed(seed)
    and is loaded fresh here. No state from any previous run is carried in.
    """
    arm_name = arm_cfg["name"]
    run_id   = _run_id(arm_name, seed)
    beta1    = arm_cfg.get("beta1", cfg.beta1)
    use_gf   = arm_cfg.get("grokfast", False)
    emb_only = arm_cfg.get("emb_only", False)
    lamb     = arm_cfg.get("gf_lamb", cfg.gf_lamb)

    eff_seed = seed if seed is not None else cfg.seed
    print(f"\n{'='*64}")
    print(f"RUN: {run_id}  |  arm={arm_name}  seed={eff_seed}  beta1={beta1}  grokfast={use_gf}  emb_only={emb_only}")
    print(f"{'='*64}")

    # Reload identical initial weights (fairness rule)
    torch.manual_seed(eff_seed)
    model = LlamaDecoder(cfg).to(device)
    model.load_state_dict(model_init_state)
    model.train()

    optim_kwargs: dict = dict(
        lr=cfg.lr, betas=(beta1, cfg.beta2),
        weight_decay=cfg.weight_decay, eps=1e-8,
    )
    if device.type == "cuda":
        optim_kwargs["fused"] = True
    try:
        optimizer = torch.optim.AdamW(model.parameters(), **optim_kwargs)
    except TypeError:
        optim_kwargs.pop("fused", None)
        optimizer = torch.optim.AdamW(model.parameters(), **optim_kwargs)

    ema_buffers: dict = {}  # zero-initialized on first use; no leakage across runs
    tr_reader = dataset.train_reader()
    vl_reader = dataset.val_reader()

    total_steps = cfg.total_steps
    ctx = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if device.type == "cuda" else contextlib.nullcontext()
    )

    results: dict = {
        "run_id":            run_id,
        "arm":               arm_name,
        "seed":              eff_seed,
        "gf_alpha":          cfg.gf_alpha,
        "gf_lamb":           lamb,
        "steps":             [],
        "tokens_seen":       [],
        "val_loss":          [],
        "bucket_perp_top1k": [],
        "bucket_perp_1k10k": [],
        "bucket_perp_rare":  [],
        "step_times_ms":     [],
    }

    tokens_seen    = 0
    recent_step_ms: list[float] = []

    for step in range(total_steps):
        t0 = time.perf_counter()

        lr = cosine_lr(step, total_steps, cfg.warmup_steps, cfg.lr, cfg.lr_min)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        for _ in range(cfg.grad_accum):
            x, y = tr_reader.next_batch(device)
            with ctx:
                logits = model(x)
                loss   = (
                    F.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1))
                    / cfg.grad_accum
                )
            loss.backward()

        tokens_seen += cfg.tokens_per_step
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        # -----------------------------------------------------------------
        # Grokfast-EMA (see src/grokfast.py) -- applied only when arm uses it
        # -----------------------------------------------------------------
        if use_gf:
            apply_grokfast(model, ema_buffers, cfg.gf_alpha, lamb, emb_only=emb_only)

        optimizer.step()

        step_ms = (time.perf_counter() - t0) * 1000.0
        recent_step_ms.append(step_ms)
        if len(recent_step_ms) > 50:
            recent_step_ms.pop(0)
        avg_step_s = (sum(recent_step_ms) / len(recent_step_ms)) / 1000.0

        is_last = step == total_steps - 1
        if step % cfg.eval_interval == 0 or is_last:
            val_loss, bp = evaluate(
                model, vl_reader, token_buckets, device, cfg.eval_batches, cfg.vocab_size
            )
            results["steps"].append(step)
            results["tokens_seen"].append(tokens_seen)
            results["val_loss"].append(round(val_loss, 6))
            results["bucket_perp_top1k"].append(round(bp[0], 4))
            results["bucket_perp_1k10k"].append(round(bp[1], 4))
            results["bucket_perp_rare"].append(round(bp[2], 4))
            results["step_times_ms"].append(round(step_ms, 2))

            eta_run_s       = (total_steps - step - 1) * avg_step_s
            steps_done_exp  = run_idx * total_steps + step + 1
            eta_exp_s       = (total_runs * total_steps - steps_done_exp) * avg_step_s
            elapsed_s       = time.perf_counter() - exp_start_time if exp_start_time else 0.0

            print(
                f"  step {step:>6}/{total_steps}"
                f" | tok {tokens_seen / 1e9:>5.2f}B"
                f" | val_loss {val_loss:.4f}"
                f" | perp [{bp[0]:.1f} / {bp[1]:.1f} / {bp[2]:.1f}]"
                f" | lr {lr:.2e}"
                f" | {step_ms:.0f}ms/step"
                f" | run_eta {_fmt_time(eta_run_s)}"
                f" | exp_eta {_fmt_time(eta_exp_s)}"
                f" | elapsed {_fmt_time(elapsed_s)}"
                f"  [{run_idx+1}/{total_runs}]"
            )
            # Crash-safe: flush to disk after every eval
            save_results(results, cfg.results_dir, run_id)

    return results


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    cfg = parse_args()
    _setup_logging(cfg.results_dir, cfg.tag)

    # ------------------------------------------------------------------
    # --report-only: scan JSONs and print final verdict without training
    # ------------------------------------------------------------------
    if cfg.report_only:
        print(f"[report-only] scanning {cfg.results_dir}/ ...")
        all_results: dict = {}
        if os.path.isdir(cfg.results_dir):
            for fname in sorted(os.listdir(cfg.results_dir)):
                if fname.endswith(".json") and not fname.startswith("summary"):
                    with open(os.path.join(cfg.results_dir, fname)) as f:
                        r = json.load(f)
                    run_id = r.get("run_id", fname[:-5])
                    all_results[run_id] = r
                    print(f"  loaded {fname}  steps={len(r.get('steps', []))}")
        report_kill_criteria(all_results, stage=cfg.stage, full_verdict=True)
        return

    # -- Determinism --
    np.random.seed(0)
    if cfg.device != "cpu":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False

    # -- Device --
    if cfg.device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA not available -- falling back to CPU")
        cfg.device = "cpu"
    device = torch.device(cfg.device)

    print(
        f"\n[config] stage={cfg.stage} device={device}"
        f" | model={cfg.n_layers}L/{cfg.d_model}H/{cfg.d_ff}FF"
        f" | steps={cfg.total_steps:,}"
        f" | tokens/step={cfg.tokens_per_step:,}"
        f" | tokens_total={cfg.tokens_total/1e9:.2f}B"
        f" | lr={cfg.lr}  gf_alpha={cfg.gf_alpha}  gf_lamb={cfg.gf_lamb}"
    )

    # -- Data --
    if cfg.smoke_test:
        _make_smoke_data(cfg)
    else:
        tokenize_and_cache(cfg)
    dataset       = TokenDataset(cfg)
    token_buckets = compute_token_buckets(dataset.train, cfg.vocab_size)

    # -- Build run queue and initial states --
    if cfg.stage == 1:
        run_queue, init_states = _build_stage1_queue(cfg, device)
    else:
        run_queue, init_states = _build_stage2_queue(cfg, device)

    total_runs = len(run_queue)
    print(f"[runs] {total_runs} runs queued")

    all_results: dict = {}
    failed_runs: list[str] = []
    exp_start   = time.perf_counter()

    for run_idx, (arm_cfg, seed) in enumerate(run_queue):
        run_id   = _run_id(arm_cfg["name"], seed)
        out_path = os.path.join(cfg.results_dir, f"{run_id}.json")

        # Crash recovery: skip if already complete
        if os.path.exists(out_path):
            with open(out_path) as f:
                cached = json.load(f)
            last_step = cached["steps"][-1] if cached.get("steps") else -1
            if last_step >= cfg.total_steps - 1:
                print(f"[skip] {run_id} already complete  [{run_idx+1}/{total_runs}]")
                all_results[run_id] = cached
                continue

        eff_seed = seed if seed is not None else cfg.seed
        try:
            results = train_run(
                arm_cfg, seed, cfg,
                init_states[eff_seed],
                dataset, token_buckets, device,
                run_idx=run_idx,
                total_runs=total_runs,
                exp_start_time=exp_start,
            )
            all_results[run_id] = results
        except Exception:
            print(f"\n[ERROR] run '{run_id}' crashed -- continuing.")
            print(traceback.format_exc())
            failed_runs.append(run_id)

    # Save combined summary (tagged to avoid multi-GPU clobber)
    os.makedirs(cfg.results_dir, exist_ok=True)
    summary_name = f"summary_{cfg.tag}.json" if cfg.tag else "summary.json"
    with open(os.path.join(cfg.results_dir, summary_name), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[done] results -> {cfg.results_dir}/  (summary: {summary_name})")

    if failed_runs:
        print(f"[warn] crashed runs: {failed_runs}")

    report_kill_criteria(all_results, stage=cfg.stage, full_verdict=False)


# ============================================================================
# QUEUE BUILDERS
# ============================================================================

def _build_stage1_queue(
    cfg: Config, device: torch.device
) -> tuple[list, dict]:
    """
    Stage 1: all arms run with the same seed and identical init weights.
    Returns (run_queue, init_states) where init_states = {seed: state_dict}.
    """
    torch.manual_seed(cfg.seed)
    m = LlamaDecoder(cfg).to(device)
    print(f"[model] unique params: {m.num_params():,}")
    init_states = {cfg.seed: {k: v.clone() for k, v in m.state_dict().items()}}
    del m

    run_queue = [(arm, None) for arm in cfg.arms]
    return run_queue, init_states


def _build_stage2_queue(
    cfg: Config, device: torch.device
) -> tuple[list, dict]:
    """
    Stage 2: each seed gets its own init from torch.manual_seed(seed).
    All arms at the same seed start from identical weights.
    Seeds are independent -- no state leaks across seeds.
    Returns (run_queue, init_states) where init_states = {seed: state_dict}.
    """
    init_states: dict = {}
    printed_params = False
    for seed in sorted(set(cfg.seeds)):
        torch.manual_seed(seed)
        m = LlamaDecoder(cfg).to(device)
        if not printed_params:
            print(f"[model] unique params: {m.num_params():,}")
            printed_params = True
        init_states[seed] = {k: v.clone() for k, v in m.state_dict().items()}
        del m

    run_queue = [(arm, seed) for seed in cfg.seeds for arm in cfg.arms]
    return run_queue, init_states


# ============================================================================
# SMOKE TEST DATA HELPER (tests/ uses this too)
# ============================================================================

def _make_smoke_data(cfg: Config) -> None:
    if os.path.exists(cfg.data_cache):
        return
    seed = cfg.seeds[0] if cfg.stage == 2 and cfg.seeds else cfg.seed
    rng  = np.random.default_rng(seed)
    tokens = rng.integers(0, cfg.vocab_size, size=cfg.tokens_total, dtype=np.uint16)
    with open(cfg.data_cache, "wb") as f:
        f.write(tokens.tobytes())
    print(f"[smoke] {len(tokens):,} random tokens -> {cfg.data_cache}")


if __name__ == "__main__":
    main()
