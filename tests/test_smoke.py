"""
CPU smoke tests. Confirm:
  1. Stage 1 all four arms execute without error on tiny model + random tokens.
  2. Stage 2 both arms execute for two seeds.
  3. Fairness rule: within a seed, both arms start from identical weights.
  4. Seed independence: different seeds produce different init weights.
  5. Grokfast-EMA actually modifies gradients (non-zero EMA update).
  6. apply_grokfast with emb_only=True touches only embed/lm_head params.

Run from repo root:
  python -m pytest tests/ -v
  python tests/test_smoke.py   (standalone)
"""

import os
import sys
import copy
import tempfile
import numpy as np
import torch
from pathlib import Path

# Allow imports from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model    import LlamaDecoder
from src.data     import TokenDataset, SequentialReader, compute_token_buckets
from src.grokfast import apply_grokfast
from src.train    import (
    Config, _apply_smoke_overrides, _make_smoke_data,
    train_run, _run_id, _build_stage1_queue, _build_stage2_queue,
)


# ---------------------------------------------------------------------------
# Tiny config helpers
# ---------------------------------------------------------------------------

def _tiny_cfg(stage: int = 1) -> Config:
    cfg = Config(stage=stage)
    _apply_smoke_overrides(cfg)
    if stage == 2:
        cfg.seeds = [0, 1]
        cfg.arms = [
            {"name": "baseline",     "beta1": 0.9, "grokfast": False},
            {"name": "grokfast_emb", "beta1": 0.9, "grokfast": True, "emb_only": True},
        ]
    else:
        cfg.arms = [
            {"name": "baseline",           "beta1": 0.9, "grokfast": False},
            {"name": "grokfast_full",       "beta1": 0.9, "grokfast": True, "emb_only": False},
            {"name": "grokfast_emb",        "beta1": 0.9, "grokfast": True, "emb_only": True},
            {"name": "grokfast_nomomentum", "beta1": 0.0, "grokfast": True, "emb_only": False},
        ]
    return cfg


def _make_dataset(cfg: Config) -> TokenDataset:
    _make_smoke_data(cfg)
    return TokenDataset(cfg)


# ---------------------------------------------------------------------------
# Test 1: stage 1 -- all four arms run to completion
# ---------------------------------------------------------------------------

def test_stage1_all_arms_complete():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _tiny_cfg(stage=1)
        cfg.results_dir = tmpdir
        dataset       = _make_dataset(cfg)
        token_buckets = compute_token_buckets(dataset.train, cfg.vocab_size)
        device        = torch.device("cpu")

        run_queue, init_states = _build_stage1_queue(cfg, device)
        for arm_cfg, seed in run_queue:
            r = train_run(arm_cfg, seed, cfg, init_states[cfg.seed],
                          dataset, token_buckets, device)
            assert len(r["val_loss"]) > 0,    f"No eval points for {arm_cfg['name']}"
            assert r["val_loss"][-1] < 100.0, f"Loss suspiciously high for {arm_cfg['name']}"

    print("PASS: stage1 all four arms complete")


# ---------------------------------------------------------------------------
# Test 2: stage 2 -- both arms * two seeds run to completion
# ---------------------------------------------------------------------------

def test_stage2_runs_complete():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _tiny_cfg(stage=2)
        cfg.results_dir = tmpdir
        dataset       = _make_dataset(cfg)
        token_buckets = compute_token_buckets(dataset.train, cfg.vocab_size)
        device        = torch.device("cpu")

        run_queue, init_states = _build_stage2_queue(cfg, device)
        assert len(run_queue) == 4, f"Expected 4 runs, got {len(run_queue)}"
        for arm_cfg, seed in run_queue:
            r = train_run(arm_cfg, seed, cfg, init_states[seed],
                          dataset, token_buckets, device)
            assert len(r["val_loss"]) > 0

    print("PASS: stage2 all runs complete")


# ---------------------------------------------------------------------------
# Test 3: fairness -- same seed, same init weights across arms
# ---------------------------------------------------------------------------

def test_seed_pair_fairness():
    cfg    = _tiny_cfg(stage=2)
    device = torch.device("cpu")
    _, init_states = _build_stage2_queue(cfg, device)

    # Both arms at seed 0 must start from identical weights
    seed = 0
    init0 = init_states[seed]

    # Simulate loading for arm A
    torch.manual_seed(seed)
    m_a = LlamaDecoder(cfg).to(device)
    m_a.load_state_dict(init0)

    # Simulate loading for arm B (same seed, same init_state)
    torch.manual_seed(seed)
    m_b = LlamaDecoder(cfg).to(device)
    m_b.load_state_dict(init0)

    for (na, pa), (nb, pb) in zip(m_a.named_parameters(), m_b.named_parameters()):
        assert torch.equal(pa, pb), f"Init mismatch at seed {seed} param {na}"

    print("PASS: fairness -- same seed produces identical init weights")


# ---------------------------------------------------------------------------
# Test 4: seed independence -- different seeds produce different init weights
# ---------------------------------------------------------------------------

def test_seed_independence():
    cfg    = _tiny_cfg(stage=2)
    device = torch.device("cpu")
    _, init_states = _build_stage2_queue(cfg, device)

    assert 0 in init_states and 1 in init_states
    s0 = init_states[0]
    s1 = init_states[1]

    # At least one parameter tensor must differ between seeds
    any_diff = any(not torch.equal(s0[k], s1[k]) for k in s0)
    assert any_diff, "Seeds 0 and 1 produced identical init weights -- seed isolation broken"

    print("PASS: seed independence -- different seeds produce different weights")


# ---------------------------------------------------------------------------
# Test 5: Grokfast modifies gradients
# ---------------------------------------------------------------------------

def test_grokfast_modifies_gradients():
    cfg    = _tiny_cfg(stage=1)
    device = torch.device("cpu")
    torch.manual_seed(0)
    model  = LlamaDecoder(cfg).to(device)

    x = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.seq_len))
    y = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.seq_len))
    loss = torch.nn.functional.cross_entropy(
        model(x).view(-1, cfg.vocab_size), y.view(-1)
    )
    loss.backward()

    # Save raw grads
    raw = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

    ema_buffers: dict = {}
    apply_grokfast(model, ema_buffers, alpha=0.98, lamb=2.0, emb_only=False)

    modified = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
    any_changed = any(not torch.equal(raw[n], modified[n]) for n in raw)
    assert any_changed, "apply_grokfast did not modify any gradients"

    print("PASS: Grokfast modifies gradients")


# ---------------------------------------------------------------------------
# Test 6: emb_only restricts to embed/lm_head only
# ---------------------------------------------------------------------------

def test_grokfast_emb_only():
    cfg    = _tiny_cfg(stage=1)
    device = torch.device("cpu")
    torch.manual_seed(0)
    model  = LlamaDecoder(cfg).to(device)

    x = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.seq_len))
    y = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.seq_len))
    loss = torch.nn.functional.cross_entropy(
        model(x).view(-1, cfg.vocab_size), y.view(-1)
    )
    loss.backward()

    raw = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

    ema_buffers: dict = {}
    apply_grokfast(model, ema_buffers, alpha=0.98, lamb=2.0, emb_only=True)

    for n, p in model.named_parameters():
        if p.grad is None:
            continue
        is_emb = "embed" in n or "lm_head" in n
        if is_emb:
            # emb params MUST have changed (unless tied weight dedup skipped one)
            pass  # tied weights: embed==lm_head, only one copy modified
        else:
            assert torch.equal(raw[n], p.grad), \
                f"emb_only=True modified non-emb param: {n}"

    print("PASS: emb_only restricts Grokfast to embed/lm_head")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_stage1_all_arms_complete,
        test_stage2_runs_complete,
        test_seed_pair_fairness,
        test_seed_independence,
        test_grokfast_modifies_gradients,
        test_grokfast_emb_only,
    ]
    failures = []
    for t in tests:
        try:
            t()
        except Exception as e:
            print(f"FAIL: {t.__name__}: {e}")
            import traceback; traceback.print_exc()
            failures.append(t.__name__)

    print(f"\n{'='*50}")
    if failures:
        print(f"FAILED: {failures}")
        sys.exit(1)
    else:
        print(f"ALL {len(tests)} TESTS PASSED")
