"""
LLaMA-style causal decoder: RMSNorm + RoPE + SwiGLU + tied embeddings.

The architecture is intentionally vanilla -- no GQA, no sliding window, no
mixture-of-experts. The experiment question is about the optimizer, not the
architecture, so we want the simplest model that is representative of modern
LLM training practice.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Protocol


class ModelCfg(Protocol):
    """Minimal interface the model reads from a config object."""
    n_layers:   int
    d_model:    int
    n_heads:    int
    d_ff:       int
    seq_len:    int
    vocab_size: int


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cast to float32 for numerical stability; return in input dtype.
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm * self.weight).to(x.dtype)


def build_rope_cache(
    seq_len: int, head_dim: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-compute cos/sin tables for RoPE. Returns (cos, sin) each (T, head_dim)."""
    inv_freq   = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    pos        = torch.arange(seq_len, device=device).float()
    freqs      = torch.outer(pos, inv_freq)              # (T, head_dim/2)
    freqs_full = torch.cat([freqs, freqs], dim=-1)       # (T, head_dim)
    return freqs_full.cos(), freqs_full.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, H, T, head_dim)"""
    cos = cos[: x.shape[2]].unsqueeze(0).unsqueeze(0)
    sin = sin[: x.shape[2]].unsqueeze(0).unsqueeze(0)
    return x * cos + rotate_half(x) * sin


# ---------------------------------------------------------------------------
# Attention + MLP
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelCfg):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads  = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        out  = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(out.transpose(1, 2).contiguous().view(B, T, C))


class SwiGLUMLP(nn.Module):
    def __init__(self, cfg: ModelCfg):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up_proj   = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down_proj = nn.Linear(cfg.d_ff,    cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelCfg):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn      = CausalSelfAttention(cfg)
        self.mlp_norm  = RMSNorm(cfg.d_model)
        self.mlp       = SwiGLUMLP(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.mlp(self.mlp_norm(x))
        return x


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class LlamaDecoder(nn.Module):
    """
    LLaMA-style causal decoder with tied embedding/lm_head weights.

    Stage 1 default: 8L/512H/8A/1408FF  -> ~51.4M unique params
    Stage 2 default: 12L/1024H/16A/2816FF -> ~205.6M unique params
    """
    def __init__(self, cfg: ModelCfg):
        super().__init__()
        self.cfg     = cfg
        self.embed   = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers  = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm    = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight  # weight tying

        self._rope_len: int = 0
        self._cos: torch.Tensor | None = None
        self._sin: torch.Tensor | None = None

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def _get_rope(self, seq_len: int, device: torch.device):
        if seq_len > self._rope_len or self._cos is None:
            max_len = max(seq_len, self.cfg.seq_len)
            self._cos, self._sin = build_rope_cache(
                max_len, self.cfg.d_model // self.cfg.n_heads, device
            )
            self._rope_len = max_len
        return self._cos, self._sin

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        cos, sin = self._get_rope(T, idx.device)
        x = self.embed(idx)
        for layer in self.layers:
            x = layer(x, cos, sin)
        return self.lm_head(self.norm(x))

    def num_params(self) -> int:
        """Count unique parameters (deduplicates tied embed/lm_head)."""
        seen: set[int] = set()
        total = 0
        for p in self.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                total += p.numel()
        return total
