"""
Grokfast-EMA gradient filter  (arXiv:2405.20233).

The mechanism, applied between loss.backward() and optimizer.step():

    ema[p] = alpha * ema[p] + (1 - alpha) * p.grad   # track slow-varying component
    p.grad = p.grad + lamb * ema[p]                   # amplify slow component

alpha=0.98 gives the EMA a half-life of ~34 steps, so it tracks gradients that
change slowly -- the signal the original paper associates with generalisation.
lamb=2.0 doubles the contribution of that slow component on each step.

The `emb_only` variant applies this only to the embedding and lm_head tensors.
Those two matrices are the interface between discrete token space and the
continuous representation: they see every token but update least uniformly,
making them a natural place for slow-component amplification to help.
"""

import torch
import torch.nn as nn


def apply_grokfast(
    model: nn.Module,
    ema_buffers: dict,
    alpha: float,
    lamb: float,
    emb_only: bool = False,
) -> None:
    """
    Modify gradients in-place with Grokfast-EMA.

    Args:
        model:       Model whose parameter gradients are modified.
        ema_buffers: Dict[id(param) -> Tensor] holding EMA state.
                     Pass an empty dict on the first call; updated in-place.
        alpha:       EMA decay (paper default 0.98).
        lamb:        Slow-component amplification (paper default 2.0).
        emb_only:    If True, only modify parameters whose name contains
                     'embed' or 'lm_head' (the embedding-only variant).
    """
    seen: set[int] = set()
    for name, p in model.named_parameters():
        pid = id(p)
        if p.grad is None or pid in seen:
            continue
        seen.add(pid)  # deduplicate tied weights (embed.weight == lm_head.weight)

        if emb_only and "embed" not in name and "lm_head" not in name:
            continue

        if pid not in ema_buffers:
            ema_buffers[pid] = torch.zeros_like(p.grad)

        # Update slow component via exponential moving average
        ema_buffers[pid].mul_(alpha).add_(p.grad, alpha=1.0 - alpha)
        # Amplify gradient: g_modified = g + lamb * ema
        p.grad.add_(ema_buffers[pid], alpha=lamb)
