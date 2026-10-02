"""
Stage 2: MaskGIT -- a bidirectional transformer over the 2D grid of VQ codes.

Training: a random fraction of the grid is replaced by a learned ``[MASK]`` embedding and the
model predicts the original code at every masked position (cross-entropy on masked positions
only). Sampling (``latentgen/sampling.py``) starts from an all-masked grid and fills it in.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from latentgen.config import TransformerModelConfig
from latentgen.nn.layers import RoPE2D, run_layers, transformer_stack


class MaskGIT(nn.Module):
    def __init__(self, cfg: TransformerModelConfig, checkpointing: bool = False) -> None:
        super().__init__()
        self.cfg = cfg
        self.checkpointing = checkpointing
        self.code_embedding = nn.Embedding(cfg.codebook_size, cfg.hidden_size)
        # masked positions are *replaced* by this vector (not added to it), so the true code never leaks
        self.mask_token = nn.Parameter(torch.randn(cfg.hidden_size))
        self.rope = RoPE2D(cfg.hidden_size // cfg.num_attention_heads, cfg.grid_h, cfg.grid_w, cfg.rope_base)
        self.layers = transformer_stack(
            cfg.num_layers, cfg.hidden_size, cfg.intermediate_size, cfg.num_attention_heads
        )
        self.final_norm = nn.RMSNorm(cfg.hidden_size)
        self.final_out = nn.Linear(cfg.hidden_size, cfg.codebook_size, bias=False)

    def forward(self, codes: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """``codes: [B, H, W]`` long (values under the mask are ignored), ``mask: [B, H, W]`` bool (True = masked).

        Returns logits ``[B, H, W, codebook_size]``.
        """
        B, H, W = codes.shape
        x = self.code_embedding(codes.reshape(B, H * W))
        x = torch.where(mask.reshape(B, H * W, 1), self.mask_token.to(x.dtype), x)
        cos, sin = self.rope()
        x = run_layers(self.layers, x, cos, sin, checkpointing=self.checkpointing)
        logits = self.final_out(self.final_norm(x))
        return logits.reshape(B, H, W, -1)


def masked_cross_entropy(logits: torch.Tensor, codes: torch.Tensor, mask: torch.Tensor):
    """Cross-entropy and accuracy over masked positions only.

    Returns ``(loss, ce_sum [B], correct_sum [B], n_masked [B])`` where ``loss`` is the mean CE
    over every masked token in the batch (what we backpropagate) and the per-sample sums let the
    trainer report CE / accuracy bucketed by mask ratio.
    """
    B = logits.size(0)
    V = logits.size(-1)
    logits = logits.float().reshape(-1, V)
    targets = codes.reshape(-1)
    m = mask.reshape(B, -1).float()
    ce = F.cross_entropy(logits, targets, reduction="none").view(B, -1)
    correct = (logits.argmax(-1) == targets).view(B, -1).float()
    ce_sum, correct_sum, n_masked = (ce * m).sum(1), (correct * m).sum(1), m.sum(1)
    loss = ce_sum.sum() / n_masked.sum().clamp_min(1.0)
    return loss, ce_sum, correct_sum, n_masked


def random_mask(
    batch_size: int, grid_h: int, grid_w: int, ratio_min: float, ratio_max: float, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-sample random masks. Each row masks exactly ``ceil(r * H*W)`` positions, ``r ~ U[min, max]``.

    Returns ``(mask [B, H, W] bool, ratio [B] float)`` where ``ratio`` is the exact fraction masked.
    """
    S = grid_h * grid_w
    r = ratio_min + (ratio_max - ratio_min) * torch.rand(batch_size, device=device)
    k = (r * S).ceil().long().clamp_(1, S)
    scores = torch.rand(batch_size, S, device=device)
    kth = scores.sort(dim=1).values.gather(1, (k - 1).unsqueeze(1))  # k-th smallest score per row
    mask = (scores <= kth).view(batch_size, grid_h, grid_w)
    return mask, k.float() / S
