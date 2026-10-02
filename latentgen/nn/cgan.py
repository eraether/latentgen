"""
Stage 3: conditional GAN that turns coarse VQ codes into a detailed AE latent.

Both networks are transformers over the ``H x W`` token grid with 2D RoPE.

* :class:`Generator`: ``codes [B, H, W]`` (+ per-position Gaussian noise injected internally)
  -> ``latent [B, bottleneck_dim, H, W]`` in ``[-1, 1]`` (``tanh``, matching the AE).
* :class:`Discriminator`: ``(codes, latent)`` -> one real/fake logit per image (mean over positions).

At inference the generator's output is passed through the frozen stage-1b decoder to get pixels.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from latentgen.config import TransformerModelConfig
from latentgen.nn.layers import RoPE2D, run_layers, transformer_stack


class Generator(nn.Module):
    def __init__(self, cfg: TransformerModelConfig, checkpointing: bool = False) -> None:
        super().__init__()
        self.cfg = cfg
        self.checkpointing = checkpointing
        self.lowres_quantized_code_embedding = nn.Embedding(cfg.codebook_size, cfg.hidden_size)
        self.rope = RoPE2D(cfg.hidden_size // cfg.num_attention_heads, cfg.grid_h, cfg.grid_w, cfg.rope_base)
        self.layers = transformer_stack(
            cfg.num_layers, cfg.hidden_size, cfg.intermediate_size, cfg.num_attention_heads
        )
        self.noise_embedding = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        self.final_norm = nn.RMSNorm(cfg.hidden_size)
        self.final_out = nn.Linear(cfg.hidden_size, cfg.bottleneck_dim, bias=False)

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        """``codes: [B, H, W]`` long -> latent ``[B, bottleneck_dim, H, W]`` in ``[-1, 1]``."""
        B, H, W = codes.shape
        x = self.lowres_quantized_code_embedding(codes).reshape(B, H * W, -1)
        x = x + self.noise_embedding(torch.randn_like(x))  # the GAN's source of randomness
        cos, sin = self.rope()
        x = run_layers(self.layers, x, cos, sin, checkpointing=self.checkpointing)
        x = self.final_out(self.final_norm(x))  # [B, HW, bottleneck_dim]
        return torch.tanh(x.permute(0, 2, 1).reshape(B, -1, H, W))


class Discriminator(nn.Module):
    def __init__(self, cfg: TransformerModelConfig, checkpointing: bool = False) -> None:
        super().__init__()
        self.cfg = cfg
        self.checkpointing = checkpointing
        self.lowres_quantized_code_embedding = nn.Embedding(cfg.codebook_size, cfg.hidden_size)
        self.rope = RoPE2D(cfg.hidden_size // cfg.num_attention_heads, cfg.grid_h, cfg.grid_w, cfg.rope_base)
        self.highres_code_proj = nn.Linear(cfg.bottleneck_dim, cfg.hidden_size, bias=False)
        self.layers = transformer_stack(
            cfg.num_layers, cfg.hidden_size, cfg.intermediate_size, cfg.num_attention_heads
        )
        self.final_norm = nn.RMSNorm(cfg.hidden_size)
        self.final_out = nn.Linear(cfg.hidden_size, 1, bias=False)

    def forward(self, codes: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        """``codes: [B, H, W]`` long, ``latents: [B, bottleneck_dim, H, W]`` -> logits ``[B, 1]``."""
        B, H, W = codes.shape
        x = self.lowres_quantized_code_embedding(codes).reshape(B, H * W, -1)
        x = x + self.highres_code_proj(latents.permute(0, 2, 3, 1).reshape(B, H * W, -1))
        cos, sin = self.rope()
        x = run_layers(self.layers, x, cos, sin, checkpointing=self.checkpointing)
        return self.final_out(self.final_norm(x)).mean(dim=1)
