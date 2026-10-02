"""
Stage 1b: continuous autoencoder.

Same encoder/decoder as the VQ-VAE but the bottleneck is a plain ``tanh`` projection, so the
latent is a ``[B, bottleneck_dim, H/p, W/p]`` tensor in ``[-1, 1]`` instead of integer codes.
It reconstructs far more detail than the VQ-VAE (its latent carries ~32x more information),
and it is what the stage-3 GAN learns to *produce* from the coarse VQ codes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from latentgen.config import AutoencoderModelConfig
from latentgen.nn.codec import PatchDecoder, PatchEncoder
from latentgen.nn.layers import RMSNorm2D


class TanhBottleneck(nn.Module):
    def __init__(self, cfg: AutoencoderModelConfig) -> None:
        super().__init__()
        self.norm = RMSNorm2D(cfg.hidden_size)
        self.down_proj_global = nn.Conv2d(cfg.hidden_size, cfg.bottleneck_dim, kernel_size=1, bias=False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.tanh(self.down_proj_global(self.norm(features)))


class Autoencoder(nn.Module):
    def __init__(self, cfg: AutoencoderModelConfig, checkpointing: bool = False) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = PatchEncoder(
            cfg.hidden_size, cfg.intermediate_size, cfg.num_encoder_layers, cfg.patch_size, checkpointing
        )
        self.bottleneck = TanhBottleneck(cfg)
        self.decoder = PatchDecoder(
            cfg.hidden_size,
            cfg.intermediate_size,
            cfg.num_decoder_layers,
            cfg.patch_size,
            cfg.bottleneck_dim,
            checkpointing,
            channels=cfg.channels,
            dims=cfg.dims,
        )

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """images -> (reconstruction, latents in [-1, 1])."""
        latents = self.bottleneck(self.encoder(images))
        return self.decoder(latents), latents

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.bottleneck(self.encoder(images))

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)
