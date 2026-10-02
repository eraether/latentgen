"""
Stage 1a: VQ-VAE with a *hypernetwork codebook*.

    image --PatchEncoder--> features --VectorQuantizer--> (codes, quantized latents) --PatchDecoder--> image

The twist is how the codebook is produced. Instead of a plain learned ``[K, D]`` table, a small
transformer (:class:`HypernetworkCodebook`) maps ``K`` fresh Gaussian vectors to ``K`` codebook
entries on **every forward pass**. Early in training this acts like a very strong regularizer:
the codebook keeps moving, no entry can die, and the encoder is forced to be robust. Once the
reconstructions are good (``cutover_step`` in the config), :meth:`HypernetworkCodebook.cutover`
samples one codebook, stores it in ``learned_codebook`` and from then on the model behaves like a
normal VQ-VAE with a learnable table (the hypernetwork weights are kept but unused).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from latentgen.config import VQVAEModelConfig
from latentgen.nn.codec import PatchDecoder, PatchEncoder
from latentgen.nn.layers import RMSNorm2D, run_layers, transformer_stack


class HypernetworkCodebook(nn.Module):
    """Produces the ``[codebook_size, bottleneck_dim]`` codebook (see module docstring)."""

    def __init__(self, cfg: VQVAEModelConfig, checkpointing: bool = False) -> None:
        super().__init__()
        self.cfg = cfg
        self.checkpointing = checkpointing
        self.layers = transformer_stack(
            cfg.num_codebook_layers, cfg.hidden_size, cfg.intermediate_size, cfg.num_attention_heads
        )
        self.final_norm = nn.RMSNorm(cfg.hidden_size)
        self.final_out = nn.Linear(cfg.hidden_size, cfg.bottleneck_dim, bias=False)
        # after cutover the hypernetwork is bypassed and this table is used (and trained) directly.
        # `cutover` is the persistent flag (saved in checkpoints); `is_cut_over` mirrors it as a plain
        # Python bool so the branch in forward() never forces a GPU sync or a torch.compile graph break.
        self.register_buffer("cutover", torch.tensor(False))
        self.learned_codebook = nn.Parameter(torch.zeros(cfg.codebook_size, cfg.bottleneck_dim))
        self.is_cut_over = False
        self.register_load_state_dict_post_hook(lambda module, _incompatible: module._sync_flag())

    def _sync_flag(self) -> None:
        self.is_cut_over = bool(self.cutover.item())

    def forward(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.is_cut_over:
            return self.learned_codebook
        x = torch.randn((1, self.cfg.codebook_size, self.cfg.hidden_size), device=device, dtype=dtype)
        x = run_layers(self.layers, x, checkpointing=self.checkpointing)
        return self.final_out(self.final_norm(x)).squeeze(0)

    @torch.no_grad()
    def cutover_to_learned_codebook(self) -> None:
        """Freeze one sampled codebook into ``learned_codebook`` and stop using the hypernetwork."""
        if self.is_cut_over:
            return
        device = self.learned_codebook.device
        self.learned_codebook.copy_(self.forward(device, torch.float32))
        self.cutover.fill_(True)
        self.is_cut_over = True


class VectorQuantizer(nn.Module):
    """Projects features to ``bottleneck_dim``, snaps each position to its nearest codebook entry.

    Returns the straight-through quantized latent (gradients flow to the encoder as if no
    quantization happened), the integer codes, and the commitment loss that pulls encoder
    outputs toward the entries they were assigned to.
    """

    def __init__(self, cfg: VQVAEModelConfig, checkpointing: bool = False) -> None:
        super().__init__()
        self.norm = RMSNorm2D(cfg.hidden_size)
        self.down_proj_global = nn.Conv2d(cfg.hidden_size, cfg.bottleneck_dim, kernel_size=1, bias=False)
        self.codebook = HypernetworkCodebook(cfg, checkpointing)

    def current_codebook(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return self.codebook(device, dtype)

    @staticmethod
    def nearest_codes(z: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
        """``z: [B, D, H, W]``, ``codebook: [K, D]`` -> ``[B, H, W]`` long. Done in fp32 without materializing [B, K, D, H, W]."""
        B, D, H, W = z.shape
        with torch.autocast(device_type=z.device.type, enabled=False):  # autocast would make the matmul bf16
            flat = z.permute(0, 2, 3, 1).reshape(-1, D).float()  # [B*H*W, D]
            cb = codebook.float()
            # ||z - e||^2 = ||z||^2 - 2 z.e + ||e||^2 ; ||z||^2 is constant per row so it is dropped
            dist = cb.pow(2).sum(1)[None, :] - 2.0 * flat @ cb.t()
            return dist.argmin(dim=1).view(B, H, W)

    @staticmethod
    def lookup(codes: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
        """``codes: [B, H, W]`` long, ``codebook: [K, D]`` -> ``[B, D, H, W]``."""
        return codebook[codes].permute(0, 3, 1, 2)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.down_proj_global(self.norm(features))  # [B, D, H, W]
        codebook = self.current_codebook(z.device, z.dtype)
        codes = self.nearest_codes(z, codebook)
        quantized = self.lookup(codes, codebook)
        commitment_loss = (z.float() - quantized.float()).pow(2).mean()
        z_straight_through = z + (quantized.to(z.dtype) - z).detach()
        return z_straight_through, codes, commitment_loss


class VQVAE(nn.Module):
    """Image <-> discrete code grid. See module docstring."""

    def __init__(self, cfg: VQVAEModelConfig, checkpointing: bool = False) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = PatchEncoder(
            cfg.hidden_size, cfg.intermediate_size, cfg.num_encoder_layers, cfg.patch_size, checkpointing
        )
        self.bottleneck = VectorQuantizer(cfg, checkpointing)
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

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """images -> (reconstruction, codes [B, H/p, W/p], commitment_loss)."""
        latents, codes, commitment_loss = self.bottleneck(self.encoder(images))
        return self.decoder(latents), codes, commitment_loss

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """images -> integer codes ``[B, H/p, W/p]``."""
        return self.bottleneck(self.encoder(images))[1]

    @torch.no_grad()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """integer codes ``[B, H/p, W/p]`` -> images (still in normalized space)."""
        codebook = self.bottleneck.current_codebook(codes.device, torch.float32)
        return self.decoder(self.bottleneck.lookup(codes, codebook))
