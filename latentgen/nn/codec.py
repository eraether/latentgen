"""
Patch encoder / decoder shared by the VQ-VAE (stage 1a) and the autoencoder (stage 1b).

Both codecs are "everything is an MLP": the input is cut into patches, each patch is projected to
``hidden_size`` channels, and a stack of per-token MLP blocks refines every token *independently*.
The only spatial mixing in the whole codec is one 3x3 convolution at the start of the decoder
(``receptive_field_expansion``).

Images (``dims=2``):   [B, C, H, W]  --unpatchify-->  [B, C*p*p, H/p, W/p]  --1x1 conv-->  [B, hidden, H/p, W/p]
Audio  (``dims=1``):   [B, C, T]     --unpatchify-->  [B, C*p, 1, T/p]      --1x1 conv-->  [B, hidden, 1, T/p]

A 1-D signal is carried through the pipeline as a grid with height 1, so the quantizer, the
MaskGIT transformer and the GAN work unchanged (``grid_h = 1``).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from latentgen.nn.layers import mlp2d_stack, run_layers


def patchify(x: torch.Tensor, patch_size: int, dims: int) -> torch.Tensor:
    """Fold ``patch_size`` neighboring samples/pixels into the channel dim (see module docstring)."""
    p = patch_size
    if dims == 2:
        return nn.functional.pixel_unshuffle(x, p)
    B, C, T = x.shape
    if T % p:
        raise ValueError(f"signal length {T} must be a multiple of patch_size {p}")
    return x.view(B, C, T // p, p).permute(0, 1, 3, 2).reshape(B, C * p, 1, T // p)


def unpatchify(x: torch.Tensor, patch_size: int, dims: int) -> torch.Tensor:
    """Inverse of :func:`patchify`."""
    p = patch_size
    if dims == 2:
        return nn.functional.pixel_shuffle(x, p)
    B, Cp, _one, Tp = x.shape
    return x.view(B, Cp // p, p, Tp).permute(0, 1, 3, 2).reshape(B, Cp // p, Tp * p)


class PatchEncoder(nn.Module):
    """``[B, C, H, W]`` (or ``[B, C, T]``) -> ``[B, hidden_size, H/p, W/p]`` (or ``[B, hidden_size, 1, T/p]``)."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_layers: int,
        patch_size: int,
        checkpointing: bool = False,
        channels: int = 3,
        dims: int = 2,
    ) -> None:
        super().__init__()
        self.checkpointing = checkpointing
        self.patch_size, self.dims = patch_size, dims
        patch_dim = channels * patch_size**dims
        self.patch_proj = nn.Conv2d(patch_dim, hidden_size, kernel_size=1, bias=False)
        self.layers = mlp2d_stack(num_layers, hidden_size, intermediate_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_proj(patchify(x, self.patch_size, self.dims))
        return run_layers(self.layers, x, checkpointing=self.checkpointing)


class PatchDecoder(nn.Module):
    """``[B, bottleneck_dim, H/p, W/p]`` latent -> ``[B, C, H, W]`` (or ``[B, C, T]`` for ``dims=1``)."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_layers: int,
        patch_size: int,
        bottleneck_dim: int,
        checkpointing: bool = False,
        channels: int = 3,
        dims: int = 2,
    ) -> None:
        super().__init__()
        self.checkpointing = checkpointing
        self.patch_size, self.dims = patch_size, dims
        # NOTE: registration order matters beyond names -- optimizer state is matched to parameters by
        # position, so reordering these lines would break resuming from existing checkpoints.
        self.layers = mlp2d_stack(num_layers, hidden_size, intermediate_size)
        self.output_proj = nn.Conv2d(hidden_size, channels * patch_size**dims, kernel_size=1, bias=False)
        # the single spatial-mixing op of the codec: lets each output patch see its neighbors
        self.receptive_field_expansion = nn.Conv2d(
            bottleneck_dim, hidden_size, kernel_size=3, padding=1, bias=False
        )

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        x = self.receptive_field_expansion(latents)
        x = run_layers(self.layers, x, checkpointing=self.checkpointing)
        return unpatchify(self.output_proj(x), self.patch_size, self.dims)
