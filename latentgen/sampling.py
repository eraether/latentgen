"""
Generating images: MaskGIT sampling, then decoding the code grid to pixels.

Sampling is deliberately simple -- one slot per forward pass, in random order, drawn straight
from the softmax (no temperature, no top-k, no confidence ordering):

1. **fill** a fully masked grid: pick a random still-masked slot, run the model, sample that slot,
   commit it. Do that for ``sample_fraction`` of the masked slots, then fill everything still
   masked with the argmax in one pass.
2. **refine**: for each ``keep`` in ``keep_schedule`` (0.01, 0.02, 0.04, ...), keep a random
   ``keep`` fraction of the finished grid, re-mask the rest and fill again. Early rounds throw
   almost everything away (global structure is re-drawn conditioned on a few anchors); later
   rounds keep most of it (local clean-up).

The final grid is decoded two ways: VQ-VAE decoder (blurry but faithful) and GAN generator +
AE decoder (sharp).
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F

from latentgen.device import autocast
from latentgen.nn import VQVAE, Autoencoder, Generator, MaskGIT


def fill_plan(num_positions: int, sample_fraction: float, keep_schedule) -> list[tuple[float, int, int]]:
    """``[(fraction kept going in, slots sampled one at a time, slots argmaxed), ...]`` per fill."""
    if not 0.0 <= sample_fraction <= 1.0:
        raise ValueError("sample_fraction must be in [0, 1]")
    if any(not 0.0 <= k < 1.0 for k in keep_schedule):
        raise ValueError("keep_schedule entries must be in [0, 1)")
    plan = []
    for keep in [0.0, *keep_schedule]:
        n_masked = num_positions - round(keep * num_positions)
        n_sample = round(sample_fraction * n_masked)
        plan.append((keep, n_sample, n_masked - n_sample))
    return plan


def forward_passes(plan) -> int:
    return sum(n_sample + (1 if n_argmax else 0) for _, n_sample, n_argmax in plan)


@torch.no_grad()
def fill(
    model: MaskGIT, codes: torch.Tensor, mask: torch.Tensor, n_sample: int, on_pass: Callable | None = None
) -> None:
    """Complete ``codes [B, S]`` in place where ``mask [B, S]`` is True (every row masks the same count)."""
    B, S = mask.shape
    H, W = model.cfg.grid_h, model.cfg.grid_w
    device = codes.device

    def logits_for(codes, mask):
        with autocast(device):
            return model(codes.view(B, H, W), mask.view(B, H, W)).reshape(B, S, -1).float()

    # random order over each row's masked slots: masked slots score in [0, 1), unmasked sort to the end
    order = torch.rand(B, S, device=device).masked_fill_(~mask, 2.0).argsort(dim=1)
    rows = torch.arange(B, device=device)
    for i in range(n_sample):
        pos = order[:, i]
        logits = logits_for(codes, mask)[rows, pos]
        codes[rows, pos] = torch.multinomial(F.softmax(logits, dim=-1), 1).squeeze(1)
        mask[rows, pos] = False
        if on_pass:
            on_pass()
    if mask.any():
        codes.copy_(torch.where(mask, logits_for(codes, mask).argmax(dim=-1), codes))
        mask.zero_()
        if on_pass:
            on_pass()


@torch.no_grad()
def generate_codes(
    model: MaskGIT, batch_size: int, sample_fraction: float, keep_schedule, on_pass: Callable | None = None
) -> torch.Tensor:
    """Sample ``[B, H, W]`` code grids from scratch (see module docstring)."""
    device = next(model.parameters()).device
    S = model.cfg.num_positions
    codes = torch.zeros(batch_size, S, dtype=torch.long, device=device)
    for keep, n_sample, _ in fill_plan(S, sample_fraction, keep_schedule):
        n_keep = round(keep * S)
        kept = torch.rand(batch_size, S, device=device).argsort(dim=1)[:, :n_keep]
        mask = torch.ones(batch_size, S, dtype=torch.bool, device=device).scatter_(1, kept, False)
        fill(model, codes, mask, n_sample, on_pass)
    return codes.view(batch_size, model.cfg.grid_h, model.cfg.grid_w)


@torch.no_grad()
def decode_vq(codes: torch.Tensor, vqvae: VQVAE, stats) -> torch.Tensor:
    """codes -> denormalised ``[B, C, H, W]`` images (or ``[B, 1, T]`` waveforms) through the VQ-VAE decoder."""
    with autocast(codes.device):
        images = vqvae.decode(codes)
    return stats.denormalize(images.float())


@torch.no_grad()
def decode_gan(codes: torch.Tensor, generator: Generator, autoencoder: Autoencoder, stats) -> torch.Tensor:
    """codes -> GAN latent -> AE decoder -> denormalised images / waveforms."""
    with autocast(codes.device):
        images = autoencoder.decode(generator(codes))
    return stats.denormalize(images.float())
