"""Reconstruction loss and PSNR used by stages 1a and 1b."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from latentgen.data.normalization import ImageStats


def reconstruction_loss(pred: torch.Tensor, target: torch.Tensor, scale: float = 10.0) -> torch.Tensor:
    """Smooth-L1 (Huber) on ``scale``-multiplied normalised images.

    With ``beta=1`` the loss is quadratic for errors below ``1/scale`` and linear above, so large
    errors are not dominated by outliers while small ones still get an MSE-like gradient. Scaling
    the inputs rather than ``beta`` keeps the gradient magnitude continuous across the switch.
    """
    return F.smooth_l1_loss(pred * scale, target * scale, beta=1.0)


@torch.no_grad()
def psnr(pred: torch.Tensor, target: torch.Tensor, stats: ImageStats) -> torch.Tensor:
    """Mean PSNR (dB) over the batch, measured on ``[0, 1]`` images."""
    mse = F.mse_loss(stats.denormalize(pred.float()), stats.denormalize(target.float()), reduction="none")
    mse = mse.mean(dim=(1, 2, 3)).clamp_min(1e-10)
    return (10 * torch.log10(1.0 / mse)).mean()
