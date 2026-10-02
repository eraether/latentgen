"""Reconstruction losses and PSNR used by stages 1a and 1b."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from latentgen.data.normalization import ImageStats


def reconstruction_loss(pred: torch.Tensor, target: torch.Tensor, scale: float = 10.0) -> torch.Tensor:
    """Smooth-L1 (Huber) on ``scale``-multiplied normalized images.

    With ``beta=1`` the loss is quadratic for errors below ``1/scale`` and linear above, so large
    errors are not dominated by outliers while small ones still get an MSE-like gradient. Scaling
    the inputs rather than ``beta`` keeps the gradient magnitude continuous across the switch.
    """
    return F.smooth_l1_loss(pred * scale, target * scale, beta=1.0)


@torch.no_grad()
def psnr(pred: torch.Tensor, target: torch.Tensor, stats: ImageStats) -> torch.Tensor:
    """Mean PSNR (dB) over the batch, measured on denormalized ``[0, 1]`` images (or ``[-1, 1]`` audio)."""
    mse = F.mse_loss(stats.denormalize(pred.float()), stats.denormalize(target.float()), reduction="none")
    mse = mse.flatten(1).mean(dim=1).clamp_min(1e-10)  # per item; works for images and waveforms
    return (10 * torch.log10(1.0 / mse)).mean()


def stft_loss(
    pred: torch.Tensor, target: torch.Tensor, fft_sizes: tuple[int, ...] = (256, 512, 1024, 2048)
) -> torch.Tensor:
    """Multi-resolution STFT loss for waveforms ``[B, 1, T]``: spectral convergence + log-magnitude L1.

    A sample-wise loss alone lets a codec trade high-frequency detail for a small error, which
    sounds muffled; matching magnitude spectra at several resolutions keeps the timbre and the
    transients. Computed in fp32 (autocast off), averaged over the resolutions.
    """
    with torch.autocast(device_type=pred.device.type, enabled=False):
        p, t = pred.float().flatten(1), target.float().flatten(1)
        total = p.new_zeros(())
        fft_sizes = [n for n in fft_sizes if n < p.shape[-1]] or [
            p.shape[-1] // 2
        ]  # short clips: fewer bands
        for n_fft in fft_sizes:
            window = torch.hann_window(n_fft, device=p.device)
            spec = {
                k: torch.stft(x, n_fft, hop_length=n_fft // 4, window=window, return_complex=True).abs()
                for k, x in (("p", p), ("t", t))
            }
            convergence = (spec["t"] - spec["p"]).norm() / spec["t"].norm().clamp_min(1e-7)
            log_mag = F.l1_loss(spec["p"].clamp_min(1e-5).log(), spec["t"].clamp_min(1e-5).log())
            total = total + convergence + log_mag
        return total / len(fft_sizes)
