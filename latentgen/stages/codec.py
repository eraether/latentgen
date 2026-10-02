"""
Stages 1a (VQ-VAE) and 1b (autoencoder): image reconstruction training.

Both stages are the same loop -- encode, decode, smooth-L1 against the input, report PSNR --
so they share :class:`ImageCodecStage` and only differ in the model and the extra VQ losses.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from latentgen.config import Config, replace
from latentgen.data import ImageBatches, ImageStats, audio_panel
from latentgen.device import autocast
from latentgen.nn import VQVAE, Autoencoder
from latentgen.training.images import image_grid
from latentgen.training.logging import MetricLogger
from latentgen.training.loop import Stage
from latentgen.training.losses import psnr, reconstruction_loss, stft_loss
from latentgen.training.manager import ModelManager


class ImageCodecStage(Stage):
    """Shared by stages 1a and 1b. Subclasses set ``name``, ``model_cls`` and ``forward_and_loss``."""

    model_cls: type[nn.Module]

    def __init__(self, cfg: Config, device: torch.device, data: ImageBatches, stats: ImageStats) -> None:
        super().__init__(cfg, device, data, stats)
        # the codec's input shape follows the data kind; resolved here so the checkpoint stores it
        audio = cfg.data.kind == "audio"
        self.stage_cfg.model = replace(
            self.stage_cfg.model, channels=1 if audio else 3, dims=1 if audio else 2
        )
        length = cfg.data.audio_length if audio else cfg.data.image_size
        if length % self.stage_cfg.model.patch_size:
            raise SystemExit(
                f"data.{'audio_length' if audio else 'image_size'}={length} must be a multiple of "
                f"{self.name}.model.patch_size={self.stage_cfg.model.patch_size}"
            )
        model = self.model_cls(self.stage_cfg.model, checkpointing=self.train_cfg.activation_checkpointing)
        self.managers = [
            ModelManager(
                self.name, model, self.train_cfg.optimizer, device, compile_model=cfg.project.compile
            )
        ]
        self._last: tuple[torch.Tensor, torch.Tensor] | None = None  # (inputs, reconstructions) for images
        self.stft_weight = self.stage_cfg.stft_loss_weight if audio else 0.0

    @property
    def model(self) -> nn.Module:
        return self.manager.model

    def forward_and_loss(
        self, images: torch.Tensor, logger: MetricLogger
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(reconstruction, total loss)``; subclasses add their own terms and logs."""
        raise NotImplementedError

    def microbatch(self, batch, scale: float, logger: MetricLogger) -> None:
        images, _ids = batch
        with autocast(self.device):
            recon, loss = self.forward_and_loss(images, logger)
        if self.stft_weight:
            spectral = stft_loss(recon, images)
            logger.log("stft", spectral)
            loss = loss + self.stft_weight * spectral
        (loss * scale).backward()
        logger.log("loss", loss)
        logger.log("psnr_db", psnr(recon, images, self.stats))
        self._last = (images, recon.detach())

    def progress_image(self, batch) -> torch.Tensor | None:
        if self._last is None:
            return None
        inputs, recon = self._last
        inputs, recon = self.stats.denormalize(inputs.float()), self.stats.denormalize(recon.float())
        if self.cfg.data.kind == "audio":  # input on top, reconstruction below (waveform + spectrogram each)
            return audio_panel([inputs[0], recon[0]])
        return image_grid([inputs, recon], max_rows=2)


class VQVAEStage(ImageCodecStage):
    """Stage 1a. Loss = smooth-L1 reconstruction + commitment. Handles the codebook cutover."""

    name = "vqvae"
    model_cls = VQVAE

    def forward_and_loss(self, images, logger):
        recon, codes, commitment = self.manager(images)
        rec = reconstruction_loss(recon, images, self.stage_cfg.loss_scale)
        logger.log("reconstruction", rec)
        logger.log("commitment", commitment)
        K = self.model.cfg.codebook_size
        # fraction of the codebook used in this micro-batch; scatter_ keeps it on the GPU (bincount would sync)
        used = torch.zeros(K, device=codes.device).scatter_(0, codes.flatten(), 1.0).sum()
        logger.log("codebook_usage", used / K)
        return recon, rec + self.stage_cfg.commitment_weight * commitment

    def on_step(self, step: int) -> None:
        codebook = self.model.bottleneck.codebook
        if not codebook.is_cut_over and step >= self.stage_cfg.cutover_step:
            self.cutover()

    def cutover(self) -> None:
        """Freeze the hypernetwork codebook into a learned table and restart the optimizer."""
        self.model.bottleneck.codebook.cutover_to_learned_codebook()
        self.manager.reset_optimizer()
        print(
            "Codebook cutover: the hypernetwork is now bypassed; training continues on a fixed, learnable codebook"
        )


class AutoencoderStage(ImageCodecStage):
    """Stage 1b. Loss = smooth-L1 reconstruction only."""

    name = "autoencoder"
    model_cls = Autoencoder

    def forward_and_loss(self, images, logger):
        recon, _latents = self.manager(images)
        return recon, reconstruction_loss(recon, images, self.stage_cfg.loss_scale)
