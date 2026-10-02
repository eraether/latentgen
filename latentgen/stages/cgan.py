"""
Stage 3: conditional GAN, VQ codes -> AE latent.

Phase schedule (decided once per optimizer step):

* The **discriminator** trains while its accuracy on the last step's reals *and* fakes is below
  ``disc_accuracy_threshold`` (a logit beyond ``+-disc_logit_margin`` counts as a confident call).
* Once it is accurate enough the **generator** trains, with the discriminator frozen as a loss
  function. During generator steps we keep scoring the discriminator on the new fakes; when it
  can no longer tell them apart it loses its "good" status and gets to train again.

So the two networks take turns, each training only while it is behind. The generator keeps an EMA
copy of itself (``ema_decay`` / ``ema_every``) which is what inference samples from.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from latentgen.config import Config
from latentgen.data import EncodedDataset, ImageStats, waveform_image
from latentgen.device import autocast
from latentgen.nn import Autoencoder, Discriminator, Generator
from latentgen.training.images import image_grid
from latentgen.training.logging import MetricLogger
from latentgen.training.loop import Stage
from latentgen.training.manager import ModelManager


class DiscriminatorScore:
    """Counts confident-correct discriminator calls on the GPU; one sync in :meth:`accuracy`."""

    def __init__(self, margin: float, device: torch.device) -> None:
        self.margin = margin
        self._counts = torch.zeros(
            4, dtype=torch.long, device=device
        )  # correct_real, n_real, correct_fake, n_fake

    def real(self, logits: torch.Tensor) -> None:
        self._counts[0] += (logits > self.margin).sum()
        self._counts[1] += logits.numel()

    def fake(self, logits: torch.Tensor) -> None:
        self._counts[2] += (logits < -self.margin).sum()
        self._counts[3] += logits.numel()

    def reset(self) -> None:
        self._counts.zero_()

    def accuracy(self) -> tuple[float | None, float | None]:
        """``(real_acc, fake_acc)``, ``None`` for a side that saw no samples. Resets the counts (one sync)."""
        c = self._counts.tolist()
        self.reset()
        real = c[0] / c[1] if c[1] else None
        fake = c[2] / c[3] if c[3] else None
        return real, fake


class CGANStage(Stage):
    name = "cgan"

    def __init__(
        self,
        cfg: Config,
        device: torch.device,
        data: EncodedDataset,
        stats: ImageStats,
        autoencoder: Autoencoder | None = None,
    ) -> None:
        super().__init__(cfg, device, data, stats)
        self.autoencoder = autoencoder  # frozen; only used to decode progress images

        tc = self.train_cfg
        gen = Generator(cfg.cgan.generator, checkpointing=tc.activation_checkpointing)
        disc = Discriminator(cfg.cgan.discriminator, checkpointing=tc.activation_checkpointing)
        self.generator = ModelManager(
            "generator",
            gen,
            tc.optimizer,
            device,
            compile_model=cfg.project.compile,
            ema_decay=self.stage_cfg.ema_decay,
            ema_every=self.stage_cfg.ema_every,
        )
        self.discriminator = ModelManager(
            "discriminator", disc, tc.optimizer, device, compile_model=cfg.project.compile
        )
        self.managers = [self.generator, self.discriminator]

        self.score = DiscriminatorScore(self.stage_cfg.disc_logit_margin, device)
        self.disc_is_good = False  # True -> the generator trains this step
        self._phase: str | None = None  # "generator" / "discriminator", fixed for the whole optimizer step
        self._last: tuple[torch.Tensor, torch.Tensor] | None = (
            None  # (codes, real latents) for progress images
        )

    # ------------------------------------------------------------------ training

    def microbatch(self, batch, scale: float, logger: MetricLogger) -> None:
        codes, real = batch
        if self._phase is None:
            self._phase = "generator" if self.disc_is_good else "discriminator"
            logger.log("phase_generator", 1.0 if self._phase == "generator" else 0.0)
        self._last = (codes, real)

        with autocast(self.device):
            if self._phase == "generator":
                fake = self.generator(codes)
                # the discriminator is only a loss function here: no parameter grads, just input grads
                self.discriminator.model.requires_grad_(False)
                try:
                    fake_logits = self.discriminator(codes, fake).float()
                finally:
                    self.discriminator.model.requires_grad_(True)
                g_loss = F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))
                (g_loss * scale).backward()
                self.score.fake(fake_logits.detach())
                logger.log("generator_loss", g_loss)
            else:
                with torch.no_grad():
                    fake = self.generator(codes)
                # reals and fakes in one forward pass of 2B samples
                logits = self.discriminator(
                    torch.cat([codes, codes]), torch.cat([real, fake.to(real.dtype)])
                ).float()
                real_logits, fake_logits = logits.chunk(2)
                d_loss = F.binary_cross_entropy_with_logits(
                    real_logits, torch.ones_like(real_logits)
                ) + F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
                (d_loss * scale).backward()
                self.score.real(real_logits.detach())
                self.score.fake(fake_logits.detach())
                logger.log("discriminator_loss", d_loss)

    def step(self, logger: MetricLogger) -> None:
        manager = self.generator if self._phase == "generator" else self.discriminator
        ok, norm = manager.clip_and_step(self.train_cfg.grad_clip)
        if ok:
            logger.log(f"grad_norm_{manager.name}", norm)
        else:
            logger.count(f"skipped_steps_{manager.name}")
        real_acc, fake_acc = self.score.accuracy()
        threshold = self.stage_cfg.disc_accuracy_threshold
        # a side with no samples this step (reals during a generator step) counts as "accurate enough"
        self.disc_is_good = (real_acc is None or real_acc > threshold) and (
            fake_acc is None or fake_acc > threshold
        )
        if real_acc is not None:
            logger.log("disc_acc_real", real_acc)
        if fake_acc is not None:
            logger.log("disc_acc_fake", fake_acc)
        self._phase = None

    def discard_partial_step(self) -> None:
        # the phase was decided from micro-batches that will never be stepped on: decide afresh next time
        self._phase = None
        self.score.reset()

    # ------------------------------------------------------------------ reporting / state

    @torch.no_grad()
    def progress_image(self, batch) -> torch.Tensor | None:
        if self.autoencoder is None or self._last is None:
            return None
        codes, real = self._last
        n = min(2, codes.size(0))
        codes, real = codes[:n], real[:n]
        with autocast(self.device):
            fake = self.generator.model(codes)  # un-compiled module: the tiny batch would trigger a recompile
            fake_ema = self.generator.ema.model(codes)
            decoded = self.autoencoder.decode(torch.cat([real, fake.to(real.dtype), fake_ema.to(real.dtype)]))
        real_out, fake_out, ema_out = (self.stats.denormalize(x.float()) for x in decoded.split(n))
        if self.cfg.data.kind == "audio":  # real / generator / EMA waveforms stacked top to bottom
            return torch.cat([waveform_image(x[0]) for x in (real_out, fake_out, ema_out)], dim=-2)
        return image_grid([real_out, fake_out, ema_out], max_rows=n, downscale=2)  # real | generator | EMA

    def state_dict(self) -> dict:
        return {"disc_is_good": self.disc_is_good}

    def load_state_dict(self, state: dict) -> None:
        self.disc_is_good = bool(state.get("disc_is_good", False))
