"""
The training loop every stage shares.

A *stage* (``latentgen/stages/*.py``) only defines what happens to one micro-batch and what happens
at an optimizer step; :class:`Trainer` does everything else:

* gradient accumulation (``batch_size / microbatch_size`` micro-batches per step)
* epoch bookkeeping that resumes mid-epoch
* periodic logging (console + TensorBoard), progress images and checkpoints
* Ctrl-C -> save a checkpoint and exit cleanly (``--pdb`` drops into a debugger instead)

Timeline of one optimizer step::

    for each micro-batch:   stage.microbatch(batch, scale=1/microbatches_per_step)   # forward + backward
    then once:              stage.step()                                            # clip + optimizer.step()
"""

from __future__ import annotations

import pdb
import signal
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from latentgen.config import Config, TrainConfig, save_config, to_dict
from latentgen.data.normalization import ImageStats
from latentgen.training.checkpoint import load_checkpoint, save_checkpoint
from latentgen.training.images import BackgroundImageSaver
from latentgen.training.logging import MetricLogger
from latentgen.training.manager import ModelManager


class Stage(ABC):
    """What a training stage must provide. See ``latentgen/stages/codec.py`` for the simplest example."""

    name: str  # "vqvae", "autoencoder", "maskgit", "cgan"
    train_cfg: TrainConfig
    managers: list[ModelManager]

    def __init__(self, cfg: Config, device: torch.device, data: Any, stats: ImageStats) -> None:
        self.cfg = cfg
        self.device = device
        self.data = data  # has .tracker, .batches(batch_size), .batches_per_epoch(batch_size), .num_items
        self.stats = stats
        self.stage_cfg = getattr(cfg, self.name)  # cfg.vqvae / cfg.autoencoder / cfg.maskgit / cfg.cgan
        self.train_cfg = self.stage_cfg.train

    @property
    def manager(self) -> ModelManager:
        """The one model of a single-model stage."""
        (manager,) = self.managers
        return manager

    @abstractmethod
    def microbatch(self, batch: Any, scale: float, logger: MetricLogger) -> None:
        """Forward + ``(loss * scale).backward()`` for one micro-batch. Log scalars via ``logger.log``."""

    def step(self, logger: MetricLogger) -> None:
        """Optimizer step(s) after the accumulated micro-batches. Default: clip + step the single manager."""
        ok, norm = self.manager.clip_and_step(self.train_cfg.grad_clip)
        if ok:
            logger.log("grad_norm", norm)
        else:
            logger.count("skipped_steps_nonfinite_grad")

    def discard_partial_step(self) -> None:
        """Called when accumulated micro-batches are thrown away (epoch end, Ctrl-C). Reset per-step state here."""

    def progress_image(self, batch: Any) -> torch.Tensor | None:
        """Optional ``[3, H, W]`` image in ``[0, 1]`` to save every ``image_every`` steps."""
        return None

    def on_log(self, logger: MetricLogger, step: int) -> None:
        """Optional extra reporting at log time (e.g. tables)."""

    def on_step(self, step: int) -> None:
        """Optional hook after every optimizer step (e.g. scheduled events like the codebook cutover)."""

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: dict) -> None:
        pass


class Trainer:
    def __init__(
        self,
        cfg: Config,
        stage: Stage,
        run_dir: str | Path,
        device: torch.device,
        pdb_on_interrupt: bool = False,
    ) -> None:
        self.cfg = cfg
        train_cfg = self.train_cfg = stage.train_cfg
        self.stage = stage
        self.device = device
        self.run_dir = Path(run_dir)
        self.pdb_on_interrupt = pdb_on_interrupt

        if train_cfg.log_every <= 0:
            raise ValueError("train.log_every must be >= 1 (metrics are only flushed at log time)")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        save_config(cfg, self.run_dir / "config.yaml")
        self.logger = MetricLogger(self.run_dir / "tensorboard")
        self.images = BackgroundImageSaver(self.run_dir / "images")

        self.epoch = 0
        self.step = 0  # optimizer steps
        self.microbatches = 0
        self.last_save_time = time.time()
        self._interrupted = False
        self._pdb_requested = False

    # ----------------------------------------------------------------------------- checkpoints

    def save(self, tag: str | None = None) -> Path:
        path = self.run_dir / "checkpoints" / f"step_{self.step:08d}.pt"
        payload = {
            "stage": self.stage.name,
            "config": to_dict(self.cfg),
            "stats": self.stage.stats.to_dict(),
            "models": {m.name: m.state_dict() for m in self.stage.managers},
            "progress": {
                "epoch": self.epoch,
                "step": self.step,
                "microbatches": self.microbatches,
                "seen_ids": self.stage.data.tracker.seen_ids(),
            },
            "stage_state": self.stage.state_dict(),
        }
        save_checkpoint(path, payload)
        print(f"Saved checkpoint {path}" + (f" ({tag})" if tag else ""))
        self.last_save_time = time.time()
        return path

    def resume(self, checkpoint: str | Path) -> None:
        ckpt = load_checkpoint(checkpoint, self.device)
        if ckpt["stage"] != self.stage.name:
            raise ValueError(
                f"{ckpt['path']} is a '{ckpt['stage']}' checkpoint, this is the '{self.stage.name}' stage"
            )
        for m in self.stage.managers:
            if m.name not in ckpt["models"]:
                raise KeyError(f"checkpoint has no model named '{m.name}' (has {list(ckpt['models'])})")
            m.load_state_dict(ckpt["models"][m.name])
        progress = ckpt["progress"]
        self.epoch, self.step, self.microbatches = (
            progress["epoch"],
            progress["step"],
            progress["microbatches"],
        )
        self.stage.data.tracker.set_seen(progress.get("seen_ids", []))
        self.stage.load_state_dict(ckpt.get("stage_state", {}))
        print(
            f"Resumed {ckpt['path']}: epoch {self.epoch}, step {self.step}, "
            f"{self.stage.data.tracker.num_seen()}/{self.stage.data.num_items} images seen this epoch"
        )

    # ----------------------------------------------------------------------------- the loop

    def _handle_interrupt(self, signum, frame) -> None:
        if self._interrupted or self._pdb_requested:  # second Ctrl-C: exit immediately
            raise KeyboardInterrupt
        if self.pdb_on_interrupt:
            self._pdb_requested = True
            print("\nCtrl-C: entering pdb at the next step boundary (press again to abort)")
            return
        self._interrupted = True
        print("\nCtrl-C: finishing the current step, then saving and exiting (press again to abort)")

    def _enter_pdb(self) -> None:
        self._pdb_requested = False
        print(
            "pdb: `self` is the trainer, `self.stage` the stage. `self.save()` writes a checkpoint, "
            "`c` continues training, `self._interrupted = True` then `c` saves and exits."
        )
        pdb.set_trace()
        self.stage.discard_partial_step()

    def run(self) -> None:
        tc = self.train_cfg
        per_step = tc.microbatches_per_step
        stage = self.stage
        signal.signal(signal.SIGINT, self._handle_interrupt)

        print(f"Run dir: {self.run_dir}")
        print(
            f"{stage.data.num_items} items/epoch, {tc.microbatch_size} per micro-batch x {per_step} = "
            f"{tc.batch_size} per optimizer step"
        )
        for m in stage.managers:
            m.zero_grad()

        try:
            while self.epoch < tc.epochs and not self._interrupted:
                self._run_epoch()
        finally:
            self.images.close()
            self.logger.close()

        if self._interrupted:
            self.save("interrupted")
            print("Exiting. Resume with --resume", self.run_dir)
        else:
            self.save("final")
            print("Training finished.")

    def _run_epoch(self) -> None:
        tc = self.train_cfg
        per_step = tc.microbatches_per_step
        stage = self.stage
        total = stage.data.batches_per_epoch(tc.microbatch_size)
        done = int(stage.data.tracker.fraction_seen * total)
        pbar = tqdm(total=total, initial=done, desc=f"epoch {self.epoch}", unit="mb", dynamic_ncols=True)
        since_step = 0
        step_t0 = time.perf_counter()
        last_batch = None

        for batch in stage.data.batches(tc.microbatch_size):
            last_batch = batch
            stage.microbatch(batch, 1.0 / per_step, self.logger)
            self.microbatches += 1
            since_step += 1
            pbar.update(1)
            if since_step < per_step:
                continue

            # ------------------------------------------------------------ optimizer step boundary
            since_step = 0
            self.step += 1
            stage.step(self.logger)
            stage.on_step(self.step)
            now = time.perf_counter()
            self.logger.time("step", now - step_t0)
            step_t0 = now

            if tc.image_every and self.step % tc.image_every == 0:
                image = stage.progress_image(last_batch)
                if image is not None and not self.images.submit(image, f"step_{self.step:08d}.png"):
                    self.logger.count("images_skipped")

            if self.step % tc.log_every == 0:
                epoch_float = self.epoch + stage.data.tracker.fraction_seen
                pbar.write(
                    f"epoch {epoch_float:.3f} | step {self.step} | micro-batches {self.microbatches} | "
                    f"lr {', '.join(f'{m.name} {m.current_lr():.2e}' for m in stage.managers)}"
                )
                self.logger.scalar("progress/epoch", epoch_float, self.step)
                for m in stage.managers:
                    self.logger.scalar(f"lr/{m.name}", m.current_lr(), self.step)
                    self.logger.scalar(f"steps/{m.name}", m.step_count, self.step)
                self.logger.flush(self.step)
                stage.on_log(self.logger, self.step)

            due_time = time.time() - self.last_save_time > tc.save_every_seconds
            due_steps = tc.save_every_steps and self.step % tc.save_every_steps == 0
            if due_time or due_steps:
                self.save("periodic")
            if self._pdb_requested:
                self._enter_pdb()
            if self._interrupted:
                break
        pbar.close()

        # the partial accumulation window (ragged tail, or the micro-batches before a Ctrl-C) is
        # dropped so a step never spans epochs or runs; those images count as seen and are not redone
        for m in stage.managers:
            m.zero_grad()
        stage.discard_partial_step()
        if not self._interrupted:
            self.epoch += 1
            stage.data.tracker.reset()
            print(f"Finished epoch {self.epoch - 1}")
