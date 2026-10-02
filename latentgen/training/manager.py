"""
:class:`ModelManager` bundles one model with its optimizer, optional EMA and compiled forward.

The trainer and the stages talk to models only through this class, so the GAN stage (two
models) and the single-model stages share the same step / save / load code.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn as nn

from latentgen.config import OptimizerConfig
from latentgen.device import count_parameters, maybe_compile
from latentgen.training.checkpoint import merge_state_dict
from latentgen.training.ema import EMA
from latentgen.training.optim import Optimizer


class ModelManager:
    def __init__(
        self,
        name: str,
        model: nn.Module,
        optimizer_cfg: OptimizerConfig,
        device: torch.device,
        compile_model: bool = True,
        ema_decay: float | None = None,
        ema_every: int = 10,
    ) -> None:
        self.name = name
        self.device = device
        self.model = model.to(device).train()
        # `net` is what you call for forward passes; `model` is what you save / inspect.
        # torch.compile returns a wrapper, so keeping both avoids "_orig_mod." prefixes in checkpoints.
        # dynamic=False: one static graph per batch shape (training only ever sees one or two shapes).
        self.net = maybe_compile(self.model, compile_model, dynamic=False)
        self.ema = EMA(self.model, ema_decay, ema_every) if ema_decay is not None else None
        self.optimizer = Optimizer(self.model.parameters(), optimizer_cfg)
        self.step_count = 0  # successful optimizer steps of *this* model
        print(f"[{name}] ", end="")
        count_parameters(self.model)

    def __call__(self, *args, **kwargs):
        return self.net(*args, **kwargs)

    def reset_optimizer(self) -> None:
        """Fresh optimizer state (after a structural change such as the VQ-VAE codebook cutover)."""
        self.optimizer = Optimizer(self.model.parameters(), self.optimizer.cfg)

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()

    def clip_and_step(self, max_norm: float) -> tuple[bool, torch.Tensor]:
        """Clip gradients, step the optimizer (and EMA). Skips the step if the gradient norm is non-finite.

        One host sync per step (the ``isfinite`` check), which is also the moment the GPU queue drains.
        """
        # max_norm <= 0 means "no clipping"; the norm is still computed for the finiteness check
        norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=max_norm if max_norm > 0 else float("inf")
        )
        if not torch.isfinite(norm):
            self.zero_grad()
            return False, norm
        self.optimizer.step()
        self.zero_grad()
        self.step_count += 1
        if self.ema is not None:
            self.ema.step(self.model)
        return True, norm

    @contextmanager
    def eval_weights(self):
        """Model in eval mode holding the weights you would sample / save with (see optim.py)."""
        self.model.eval()
        with self.optimizer.eval_weights():
            try:
                yield self.model
            finally:
                self.model.train()

    def current_lr(self) -> float:
        return self.optimizer.current_lr()

    # -- checkpointing ----------------------------------------------------------------------
    def state_dict(self) -> dict:
        # Everything is cloned INSIDE eval_weights(): state_dict() returns views of the live
        # parameters, and leaving the context swaps the schedule-free training point back in place.
        with self.eval_weights():
            return {
                "model": {k: v.detach().clone() for k, v in self.model.state_dict().items()},
                "optimizer": self.optimizer.state_dict(),
                "ema": None if self.ema is None else self.ema.state_dict(),
                "step_count": self.step_count,
            }

    def load_state_dict(self, state: dict, load_optimizer: bool = True) -> None:
        exact = merge_state_dict(self.model, state["model"])
        if load_optimizer and exact and state.get("optimizer") is not None:
            try:
                self.optimizer.load_state_dict(state["optimizer"])
            except (ValueError, KeyError, RuntimeError) as e:  # different optimizer type / library version
                print(f"[{self.name}] optimizer state not restored ({e}); starting the optimizer fresh")
        elif load_optimizer:
            print(
                f"[{self.name}] optimizer state not restored (weights changed shape); starting the optimizer fresh"
            )
        if self.ema is not None:
            if exact and state.get("ema") is not None:
                self.ema.load_state_dict(state["ema"])
                print(f"[{self.name}] EMA restored ({self.ema.num_updates} blends)")
            else:
                self.ema.copy_from(self.model)
                print(f"[{self.name}] EMA initialised from the loaded weights")
        self.step_count = int(state.get("step_count", 0))
