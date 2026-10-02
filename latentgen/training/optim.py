"""
One optimizer wrapper for the two optimizers the pipeline uses.

* ``adamw_schedulefree`` (stages 1a, 1b, 2): Schedule-Free AdamW. No LR schedule beyond its
  built-in warm-up. Important subtlety: the weights it *steps on* (``y``) are not the weights you
  want to *evaluate or save* (``x``, a running average). ``optimizer.eval()`` swaps ``x`` into the
  model, ``optimizer.train()`` swaps ``y`` back. :meth:`Optimizer.eval_weights` handles that.
* ``adamw`` (stage 3): plain AdamW with a linear warm-up, no averaging.

Either way the trainer only calls ``step``, ``zero_grad``, ``eval_weights`` and ``current_lr``.
"""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import contextmanager

import torch
from torch.optim.lr_scheduler import LambdaLR

from latentgen.config import OptimizerConfig


class Optimizer:
    def __init__(self, params: Iterable[torch.nn.Parameter], cfg: OptimizerConfig) -> None:
        self.cfg = cfg
        params = list(params)
        self.scheduler = None
        if cfg.type == "adamw_schedulefree":
            import schedulefree  # imported lazily so stage 3 (plain AdamW) does not need it

            self.opt = schedulefree.AdamWScheduleFree(
                params,
                lr=cfg.lr,
                betas=tuple(cfg.betas),
                weight_decay=cfg.weight_decay,
                warmup_steps=cfg.warmup_steps,
            )
            self.opt.train()
        elif cfg.type == "adamw":
            self.opt = torch.optim.AdamW(
                params, lr=cfg.lr, betas=tuple(cfg.betas), weight_decay=cfg.weight_decay
            )
            warmup = max(1, cfg.warmup_steps)
            self.scheduler = LambdaLR(self.opt, lambda step: min(1.0, (step + 1) / warmup))
        else:
            raise ValueError(f"unknown optimizer type '{cfg.type}' (use 'adamw_schedulefree' or 'adamw')")

    @property
    def is_schedule_free(self) -> bool:
        return self.scheduler is None

    def step(self) -> None:
        self.opt.step()
        if self.scheduler is not None:
            self.scheduler.step()

    def zero_grad(self) -> None:
        self.opt.zero_grad(set_to_none=True)

    def current_lr(self) -> float:
        group = self.opt.param_groups[0]
        return float(group.get("scheduled_lr", group["lr"]))  # schedule-free exposes the warmed-up LR here

    @contextmanager
    def eval_weights(self):
        """Inside this block the model holds the weights you would evaluate / save with."""
        if self.is_schedule_free:
            self.opt.eval()
        try:
            yield
        finally:
            if self.is_schedule_free:
                self.opt.train()

    def state_dict(self) -> dict:
        return {
            "type": self.cfg.type,
            "optimizer": self.opt.state_dict(),
            "scheduler": None if self.scheduler is None else self.scheduler.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("type") != self.cfg.type:
            raise ValueError(f"checkpoint optimizer is '{state.get('type')}', config says '{self.cfg.type}'")
        _check_state_shapes(self.opt, state["optimizer"])  # must raise BEFORE anything touches the weights
        self.opt.load_state_dict(state["optimizer"])
        if self.scheduler is not None and state.get("scheduler") is not None:
            self.scheduler.load_state_dict(state["scheduler"])
        # load_state_dict also restores the OLD hyper-parameters; the current config wins so that
        # `--set ...optimizer.lr=...` on resume actually changes the learning rate.
        for group in self.opt.param_groups:
            group["lr"] = self.cfg.lr
            group["betas"] = tuple(self.cfg.betas)
            group["weight_decay"] = self.cfg.weight_decay
            if "warmup_steps" in group:
                group["warmup_steps"] = self.cfg.warmup_steps
        if self.scheduler is not None:
            self.scheduler.base_lrs = [self.cfg.lr for _ in self.opt.param_groups]
        if self.is_schedule_free:
            self.opt.train()  # the state was saved in eval mode; this rebuilds the training point y


def _check_state_shapes(opt: torch.optim.Optimizer, state: dict) -> None:
    """Raise if a saved per-parameter tensor does not match its parameter's shape.

    ``torch.optim.Optimizer.load_state_dict`` matches state to parameters *by position*, and
    schedule-free's ``train()`` then lerps every parameter with its saved ``z`` -- a mismatch would
    half-rewrite the weights before failing. Checking first keeps the model untouched on error.
    """
    params = [p for group in opt.param_groups for p in group["params"]]
    saved_ids = [i for group in state.get("param_groups", []) for i in group["params"]]
    if len(saved_ids) != len(params):
        raise ValueError(f"optimizer state has {len(saved_ids)} parameters, model has {len(params)}")
    for param, saved_id in zip(params, saved_ids, strict=True):
        for key, value in state.get("state", {}).get(saved_id, {}).items():
            if torch.is_tensor(value) and value.dim() > 0 and value.shape != param.shape:
                raise ValueError(
                    f"optimizer state '{key}' has shape {tuple(value.shape)} for a {tuple(param.shape)} parameter"
                )
