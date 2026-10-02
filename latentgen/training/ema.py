"""
Exponential moving average of a model's weights (used by the stage-3 generator).

Every ``every`` successful optimizer steps: ``ema = decay * ema + (1 - decay) * weights``. The
EMA model is a frozen deep copy; sample from it (``ema.model``) for smoother results than the raw
generator. The counter is relative (steps since the last blend) so resuming at any step is fine.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.99, every: int = 10) -> None:
        self.decay = float(decay)
        self.every = int(every)
        self.num_updates = 0
        self.steps_since_update = 0
        self.model = copy.deepcopy(model).eval().requires_grad_(False)
        self._params = list(self.model.parameters())
        self._buffers = list(self.model.buffers())

    def step(self, model: nn.Module) -> bool:
        """Call once per successful optimizer step. Returns True when a blend happened."""
        self.steps_since_update += 1
        if self.steps_since_update < self.every:
            return False
        self.update(model)
        return True

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.steps_since_update = 0
        self.num_updates += 1
        torch._foreach_lerp_(self._params, [p.detach() for p in model.parameters()], 1.0 - self.decay)
        for b_ema, b in zip(self._buffers, model.buffers(), strict=True):
            b_ema.copy_(b)

    @torch.no_grad()
    def copy_from(self, model: nn.Module) -> None:
        """Hard reset: EMA := live weights."""
        for p_ema, p in zip(self._params, model.parameters(), strict=True):
            p_ema.copy_(p.detach())
        for b_ema, b in zip(self._buffers, model.buffers(), strict=True):
            b_ema.copy_(b)
        self.num_updates = 0
        self.steps_since_update = 0

    def state_dict(self) -> dict:
        return {
            "model": self.model.state_dict(),
            "num_updates": self.num_updates,
            "steps_since_update": self.steps_since_update,
            "decay": self.decay,
            "every": self.every,
        }

    def load_state_dict(self, state: dict) -> None:
        self.model.load_state_dict(state["model"])
        self.num_updates = int(state.get("num_updates", 0))
        self.steps_since_update = min(int(state.get("steps_since_update", 0)), self.every - 1)
        # decay / every intentionally come from the current config, so they can be changed on resume
