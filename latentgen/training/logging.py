"""
Console + TensorBoard logging with no per-step GPU syncs.

Scalars are accumulated as GPU tensors and only converted to Python floats when a log line is
written (every ``log_every`` optimizer steps), so logging never stalls the training stream.
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter


class MetricLogger:
    def __init__(self, log_dir: str | Path) -> None:
        self.writer = SummaryWriter(str(log_dir))
        self._running: dict[str, list[torch.Tensor | float]] = defaultdict(list)
        self._timings: dict[str, list[float]] = defaultdict(list)
        self._counts: dict[str, int] = defaultdict(int)

    # -- accumulate (cheap, no sync) ------------------------------------------------------
    def log(self, name: str, value) -> None:
        self._running[name].append(value.detach().float() if torch.is_tensor(value) else float(value))

    def time(self, name: str, seconds: float) -> None:
        self._timings[name].append(seconds)

    def count(self, name: str, n: int = 1) -> None:
        self._counts[name] += n

    # -- flush (one sync) -----------------------------------------------------------------
    def scalar(self, name: str, value: float, step: int) -> None:
        self.writer.add_scalar(name, value, step)

    def flush(self, step: int, prefix: str = "train") -> dict[str, float]:
        """Write all accumulated metrics for ``step``, print them, and reset. Returns the means."""
        means: dict[str, float] = {}
        for name in sorted(self._running):
            values = [v.reshape(()) if torch.is_tensor(v) else torch.tensor(v) for v in self._running[name]]
            device = next((v.device for v in values if torch.is_tensor(v)), "cpu")
            mean = torch.stack([v.to(device) for v in values]).mean().item()
            if math.isfinite(mean):
                means[name] = mean
                self.writer.add_scalar(f"{prefix}/{name}", mean, step)
        for name, values in self._timings.items():
            self.writer.add_scalar(f"timing/{name}_ms", 1000 * sum(values) / len(values), step)
            means[f"{name}_ms"] = 1000 * sum(values) / len(values)
        for name, n in self._counts.items():
            self.writer.add_scalar(f"count/{name}", n, step)
            means[f"#{name}"] = n
        self._running.clear()
        self._timings.clear()
        self._counts.clear()
        if means:
            print("  " + " | ".join(f"{k} {v:.4g}" for k, v in means.items()))
        return means

    def close(self) -> None:
        self.writer.flush()
        self.writer.close()
