"""
Per-channel image normalisation.

Images are stored as ``[0, 1]`` floats and normalised to roughly zero mean / unit variance with
per-channel statistics of the training set. The defaults are FFHQ's; for your own dataset run
``scripts/00_compute_dataset_stats.py`` and point ``data.stats_file`` at the JSON it writes.

The stats travel with every checkpoint and with the encoded dataset, so downstream stages and
inference never need the original config to undo the normalisation.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

FFHQ_MEAN = (0.5326635696186963, 0.4840257880077774, 0.4746593392437331)
FFHQ_STD = (0.23375539610753213, 0.22486911131803633, 0.22463677431223736)
AUDIO_MEAN = (0.0,)  # mono waveforms in [-1, 1] are already centred; scale them up to ~unit variance
AUDIO_STD = (0.1,)


@dataclass(frozen=True, eq=True)
class ImageStats:
    """Per-channel mean / std. Three values for RGB images, one for mono audio."""

    mean: tuple[float, ...] = FFHQ_MEAN
    std: tuple[float, ...] = FFHQ_STD

    @classmethod
    def load(cls, path: str | Path | None, kind: str = "image") -> ImageStats:
        """Read ``{"mean": [...], "std": [...]}``; ``None`` gives the defaults for ``kind`` (image / audio)."""
        if path is None:
            return cls(mean=AUDIO_MEAN, std=AUDIO_STD) if kind == "audio" else cls()
        with open(path) as f:
            data = json.load(f)
        return cls(mean=tuple(data["mean"]), std=tuple(data["std"]))

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict | None) -> ImageStats:
        if not data:
            return cls()
        return cls(mean=tuple(data["mean"]), std=tuple(data["std"]))

    def _tensors(self, x: torch.Tensor, batched: bool) -> tuple[torch.Tensor, torch.Tensor]:
        # cached per (device, dtype, shape): called every micro-batch, and a fresh host->device copy each
        # time would stall. Works for images [B, C, H, W] and 1-D signals [B, C, T] (and unbatched).
        channel_dim = 1 if batched else 0
        shape = [1] * x.dim()
        shape[channel_dim] = -1
        key = (x.device, x.dtype, tuple(shape))
        cache = self.__dict__.setdefault("_cache", {})  # frozen dataclass: bypass __setattr__
        if key not in cache:
            cache[key] = (
                torch.tensor(self.mean, device=x.device, dtype=x.dtype).view(shape),
                torch.tensor(self.std, device=x.device, dtype=x.dtype).view(shape),
            )
        return cache[key]

    def __getstate__(self) -> dict:  # DataLoader workers pickle the dataset: never ship cached CUDA tensors
        return {"mean": self.mean, "std": self.std}

    def __setstate__(self, state: dict) -> None:
        object.__setattr__(self, "mean", state["mean"])
        object.__setattr__(self, "std", state["std"])

    def normalize(self, x: torch.Tensor, batched: bool = True) -> torch.Tensor:
        """raw ``[0, 1]`` image (or waveform) -> normalised. ``batched=False`` for a single ``[C, ...]`` item."""
        mean, std = self._tensors(x, batched)
        return (x - mean) / std

    def denormalize(self, x: torch.Tensor, batched: bool = True) -> torch.Tensor:
        """normalised -> raw (not clamped)."""
        mean, std = self._tensors(x, batched)
        return x * std + mean
