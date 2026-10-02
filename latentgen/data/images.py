"""
Stage 1 input: a plain folder of images.

Any folder works -- file names do not matter, sub-folders are searched, and every image is
resized (shorter side) and centre-cropped to ``image_size``. Items are addressed by their index in
the sorted file list, which is what the epoch tracker and the encoded dataset use as the image id.
"""

from __future__ import annotations

import signal
from collections.abc import Iterator
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from latentgen.config import IMAGE_EXTENSIONS
from latentgen.data.epoch import EpochTracker
from latentgen.data.normalization import ImageStats


def _ignore_sigint(_worker_id: int) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def list_images(image_dir: str | Path, extensions=IMAGE_EXTENSIONS) -> list[Path]:
    image_dir = Path(image_dir)
    if not image_dir.is_dir():
        raise FileNotFoundError(f"image folder not found: {image_dir} (set data.image_dir in your config)")
    exts = {f".{e.lower().lstrip('.')}" for e in extensions}
    files = sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in exts)
    if not files:
        raise FileNotFoundError(f"no images with extensions {sorted(exts)} under {image_dir}")
    return files


def load_image(path: str | Path, image_size: int | None = None) -> torch.Tensor:
    """Read an image as a ``[3, H, W]`` float tensor in ``[0, 1]`` (resized + centre-cropped if asked)."""
    with Image.open(path) as img:
        img = img.convert("RGB")
        if image_size is not None and img.size != (image_size, image_size):
            w, h = img.size
            scale = image_size / min(w, h)
            img = img.resize(
                (max(image_size, round(w * scale)), max(image_size, round(h * scale))), Image.BICUBIC
            )
            w, h = img.size
            left, top = (w - image_size) // 2, (h - image_size) // 2
            img = img.crop((left, top, left + image_size, top + image_size))
        data = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8)
        return data.view(img.height, img.width, 3).permute(2, 0, 1).float().div_(255.0)


def to_pil(image: torch.Tensor) -> Image.Image:
    """``[3, H, W]`` float in ``[0, 1]`` -> PIL image."""
    data = image.detach().clamp(0, 1).mul(255).round().to(torch.uint8).permute(1, 2, 0).contiguous().cpu()
    return Image.frombytes("RGB", (data.shape[1], data.shape[0]), data.numpy().tobytes())


class ImageFolderDataset(Dataset):
    """Yields ``(normalised image [3, S, S], index)``. Optional random horizontal flip."""

    def __init__(
        self,
        image_dir: str | Path,
        image_size: int,
        stats: ImageStats,
        horizontal_flip: bool = True,
        extensions=IMAGE_EXTENSIONS,
    ) -> None:
        self.files = list_images(image_dir, extensions)
        self.image_size = image_size
        self.stats = stats
        self.horizontal_flip = horizontal_flip

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image = self.stats.normalize(load_image(self.files[index], self.image_size), batched=False)
        if self.horizontal_flip and torch.rand(()) < 0.5:
            image = image.flip(-1)
        return image, index


def image_dataset_from_config(cfg, stats: ImageStats, horizontal_flip: bool | None = None) -> Dataset:
    """The stage-1 dataset described by ``cfg.data``: an image folder, or a wav folder when ``data.kind`` is audio."""
    flip = cfg.data.horizontal_flip if horizontal_flip is None else horizontal_flip
    if cfg.data.kind == "audio":
        from latentgen.data.audio import AudioFolderDataset

        return AudioFolderDataset(cfg.data.image_dir, cfg.data.audio_length, stats, flip)
    if cfg.data.kind != "image":
        raise ValueError(f"data.kind must be 'image' or 'audio', got {cfg.data.kind!r}")
    return ImageFolderDataset(cfg.data.image_dir, cfg.data.image_size, stats, flip, cfg.data.image_extensions)


class ImageBatches:
    """Epoch-aware batch iterator over an :class:`ImageFolderDataset` (the stage-1 data source).

    ``batches()`` yields ``(images on device, ids)`` for the part of the epoch not yet seen, marking
    each batch as seen so a checkpoint taken mid-epoch resumes exactly where it stopped.
    """

    def __init__(self, dataset: ImageFolderDataset, device: torch.device, num_workers: int = 4) -> None:
        self.dataset = dataset
        self.device = device
        self.num_workers = num_workers
        self.tracker = EpochTracker(len(dataset), device="cpu")

    @property
    def num_items(self) -> int:
        return len(self.dataset)

    def batches(self, batch_size: int) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        order = self.tracker.remaining_order().tolist()
        loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            sampler=order,
            drop_last=True,
            num_workers=self.num_workers,
            pin_memory=self.device.type == "cuda",
            worker_init_fn=_ignore_sigint,
        )  # Ctrl-C is handled by the trainer, not the workers
        for images, ids in loader:
            self.tracker.mark(ids)
            yield images.to(self.device, non_blocking=True), ids

    def batches_per_epoch(self, batch_size: int) -> int:
        return len(self.dataset) // batch_size
