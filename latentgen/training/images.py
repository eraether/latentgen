"""
Progress images: composing a grid of samples and writing it to disk off the training thread.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from latentgen.data.images import to_pil


def image_grid(columns: list[torch.Tensor], max_rows: int = 2, downscale: int = 1) -> torch.Tensor:
    """Tile images: each entry of ``columns`` is a ``[B, 3, H, W]`` batch shown as one column,
    rows are batch items. Returns one ``[3, rows*H, cols*W]`` image in ``[0, 1]``.

    Example: ``image_grid([inputs, reconstructions])`` shows input | reconstruction per row.
    """
    rows = min(max_rows, min(c.size(0) for c in columns))
    tiled = torch.cat([torch.cat([col[i] for col in columns], dim=-1) for i in range(rows)], dim=-2)
    if downscale > 1:
        C, H, W = tiled.shape
        tiled = tiled.view(C, H // downscale, downscale, W // downscale, downscale).mean(dim=(2, 4))
    return tiled.float().clamp(0, 1)


def tile(images: list[torch.Tensor], per_row: int) -> torch.Tensor:
    """Lay ``[3, H, W]`` images out ``per_row`` to a row (left to right, then top to bottom), padding the last row."""
    rows = [torch.cat(images[i : i + per_row], dim=-1) for i in range(0, len(images), per_row)]
    width = max(r.shape[-1] for r in rows)
    rows = [torch.nn.functional.pad(r, (0, width - r.shape[-1])) for r in rows]
    return torch.cat(rows, dim=-2)


def save_image(image: torch.Tensor, path: str | Path) -> None:
    """Write a ``[3, H, W]`` float image in ``[0, 1]`` as PNG/JPEG (by extension)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    to_pil(image).save(path)


class BackgroundImageSaver:
    """Saves progress images on a worker thread so the training loop only pays for the GPU->CPU copy.

    If the previous image is still being written, the new one is skipped (never queued), so a
    slow disk can never back the trainer up.
    """

    def __init__(self, out_dir: str | Path) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="image_save")
        self._pending = None

    def submit(self, image: torch.Tensor, filename: str) -> bool:
        if self._pending is not None and not self._pending.done():
            return False
        cpu = image.detach().to("cpu", non_blocking=False)  # sync copy keeps this simple and correct
        self._pending = self._pool.submit(self._write, cpu, self.out_dir / filename)
        return True

    @staticmethod
    def _write(cpu: torch.Tensor, path: Path) -> None:
        try:
            save_image(cpu, path)
        except Exception as e:  # noqa: BLE001 - never let a failed PNG write kill training
            print(f"[BackgroundImageSaver] failed to write {os.fspath(path)}: {e}")

    def close(self) -> None:
        self._pool.shutdown(wait=True)
