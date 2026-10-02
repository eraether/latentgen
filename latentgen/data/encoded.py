"""
The encoded dataset: output of stage 1c, input of stages 2 and 3.

One ``.pt`` file holding a dict::

    codes     int16 [V, N, H, W]      VQ-VAE code grid per image
    latents   int8  [V, N, C, H, W]   AE latent per image, stored as round(x * 127)
    stats     {"mean": [...], "std": [...]}  normalisation used when encoding
    meta      {"codebook_size", "latent_scale": 127, "image_size", "patch_size", "files": [...]}

``V`` is the number of variants per image (1, or 2 = original + horizontal flip). Stage 1c writes
it; :class:`EncodedDataset` serves random batches from it on either the GPU (fastest) or CPU RAM
(``data.encoded_device: cpu`` -- a few GB of VRAM saved, batches are copied over as needed).

The old two-file layout (``all_vq_codes.pt`` + ``all_latents.pt``) is converted by
``scripts/convert_legacy.py dataset``.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import torch

from latentgen.data.epoch import EpochTracker
from latentgen.data.normalization import ImageStats

LATENT_SCALE = 127  # int8 quantisation of the [-1, 1] AE latent


def save_encoded(
    path: str | Path, codes: torch.Tensor, latents: torch.Tensor | None, stats: ImageStats, meta: dict
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 2,
            "codes": codes.to(torch.int16),
            "latents": latents,
            "stats": stats.to_dict(),
            "meta": {"latent_scale": LATENT_SCALE, **meta},
        },
        path,
    )


class EncodedDataset:
    """Random-access batches of (codes, latents) with epoch tracking.

    Args:
        path: the encoded ``.pt`` file.
        device: the *training* device; batches are returned here.
        storage: where the tensors live between batches, ``"cpu"`` or ``"cuda"``.
        need_latents: stage 2 (MaskGIT) only needs the codes; skipping the latents saves memory.
    """

    def __init__(
        self, path: str | Path, device: torch.device, storage: str = "cpu", need_latents: bool = True
    ) -> None:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(
                f"encoded dataset not found: {path}. Run scripts/01c_encode_dataset.py first "
                f"(or scripts/convert_legacy.py dataset for old all_vq_codes.pt files)."
            )
        blob = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        if not isinstance(blob, dict) or "codes" not in blob:
            raise ValueError(
                f"{path} is not an encoded dataset (expected a dict with 'codes'); "
                f"legacy files need scripts/convert_legacy.py dataset"
            )

        self.device = device
        self.storage = torch.device(storage if storage != "cuda" else device)
        self.stats = ImageStats.from_dict(blob.get("stats"))
        self.meta = dict(blob.get("meta", {}))

        codes = blob["codes"]
        if codes.dim() != 4:
            raise ValueError(f"codes must be [V, N, H, W], got {tuple(codes.shape)}")
        self.codes = self._store(codes)
        self.num_variants, self.num_items, self.grid_h, self.grid_w = self.codes.shape
        self.codebook_size = int(self.meta.get("codebook_size") or int(self.codes.max()) + 1)

        self.latents = None
        self.latent_dim = None
        if need_latents:
            latents = blob.get("latents")
            if latents is None:
                raise ValueError(
                    f"{path} has no latents; stage 3 needs them (re-run stage 1c with the autoencoder)"
                )
            if latents.shape[:2] != codes.shape[:2] or latents.shape[-2:] != codes.shape[-2:]:
                raise ValueError(f"latents {tuple(latents.shape)} do not match codes {tuple(codes.shape)}")
            if latents.dtype != torch.int8:
                raise ValueError(f"latents must be int8 (round(x * {LATENT_SCALE})), got {latents.dtype}")
            self.latents = self._store(latents)
            self.latent_dim = self.latents.shape[2]
        del blob

        self.tracker = EpochTracker(self.num_items, device=self.storage)
        print(
            f"Encoded dataset: {self.num_items} images x {self.num_variants} variants, grid {self.grid_h}x{self.grid_w}, "
            f"codebook {self.codebook_size}, latents {'none' if self.latents is None else self.latent_dim}ch, "
            f"{self.size_gb():.2f} GB on {self.storage}"
        )

    def _store(self, t: torch.Tensor) -> torch.Tensor:
        # the file is memory-mapped; copying makes the tensor live in RAM (or VRAM) for fast random gathers
        return t.to(self.storage) if self.storage.type == "cuda" else t.clone()

    def size_gb(self) -> float:
        n = self.codes.numel() * self.codes.element_size()
        if self.latents is not None:
            n += self.latents.numel() * self.latents.element_size()
        return n / 1024**3

    def _to_device(self, t: torch.Tensor) -> torch.Tensor:
        return t if t.device == self.device else t.to(self.device)

    def get_batch(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """``idx: [B]`` long on the storage device -> ``(codes [B, H, W] long, latents [B, C, H, W] float or None)``.

        A random variant (flip) is picked per sample. Latents are dequantised to ``[-1, 1]`` with
        uniform noise one int8 step wide, then clamped so a "real" latent can never fall outside
        the generator's ``tanh`` range.
        """
        variant = torch.randint(0, self.num_variants, (idx.numel(),), device=self.storage)
        codes = self._to_device(self.codes[variant, idx]).long()
        latents = None
        if self.latents is not None:
            fine = self._to_device(self.latents[variant, idx]).float() / LATENT_SCALE
            fine = (fine + (torch.rand_like(fine) - 0.5) / LATENT_SCALE).clamp_(-1.0, 1.0)
            latents = fine
        self.tracker.mark(idx)
        return codes, latents

    def batches(self, batch_size: int) -> Iterator[tuple[torch.Tensor, torch.Tensor | None]]:
        """Batches for the part of the epoch not yet seen (ragged tail dropped)."""
        order = self.tracker.remaining_order()
        for b in range(order.numel() // batch_size):
            yield self.get_batch(order[b * batch_size : (b + 1) * batch_size])

    def batches_per_epoch(self, batch_size: int) -> int:
        return self.num_items // batch_size
