"""
Stage 2: MaskGIT training on the encoded VQ codes.

Per micro-batch: draw a random mask ratio per sample, mask, predict, cross-entropy on the masked
positions. Besides the mean loss we report CE and accuracy *per mask-ratio bucket* (100-91 %
masked, 90-81 %, ...), which is far more informative than one number: the nearly-empty grids
tell you about global structure, the mostly-filled ones about local detail.
"""

from __future__ import annotations

import torch

from latentgen.config import Config
from latentgen.data import ImageStats
from latentgen.device import autocast
from latentgen.nn.maskgit import MaskGIT, masked_cross_entropy, random_mask
from latentgen.training.logging import MetricLogger
from latentgen.training.loop import Stage
from latentgen.training.manager import ModelManager


class MaskRatioBuckets:
    """CE / accuracy accumulated per 10 %-wide mask-ratio bucket, on the GPU, synced once per log."""

    def __init__(self, ratio_min: float, ratio_max: float, device: torch.device) -> None:
        self.num_buckets = max(1, round((ratio_max - ratio_min) * 10))
        self.ratio_min, self.ratio_max = ratio_min, ratio_max
        self.labels = []
        for i in range(self.num_buckets):
            hi = round(ratio_max * 100) - 10 * i
            lo = hi - 9 if i < self.num_buckets - 1 else round(ratio_min * 100)
            self.labels.append(f"{hi}-{lo}%")
        self._sums = torch.zeros(
            3, self.num_buckets, dtype=torch.float64, device=device
        )  # ce, correct, tokens

    def bucket_of(self, ratio: torch.Tensor) -> torch.Tensor:
        return ((self.ratio_max - ratio) * 10 + 1e-6).floor().long().clamp_(0, self.num_buckets - 1)

    @torch.no_grad()
    def add(self, ratio, ce_sum, correct_sum, n_masked) -> None:
        vals = torch.stack([ce_sum, correct_sum, n_masked]).double()
        self._sums.index_add_(1, self.bucket_of(ratio), vals)

    def pop(self) -> list[tuple[str, float, float, int]]:
        """``[(label, ce, accuracy, tokens), ...]`` for each bucket plus an "all" row; resets."""
        ce, correct, tokens = self._sums.cpu().unbind(0)
        self._sums.zero_()
        rows = []
        for i in range(self.num_buckets):
            t = float(tokens[i])
            rows.append(
                (
                    self.labels[i],
                    float(ce[i]) / t if t else float("nan"),
                    float(correct[i]) / t if t else float("nan"),
                    int(t),
                )
            )
        t = float(tokens.sum())
        rows.append(
            (
                "all",
                float(ce.sum()) / t if t else float("nan"),
                float(correct.sum()) / t if t else float("nan"),
                int(t),
            )
        )
        return rows


class MaskGITStage(Stage):
    name = "maskgit"

    def __init__(self, cfg: Config, device: torch.device, data, stats: ImageStats) -> None:
        super().__init__(cfg, device, data, stats)
        model = MaskGIT(cfg.maskgit.model, checkpointing=self.train_cfg.activation_checkpointing)
        self.managers = [
            ModelManager(
                "maskgit", model, self.train_cfg.optimizer, device, compile_model=cfg.project.compile
            )
        ]
        self.buckets = MaskRatioBuckets(self.stage_cfg.mask_ratio_min, self.stage_cfg.mask_ratio_max, device)

    def discard_partial_step(self) -> None:
        self.buckets.pop()  # drop the stats of micro-batches that were never stepped on

    def microbatch(self, batch, scale: float, logger: MetricLogger) -> None:
        codes, _latents = batch
        B, H, W = codes.shape
        mask, ratio = random_mask(
            B, H, W, self.stage_cfg.mask_ratio_min, self.stage_cfg.mask_ratio_max, self.device
        )
        with autocast(self.device):
            logits = self.manager(codes, mask)
        loss, ce_sum, correct_sum, n_masked = masked_cross_entropy(logits, codes, mask)
        (loss * scale).backward()
        logger.log("masked_ce", loss)
        self.buckets.add(ratio, ce_sum.detach(), correct_sum.detach(), n_masked)

    def on_log(self, logger: MetricLogger, step: int) -> None:
        rows = self.buckets.pop()
        print("  masked   CE      acc     tokens")
        for label, ce, acc, tokens in rows:
            if tokens:
                print(f"  {label:<8} {ce:6.4f}  {100 * acc:5.1f}%  {tokens:>9,}")
                key = label.replace("%", "").replace("-", "_")
                logger.scalar(f"bucket_ce/{key}", ce, step)
                logger.scalar(f"bucket_acc/{key}", acc, step)
