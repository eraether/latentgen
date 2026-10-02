"""
Epoch bookkeeping that survives interruption.

Every training stage tracks which items it has seen this epoch in a boolean tensor. The set of
seen ids is saved with each checkpoint, so a resumed run finishes the *remaining* images of the
epoch instead of starting over or re-seeing the same ones.
"""

from __future__ import annotations

import torch


class EpochTracker:
    def __init__(self, num_items: int, device: torch.device | str = "cpu") -> None:
        self.num_items = num_items
        self.seen = torch.zeros(num_items, dtype=torch.bool, device=device)

    def remaining_order(self) -> torch.Tensor:
        """Shuffled ids not yet seen this epoch (``[R]`` long on the tracker's device)."""
        unseen = (~self.seen).nonzero().squeeze(1)
        return unseen[torch.randperm(unseen.numel(), device=unseen.device)]

    def mark(self, idx: torch.Tensor) -> None:
        self.seen[idx.to(self.seen.device)] = True

    def num_seen(self) -> int:
        return int(self.seen.sum())

    def seen_ids(self) -> list[int]:
        return self.seen.nonzero().squeeze(1).tolist()

    def set_seen(self, ids) -> None:
        self.seen.zero_()
        ids = list(ids)
        valid = [i for i in ids if 0 <= i < self.num_items]
        if len(valid) != len(ids):
            print(
                f"[epoch] {len(ids) - len(valid)} seen ids from the checkpoint are out of range "
                f"(dataset now has {self.num_items} items); ignoring them"
            )
        ids = valid
        if ids:
            self.seen[torch.as_tensor(ids, dtype=torch.long, device=self.seen.device)] = True

    def reset(self) -> None:
        self.seen.zero_()

    @property
    def fraction_seen(self) -> float:
        return self.num_seen() / max(1, self.num_items)
