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

    def seen_ids(self) -> torch.Tensor:
        """Ids seen this epoch as a CPU long tensor (what checkpoints store)."""
        return self.seen.nonzero().squeeze(1).cpu()

    def set_seen(self, ids) -> None:
        self.seen.zero_()
        ids = torch.as_tensor(ids, dtype=torch.long).flatten().cpu()
        valid = (ids >= 0) & (ids < self.num_items)
        if not bool(valid.all()):
            print(
                f"[epoch] {int((~valid).sum())} seen ids from the checkpoint are out of range "
                f"(dataset now has {self.num_items} items); ignoring them"
            )
        ids = ids[valid]
        if ids.numel():
            self.seen[ids.to(self.seen.device)] = True

    def reset(self) -> None:
        self.seen.zero_()

    @property
    def fraction_seen(self) -> float:
        return self.num_seen() / max(1, self.num_items)
