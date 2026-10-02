"""
Checkpoint files and run directories.

Run directory layout (one per training run)::

    runs/<stage>/<run_name>/
        config.yaml           the full config the run was started with
        checkpoints/step_000123.pt
        tensorboard/          `tensorboard --logdir runs`
        images/               progress images

Checkpoint format (``format_version: 2``)::

    {
      "format_version": 2,
      "stage": "vqvae" | "autoencoder" | "maskgit" | "cgan",
      "config": {...full Config as a dict...},
      "stats": {"mean": [...], "std": [...]},          # image normalization
      "models": {<name>: {"model": state_dict, "optimizer": ..., "ema": ..., "step_count": int}},
      "progress": {"epoch": int, "step": int, "microbatches": int, "seen_ids": [...]},
      "stage_state": {...anything stage-specific...},
    }

Any path argument that names a checkpoint may be a ``.pt`` file, a run directory (newest step is
used) or ``runs/<stage>/latest`` (newest run of that stage). See :func:`resolve_checkpoint`.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    import torch.nn as nn

FORMAT_VERSION = 2
_STEP_RE = re.compile(r"step_(\d+)\.pt$")


# ----------------------------------------------------------------------------- paths


def run_dir_for(runs_dir: str | Path, stage: str, run_name: str) -> Path:
    return Path(runs_dir) / stage / run_name


def newest_run_dir(stage_dir: Path) -> Path | None:
    """The run whose newest checkpoint file was written most recently (not the newest run *name*)."""
    if not stage_dir.is_dir():
        return None
    runs = [p for p in stage_dir.iterdir() if p.is_dir() and (p / "checkpoints").is_dir()]
    if not runs:
        return None
    return max(
        runs, key=lambda p: max((c.stat().st_mtime for c in (p / "checkpoints").glob("step_*.pt")), default=0)
    )


def newest_checkpoint(run_dir: Path) -> Path | None:
    files = [
        (int(m.group(1)), p)
        for p in (run_dir / "checkpoints").glob("step_*.pt")
        if (m := _STEP_RE.search(p.name))
    ]
    return max(files)[1] if files else None


def resolve_checkpoint(path: str | Path) -> Path:
    """``.pt`` file -> itself; run dir -> its newest step; ``runs/<stage>/latest`` -> newest run's newest step."""
    path = Path(path)
    if path.is_file():
        return path
    if path.name == "latest" and path.parent.is_dir():
        run = newest_run_dir(path.parent)
        if run is None:
            raise FileNotFoundError(f"no runs with checkpoints under {path.parent}")
        path = run
    if path.is_dir():
        ckpt = newest_checkpoint(path)
        if ckpt is None:
            raise FileNotFoundError(f"no checkpoints/step_*.pt files in {path}")
        return ckpt
    raise FileNotFoundError(f"checkpoint not found: {path}")


def find_checkpoint(value: str | Path, runs_dir: str | Path, stage: str) -> Path:
    """Resolve a ``*_checkpoint`` config value for ``stage``.

    ``"latest"`` -> newest run under ``<runs_dir>/<stage>/``; a run name -> ``<runs_dir>/<stage>/<name>``;
    anything that exists as given (file, run dir, ``.../latest``) -> used as is.
    """
    stage_dir = Path(runs_dir) / stage
    given = Path(value)
    try:
        if str(value) == "latest":
            return resolve_checkpoint(stage_dir / "latest")
        if given.exists() or (given.name == "latest" and given.parent != Path() and given.parent.is_dir()):
            return resolve_checkpoint(given)
        if (stage_dir / str(value)).exists():
            return resolve_checkpoint(stage_dir / str(value))
    except FileNotFoundError as e:
        raise FileNotFoundError(f"{stage} checkpoint '{value}' not found: {e}") from None
    raise FileNotFoundError(
        f"{stage} checkpoint '{value}' not found: tried '{given}' and '{stage_dir / str(value)}'. "
        f"Train the stage first, or point the config at a run dir / step_*.pt file."
    )


# ----------------------------------------------------------------------------- state dict merging


def merge_state_dict(model: nn.Module, loaded: dict[str, torch.Tensor], verbose: bool = True) -> bool:
    """Load ``loaded`` into ``model`` as far as shapes allow.

    Keys that are missing keep their initial values, extra keys are dropped, and a tensor whose
    size changed in exactly one dimension is copied over the overlapping slice (handy when e.g.
    growing the codebook). Returns True when everything matched exactly -- the caller uses that
    to decide whether the optimizer state can be restored too.
    """
    current = model.state_dict()
    merged = {}
    exact = True
    for key, value in current.items():
        if key not in loaded:
            exact = False
            if verbose:
                print(f"  [load] missing in checkpoint, kept initial: {key}")
            merged[key] = value
            continue
        old = loaded[key]
        if old.shape == value.shape:
            merged[key] = old
            continue
        exact = False
        if verbose:
            print(f"  [load] shape changed {tuple(old.shape)} -> {tuple(value.shape)}: {key}")
        if old.dim() == value.dim() and sum(a != b for a, b in zip(old.shape, value.shape, strict=True)) == 1:
            slices = tuple(slice(0, min(a, b)) for a, b in zip(old.shape, value.shape, strict=True))
            value = value.clone()
            value[slices] = old[slices]
            if verbose:
                print(f"         copied the overlapping slice {[s.stop for s in slices]}")
        merged[key] = value
    for key in loaded:
        if key not in current:
            exact = False
            if verbose:
                print(f"  [load] in checkpoint but not in model, dropped: {key}")
    model.load_state_dict(merged)
    return exact


# ----------------------------------------------------------------------------- read / write


def save_checkpoint(path: str | Path, payload: dict) -> Path:
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({"format_version": FORMAT_VERSION, **payload}, tmp)
    tmp.replace(path)  # atomic: a crash mid-write never leaves a half-written step file
    return path


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> dict:
    import torch

    path = resolve_checkpoint(path)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if ckpt.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"{path} is not a format-{FORMAT_VERSION} checkpoint written by this code")
    ckpt["path"] = path
    return ckpt
