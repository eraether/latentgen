"""
Command-line plumbing shared by the ``scripts/``: config loading, device setup, run naming.

Every script accepts::

    --config PATH          YAML config (default: configs/ffhq512.yaml)
    --set key=value ...    override any config value, e.g. --set vqvae.train.epochs=3
    --run-name NAME        name of the run directory (default: <timestamp>_<id>)
    --resume PATH          checkpoint file / run dir / runs/<stage>/latest to continue from
                           (the run's own config.yaml is used; --set still applies on top)
    --pdb                  Ctrl-C drops into pdb at the next step boundary instead of saving and exiting
"""

from __future__ import annotations

import argparse
import uuid
from datetime import datetime
from pathlib import Path

import torch

from latentgen.config import Config, load_config
from latentgen.device import configure_torch, get_device, probe_flash_attention, seed_everything
from latentgen.training.checkpoint import find_checkpoint, resolve_checkpoint, run_dir_for

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "ffhq512.yaml"


class _HelpFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    """Keep the usage examples in each script's docstring *and* show argument defaults."""


def build_parser(description: str, training: bool = True) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description, formatter_class=_HelpFormatter)
    p.add_argument("--config", default=None, help=f"YAML config file (default: {DEFAULT_CONFIG})")
    p.add_argument(
        "--set",
        nargs="*",
        action="extend",
        default=[],
        metavar="KEY=VALUE",
        help="config overrides (YAML-parsed values); may be repeated",
    )
    if training:
        p.add_argument("--run-name", default=None, help="run directory name under runs/<stage>/")
        p.add_argument("--resume", default=None, help="checkpoint / run dir / runs/<stage>/latest to resume")
        p.add_argument("--pdb", action="store_true", help="Ctrl-C enters pdb instead of saving and exiting")
    return p


def load_cfg(args: argparse.Namespace) -> Config:
    """The config for this invocation.

    * fresh run: ``--config`` (default ``configs/ffhq512.yaml``) + ``--set`` overrides
    * ``--resume``: the resumed run's own ``config.yaml`` + ``--set`` overrides, so a resumed run is the
      same experiment (same model sizes, same data) unless you explicitly change something. ``--config``
      is ignored in that case, with a notice.
    """
    if getattr(args, "resume", None):
        run_dir = resolve_checkpoint(args.resume).parent.parent
        run_config = run_dir / "config.yaml"
        if run_config.is_file():
            if args.config:
                print(f"--resume: using {run_config} (ignoring --config {args.config})")
            return load_config(run_config, args.set)
        print(f"--resume: {run_config} not found, falling back to --config")
    return load_config(args.config or DEFAULT_CONFIG, args.set)


def setup(cfg: Config) -> torch.device:
    """Global torch settings, seeding, device, attention-kernel probe. Call once at the top of every script."""
    configure_torch(cfg.project.matmul_precision)
    seed = seed_everything(cfg.project.seed)
    print(f"Seed: {seed}")
    device = get_device(allow_cpu=cfg.project.allow_cpu)
    if probe_flash_attention(device):
        print("FlashAttention: available")
    return device


def checkpoint_for(cfg: Config, value: str, stage: str) -> Path:
    """Turn a ``*_checkpoint`` config value into a file (see :func:`find_checkpoint`)."""
    return find_checkpoint(value, cfg.project.runs_dir, stage)


def new_run_name() -> str:
    return f"{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"


def choose_run_dir(cfg: Config, stage: str, args: argparse.Namespace) -> tuple[Path, Path | None]:
    """``(run_dir, checkpoint_to_resume_or_None)``. Resuming reuses the checkpoint's own run dir."""
    if args.resume:
        ckpt = resolve_checkpoint(args.resume)
        if args.run_name:
            print(f"--run-name is ignored with --resume (continuing in {ckpt.parent.parent})")
        return ckpt.parent.parent, ckpt
    return run_dir_for(cfg.project.runs_dir, stage, args.run_name or new_run_name()), None
