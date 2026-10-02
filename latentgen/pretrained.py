"""
Loading *frozen* models from checkpoints for the stages that consume them.

    stage 1c  needs the VQ-VAE and the autoencoder     -> load_vqvae / load_autoencoder
    stage 3   needs the autoencoder (progress images)   -> load_autoencoder
    generate  needs all four                            -> + load_maskgit / load_generator

Every loader returns ``(model in eval mode with no grads, ImageStats)`` where the stats are the
normalization the model was trained with, read from the checkpoint itself.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from latentgen.config import (
    AutoencoderModelConfig,
    TransformerModelConfig,
    VQVAEModelConfig,
    from_dict,
)
from latentgen.data.normalization import ImageStats
from latentgen.nn import VQVAE, Autoencoder, Generator, MaskGIT
from latentgen.training.checkpoint import load_checkpoint

_CONFIG_CLASS = {
    VQVAE: VQVAEModelConfig,
    Autoencoder: AutoencoderModelConfig,
    MaskGIT: TransformerModelConfig,
    Generator: TransformerModelConfig,
}


def _load(path, stage: str, model_cls: type[nn.Module], section: tuple[str, ...]):
    """Common part of every loader: read on CPU, build the model from the stored config, freeze on ``device``."""
    ckpt = load_checkpoint(path, "cpu")  # CPU: a checkpoint also holds optimizer / EMA / other models
    if ckpt["stage"] != stage:
        raise ValueError(f"{ckpt['path']} is a '{ckpt['stage']}' checkpoint, expected '{stage}'")
    cfg_dict = ckpt["config"]
    for key in section:
        cfg_dict = cfg_dict[key]
    model = model_cls(from_dict(_CONFIG_CLASS[model_cls], cfg_dict))
    return ckpt, model


def _freeze(model: nn.Module, ckpt: dict, device: torch.device) -> nn.Module:
    model.checkpoint_path = ckpt["path"]  # handy for provenance (stage 1c records it in encoded.pt)
    return model.to(device).eval().requires_grad_(False)


def load_vqvae(path, device: torch.device, require_cutover: bool = True) -> tuple[VQVAE, ImageStats]:
    ckpt, model = _load(path, "vqvae", VQVAE, ("vqvae", "model"))
    model.load_state_dict(ckpt["models"]["vqvae"]["model"])
    if require_cutover and not model.bottleneck.codebook.is_cut_over:
        raise SystemExit(
            f"{ckpt['path']}: this VQ-VAE has not cut over to a fixed codebook yet, so its codes are "
            "re-randomized on every call and cannot be used downstream. Train past vqvae.cutover_step, "
            "or resume it with --cutover-now."
        )
    print(f"Loaded VQ-VAE {ckpt['path']} (step {ckpt['progress']['step']})")
    return _freeze(model, ckpt, device), ImageStats.from_dict(ckpt["stats"])


def load_autoencoder(path, device: torch.device) -> tuple[Autoencoder, ImageStats]:
    ckpt, model = _load(path, "autoencoder", Autoencoder, ("autoencoder", "model"))
    model.load_state_dict(ckpt["models"]["autoencoder"]["model"])
    print(f"Loaded autoencoder {ckpt['path']} (step {ckpt['progress']['step']})")
    return _freeze(model, ckpt, device), ImageStats.from_dict(ckpt["stats"])


def load_maskgit(path, device: torch.device) -> tuple[MaskGIT, ImageStats]:
    ckpt, model = _load(path, "maskgit", MaskGIT, ("maskgit", "model"))
    model.load_state_dict(ckpt["models"]["maskgit"]["model"])  # schedule-free averaged weights
    print(f"Loaded MaskGIT {ckpt['path']} (step {ckpt['progress']['step']}, {model.cfg.num_layers} layers)")
    return _freeze(model, ckpt, device), ImageStats.from_dict(ckpt["stats"])


def load_generator(path, device: torch.device, use_ema: bool = True) -> tuple[Generator, ImageStats]:
    ckpt, model = _load(path, "cgan", Generator, ("cgan", "generator"))
    state = ckpt["models"]["generator"]
    if use_ema and state.get("ema") is not None:
        model.load_state_dict(state["ema"]["model"])
        which = f"EMA weights ({state['ema'].get('num_updates', '?')} blends)"
    else:
        model.load_state_dict(state["model"])
        which = "raw weights"
    print(f"Loaded GAN generator {ckpt['path']} (step {ckpt['progress']['step']}, {which})")
    return _freeze(model, ckpt, device), ImageStats.from_dict(ckpt["stats"])
