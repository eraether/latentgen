"""
Configuration: typed dataclasses, loaded from one YAML file, with ``--set key=value`` overrides.

Every stage reads the same YAML (see ``configs/ffhq512.yaml``). The dataclasses below are
the single source of truth for every setting and its default, so a YAML file only needs to
list what differs from the defaults.

Usage::

    cfg = load_config("configs/ffhq512.yaml", overrides=["vqvae.train.epochs=3"])
    cfg.vqvae.model.codebook_size   # -> 256
"""

from __future__ import annotations

import dataclasses
import typing
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

# ----------------------------------------------------------------------------- shared pieces


@dataclass
class ProjectConfig:
    """Settings that apply to every stage."""

    runs_dir: str = "runs"  # every training run writes to <runs_dir>/<stage>/<run_name>/
    seed: int | None = None  # None = random seed each run
    compile: bool = True  # torch.compile the models (big speed-up, slow first step)
    allow_cpu: bool = False  # refuse to train on CPU unless explicitly allowed
    matmul_precision: str = "high"  # torch.set_float32_matmul_precision


IMAGE_EXTENSIONS = ("png", "jpg", "jpeg", "webp", "bmp")


@dataclass
class DataConfig:
    """Where the images are and how to read them."""

    kind: str = "image"  # "image" (folder of pictures, 2-D) or "audio" (folder of 16-bit wav files, 1-D)
    image_dir: str = "data/images"  # folder of images -- or of .wav files when kind is "audio"
    image_size: int = 512  # images are resized (shorter side) + centre-cropped to this
    audio_length: int = 16384  # audio: clips are cropped / zero-padded to this many samples
    sample_rate: int = 16000  # audio: only used when writing generated .wav files
    image_extensions: tuple[str, ...] = IMAGE_EXTENSIONS
    stats_file: str | None = None  # JSON with per-channel mean/std; None = FFHQ defaults
    horizontal_flip: bool = True  # random horizontal flip augmentation
    num_workers: int = 4  # DataLoader workers for stages 1a / 1b
    encoded_file: str = "data/encoded.pt"  # output of stage 1c, input of stages 2 and 3
    encoded_device: str = "cpu"  # where the encoded dataset lives: "cpu" (saves VRAM) or "cuda"


@dataclass
class OptimizerConfig:
    type: str = "adamw_schedulefree"  # "adamw_schedulefree" or "adamw"
    lr: float = 3e-4
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 5e-5
    warmup_steps: int = 100  # linear LR warm-up in optimizer steps


@dataclass
class TrainConfig:
    """The training loop knobs shared by every stage."""

    epochs: int = 10
    batch_size: int = 64  # samples per optimizer step
    microbatch_size: int = 8  # samples per forward/backward (gradient accumulation = batch/microbatch)
    grad_clip: float = 1.0  # gradient-norm clip (0 = no clipping; non-finite norms always skip the step)
    log_every: int = 10  # optimizer steps between console / TensorBoard logs
    image_every: int = 20  # optimizer steps between progress images (0 = never; MaskGIT has none)
    save_every_seconds: int = 3600  # wall-clock checkpoint interval
    save_every_steps: int = 0  # additional step-based checkpoint interval (0 = off)
    activation_checkpointing: bool = False  # recompute layer activations in backward (less VRAM, ~30% slower)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)

    @property
    def microbatches_per_step(self) -> int:
        if self.batch_size % self.microbatch_size:
            raise ValueError(
                f"batch_size={self.batch_size} must be a multiple of microbatch_size={self.microbatch_size}"
            )
        return self.batch_size // self.microbatch_size


# ----------------------------------------------------------------------------- stage 1a: VQ-VAE


@dataclass
class VQVAEModelConfig:
    hidden_size: int = 256
    intermediate_size: int = 1024
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    bottleneck_dim: int = 64  # dimension of each codebook vector
    codebook_size: int = 256  # number of codes (= MaskGIT vocabulary)
    num_codebook_layers: int = 6  # transformer layers in the codebook hypernetwork
    num_attention_heads: int = 16
    channels: int = 3  # filled from data.kind: 3 for images, 1 for audio
    dims: int = 2  # filled from data.kind: 2 for images, 1 for audio
    patch_size: int = 16  # image_size / patch_size = token grid side (512/16 = 32)


@dataclass
class VQVAEStageConfig:
    model: VQVAEModelConfig = field(default_factory=VQVAEModelConfig)
    train: TrainConfig = field(default_factory=lambda: TrainConfig(batch_size=64, microbatch_size=8))
    cutover_step: int = 2200  # optimizer step at which the hypernetwork codebook is frozen into a plain table
    commitment_weight: float = 1.0
    loss_scale: float = 10.0  # inputs/targets are multiplied by this before smooth-L1 (see losses.py)


# ----------------------------------------------------------------------------- stage 1b: AE


@dataclass
class AutoencoderModelConfig:
    hidden_size: int = 256
    intermediate_size: int = 1024
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    bottleneck_dim: int = 32  # channels of the continuous latent
    channels: int = 3  # filled from data.kind: 3 for images, 1 for audio
    dims: int = 2  # filled from data.kind: 2 for images, 1 for audio
    patch_size: int = 16


@dataclass
class AutoencoderStageConfig:
    model: AutoencoderModelConfig = field(default_factory=AutoencoderModelConfig)
    train: TrainConfig = field(default_factory=lambda: TrainConfig(batch_size=64, microbatch_size=8))
    loss_scale: float = 10.0


# ----------------------------------------------------------------------------- stage 1c: encode


@dataclass
class EncodeConfig:
    # "latest" = newest run under <runs_dir>/<stage>/; or a run name, a run dir, or a step_*.pt file
    vqvae_checkpoint: str = "latest"
    autoencoder_checkpoint: str = "latest"
    batch_size: int = 16
    include_flipped: bool = True  # also store the horizontally flipped version of every image


# ----------------------------------------------------------------------------- stages 2 / 3: transformers


@dataclass
class TransformerModelConfig:
    """Shared by MaskGIT, the GAN generator and the GAN discriminator.

    ``codebook_size``, ``grid_h``, ``grid_w`` and ``bottleneck_dim`` are normally left at
    None and filled in from the encoded dataset at runtime.
    """

    hidden_size: int = 512
    intermediate_size: int = 2048
    num_layers: int = 24
    num_attention_heads: int = 16
    rope_base: float = 100.0  # small base: the grid is only 32 wide, 10000 would waste most frequencies
    codebook_size: int | None = None
    grid_h: int | None = None
    grid_w: int | None = None
    bottleneck_dim: int | None = None  # only used by the GAN (channels of the AE latent)

    @property
    def num_positions(self) -> int:
        return self.grid_h * self.grid_w


@dataclass
class MaskGITStageConfig:
    model: TransformerModelConfig = field(default_factory=lambda: TransformerModelConfig(num_layers=24))
    train: TrainConfig = field(
        default_factory=lambda: TrainConfig(
            batch_size=128,
            microbatch_size=16,
            log_every=50,
            image_every=0,
            optimizer=OptimizerConfig(lr=1e-4, warmup_steps=1000),
        )
    )
    mask_ratio_min: float = 0.3  # each sample masks a uniform-random fraction in [min, max]
    mask_ratio_max: float = 1.0


@dataclass
class CGANStageConfig:
    generator: TransformerModelConfig = field(default_factory=lambda: TransformerModelConfig(num_layers=12))
    discriminator: TransformerModelConfig = field(
        default_factory=lambda: TransformerModelConfig(num_layers=16)
    )
    train: TrainConfig = field(
        default_factory=lambda: TrainConfig(
            batch_size=128,
            microbatch_size=16,
            log_every=50,
            image_every=50,
            optimizer=OptimizerConfig(
                type="adamw", lr=1e-4, betas=(0.5, 0.999), weight_decay=0.0, warmup_steps=100
            ),
        )
    )
    disc_accuracy_threshold: float = (
        0.85  # discriminator trains until this accurate, then the generator trains
    )
    disc_logit_margin: float = 0.5  # a prediction only counts as correct beyond this logit margin
    ema_decay: float = 0.99  # generator EMA: ema = decay * ema + (1 - decay) * weights ...
    ema_every: int = 10  # ... applied every N successful generator steps
    autoencoder_checkpoint: str = (
        "latest"  # frozen AE decoder, for progress images only ("latest" or run/file)
    )


# ----------------------------------------------------------------------------- generation


@dataclass
class GenerateConfig:
    # each: "latest" (newest run of that stage under <runs_dir>), a run name, a run dir, or a step_*.pt file
    maskgit_checkpoint: str = "latest"
    vqvae_checkpoint: str = "latest"
    gan_checkpoint: str = "latest"
    autoencoder_checkpoint: str = "latest"
    out_dir: str = "generated"
    num_images: int = 16  # 0 = run until Ctrl-C
    batch_size: int = 2
    decode: str = "both"  # "vq", "gan" or "both" (side by side)
    use_ema: bool = True  # use the generator's EMA weights when present
    sample_fraction: float = 0.75  # per fill: fraction of masked slots sampled one at a time, rest argmaxed
    keep_schedule: tuple[float, ...] = (0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64)  # re-fill schedule
    pairs_per_row: int = 1


# ----------------------------------------------------------------------------- root


@dataclass
class Config:
    project: ProjectConfig = field(default_factory=ProjectConfig)
    data: DataConfig = field(default_factory=DataConfig)
    vqvae: VQVAEStageConfig = field(default_factory=VQVAEStageConfig)
    autoencoder: AutoencoderStageConfig = field(default_factory=AutoencoderStageConfig)
    encode: EncodeConfig = field(default_factory=EncodeConfig)
    maskgit: MaskGITStageConfig = field(default_factory=MaskGITStageConfig)
    cgan: CGANStageConfig = field(default_factory=CGANStageConfig)
    generate: GenerateConfig = field(default_factory=GenerateConfig)


# ----------------------------------------------------------------------------- dict <-> dataclass


def from_dict(cls: type, data: dict[str, Any] | None, path: str = "") -> Any:
    """Build dataclass ``cls`` from a (possibly partial) dict. Unknown keys raise a clear error.

    Missing keys keep the defaults *of the enclosing section*: ``cgan: {train: {epochs: 5}}``
    keeps the GAN's own optimizer defaults (plain AdamW), not the generic ``TrainConfig`` ones.
    """
    return merge_into(cls(), data, path)


def merge_into(instance: Any, data: dict[str, Any] | None, path: str = "") -> Any:
    """Return a copy of dataclass ``instance`` with the values in ``data`` applied recursively."""
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise TypeError(
            f"config section '{path or type(instance).__name__}' must be a mapping, got {type(data).__name__}"
        )
    known = {f.name for f in fields(instance)}
    unknown = set(data) - known
    if unknown:
        raise KeyError(
            f"unknown config key(s) {sorted(unknown)} in section '{path or '<root>'}'. Valid keys: {sorted(known)}"
        )
    hints = typing.get_type_hints(type(instance))
    changes = {}
    for name, value in data.items():
        current = getattr(instance, name)
        sub_path = f"{path}.{name}" if path else name
        if is_dataclass(current) and not isinstance(current, type):
            changes[name] = merge_into(current, value, sub_path)
        elif isinstance(value, list):
            example = current[0] if isinstance(current, tuple) and current else None
            changes[name] = tuple(_coerce(example, v, sub_path) for v in value)  # YAML lists -> tuples
        else:
            example = current if current is not None else _example_from_hint(hints[name])
            changes[name] = _coerce(example, value, sub_path)
    return dataclasses.replace(instance, **changes)


def _example_from_hint(hint: Any) -> Any:
    """For a default of ``None``, an example value of the annotated type (``int | None`` -> ``0``)."""
    for t in typing.get_args(hint) or (hint,):
        if t is int:
            return 0
        if t is float:
            return 0.0
    return None


def _coerce(current: Any, value: Any, path: str) -> Any:
    """Make ``value`` the same kind of number as the default ``current``.

    YAML (1.1, as PyYAML reads it) does not treat ``1e-4`` as a float, so ``--set lr=1e-4`` arrives as
    the string ``"1e-4"``; this turns it into ``0.0001`` because the default is a float.
    """
    if isinstance(current, bool) or not isinstance(current, (int, float)):
        return value
    if isinstance(value, bool) or value is None:
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise TypeError(f"config '{path}' expects a number like {current!r}, got {value!r}") from None
    if isinstance(current, int):
        if not number.is_integer():
            raise TypeError(f"config '{path}' expects an integer like {current!r}, got {value!r}")
        return int(number)
    return number


def fill_from_data(model_cfg: TransformerModelConfig, **from_data: int) -> TransformerModelConfig:
    """Fill ``codebook_size`` / ``grid_h`` / ``grid_w`` / ``bottleneck_dim`` from the encoded dataset.

    A value already set in the config (e.g. a converted legacy GAN whose embedding has 1024 rows) is
    kept, as long as it is compatible with the data; a grid mismatch is an error.
    """
    changes = {}
    for name, value in from_data.items():
        current = getattr(model_cfg, name)
        if current is None:
            changes[name] = value
        elif name == "codebook_size" and current < value:
            raise ValueError(f"config codebook_size={current} is smaller than the data's {value}")
        elif name != "codebook_size" and current != value:
            raise ValueError(f"config {name}={current} does not match the encoded dataset ({value})")
    return dataclasses.replace(model_cfg, **changes)


def to_dict(obj: Any) -> Any:
    """Dataclass -> plain dict (tuples become lists so the result is clean YAML/JSON)."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_dict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [to_dict(v) for v in obj]
    if isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    return obj


def _set_nested(data: dict, dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    node = data
    for k in keys[:-1]:
        node = node.setdefault(k, {})
        if not isinstance(node, dict):
            raise KeyError(f"cannot set '{dotted_key}': '{k}' is not a section")
    node[keys[-1]] = value


def apply_overrides(data: dict, overrides: list[str] | None) -> dict:
    """Apply ``["a.b.c=1", "x.y=[1,2]"]`` style overrides. Values are parsed as YAML."""
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override '{item}' must look like section.key=value")
        key, raw = item.split("=", 1)
        _set_nested(data, key.strip(), yaml.safe_load(raw))
    return data


def load_config(path: str | Path | None, overrides: list[str] | None = None) -> Config:
    """Load a YAML file (or just the defaults when ``path`` is None) and apply overrides."""
    data: dict = {}
    if path is not None:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
    data = apply_overrides(data, overrides)
    return from_dict(Config, data)


def save_config(cfg: Config, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(to_dict(cfg), f, sort_keys=False)


def replace(obj: Any, **changes: Any) -> Any:
    """``dataclasses.replace`` re-exported so callers only import this module."""
    return dataclasses.replace(obj, **changes)
