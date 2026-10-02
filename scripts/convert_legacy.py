#!/usr/bin/env python
"""
Convert files written by the ORIGINAL scripts into the formats this repo uses.

The model weights are byte-for-byte the same (parameter names were kept), only the container
changed, so conversion is loss-free and takes seconds.

Checkpoints (one per stage; the GAN had two files, generator + discriminator):

    python scripts/convert_legacy.py checkpoint --stage vqvae \\
        --input models/vqvae_stage_1a_bcb5a61d_vq_32x32_epoch_5_step_6021_vqvae.pth
    python scripts/convert_legacy.py checkpoint --stage autoencoder --input models/ae_stage_1b_....pth
    python scripts/convert_legacy.py checkpoint --stage maskgit     --input models/6699378d_maskgit_24_....pth
    python scripts/convert_legacy.py checkpoint --stage cgan \\
        --input /mnt/f/models_trained/97f294f7_76410/97f294f7_gen_12_epoch_139_step_76410_generator.pth \\
        --discriminator /mnt/f/models_trained/97f294f7_76410/97f294f7_disc_16_epoch_139_step_76410_discriminator.pth

    Output defaults to runs/<stage>/legacy/checkpoints/step_<N>.pt, i.e. `runs/<stage>/latest` finds it.
    The VQ-VAE / autoencoder architecture is read from --config (the originals did not store it).

Encoded dataset (the old all_vq_codes.pt + all_latents.pt pair):

    python scripts/convert_legacy.py dataset --codes data/all_vq_codes.pt --latents data/all_latents.pt
"""

import argparse
import sys
from pathlib import Path

# make `latentgen` importable without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from latentgen.config import TransformerModelConfig, load_config, replace, save_config, to_dict  # noqa: E402
from latentgen.data import ImageStats, save_encoded  # noqa: E402
from latentgen.nn import VQVAE, Autoencoder, Discriminator, Generator, MaskGIT  # noqa: E402
from latentgen.training.checkpoint import run_dir_for, save_checkpoint  # noqa: E402

STAGES = ("vqvae", "autoencoder", "maskgit", "cgan")


def _load(path: str) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "model_state_dict" not in ckpt:
        raise SystemExit(f"{path} does not look like a legacy checkpoint (no 'model_state_dict')")
    return ckpt


def _check_weights(model: torch.nn.Module, state: dict, what: str) -> None:
    """Strict load into a freshly built model: proves the config matches the weights."""
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as e:
        raise SystemExit(
            f"{what}: the weights do not match the model built from the config.\n{e}\n"
            f"Check the model section of your --config (sizes, layers, codebook)."
        ) from e


def _transformer_config_from_weights(
    sd: dict, num_heads: int, bottleneck_dim: int | None
) -> TransformerModelConfig:
    """The original GAN checkpoints stored no config; recover it from the weight shapes."""
    codebook_size, hidden = sd["lowres_quantized_code_embedding.weight"].shape
    return TransformerModelConfig(
        hidden_size=hidden,
        intermediate_size=sd["layers.0.mlp.down_proj.weight"].shape[1],
        num_layers=len({k.split(".")[1] for k in sd if k.startswith("layers.")}),
        num_attention_heads=num_heads,
        codebook_size=codebook_size,
        bottleneck_dim=bottleneck_dim,
    )


def _optimizer_state(ckpt: dict, opt_type: str, drop: bool) -> dict | None:
    if drop or "optimizer_state_dict" not in ckpt:
        return None
    return {
        "type": opt_type,
        "optimizer": ckpt["optimizer_state_dict"],
        "scheduler": ckpt.get("scheduler_state_dict") if opt_type == "adamw" else None,
    }


def convert_checkpoint(args: argparse.Namespace) -> None:
    cfg = load_config(args.config, args.set)
    stats = ImageStats.load(cfg.data.stats_file, cfg.data.kind)
    ckpt = _load(args.input)
    sd = ckpt["model_state_dict"]
    seen = ckpt.get("seen_ids") or (ckpt.get("loader_state") or {}).get("seen_ids") or []
    progress = {
        "epoch": int(ckpt["epoch"]),
        "step": int(ckpt["steps"]),
        "microbatches": int(ckpt.get("total_microbatches", 0)),
        "seen_ids": list(seen),
    }
    stage_state: dict = {}
    models: dict = {}

    if args.stage == "vqvae":
        _check_weights(VQVAE(cfg.vqvae.model), sd, "VQ-VAE")
        models["vqvae"] = {
            "model": sd,
            "optimizer": _optimizer_state(ckpt, "adamw_schedulefree", args.drop_optimizer),
            "ema": None,
            "step_count": progress["step"],
        }
        print("codebook cut over:", bool(sd["bottleneck.codebook.cutover"]))

    elif args.stage == "autoencoder":
        _check_weights(Autoencoder(cfg.autoencoder.model), sd, "autoencoder")
        models["autoencoder"] = {
            "model": sd,
            "optimizer": _optimizer_state(ckpt, "adamw_schedulefree", args.drop_optimizer),
            "ema": None,
            "step_count": progress["step"],
        }

    elif args.stage == "maskgit":
        legacy_cfg = dict(ckpt["config"])  # the original stage-2 script did store its config
        legacy_cfg.pop("num_positions", None)
        cfg.maskgit.model = replace(cfg.maskgit.model, **legacy_cfg)
        _check_weights(MaskGIT(cfg.maskgit.model), sd, "MaskGIT")
        models["maskgit"] = {
            "model": sd,
            "optimizer": _optimizer_state(ckpt, "adamw_schedulefree", args.drop_optimizer),
            "ema": None,
            "step_count": int(ckpt.get("local_steps", progress["step"])),
        }

    elif args.stage == "cgan":
        if not args.discriminator:
            raise SystemExit(
                "--stage cgan needs --input <generator.pth> and --discriminator <discriminator.pth>"
            )
        d_ckpt = _load(args.discriminator)
        d_sd = d_ckpt["model_state_dict"]
        bottleneck_dim = sd["final_out.weight"].shape[0]
        gen_cfg = _transformer_config_from_weights(sd, args.num_heads, bottleneck_dim)
        disc_cfg = _transformer_config_from_weights(d_sd, args.num_heads, bottleneck_dim)
        grid = args.grid
        gen_cfg = replace(gen_cfg, grid_h=grid, grid_w=grid, rope_base=cfg.cgan.generator.rope_base)
        disc_cfg = replace(disc_cfg, grid_h=grid, grid_w=grid, rope_base=cfg.cgan.discriminator.rope_base)
        cfg.cgan.generator, cfg.cgan.discriminator = gen_cfg, disc_cfg
        _check_weights(Generator(gen_cfg), sd, "generator")
        _check_weights(Discriminator(disc_cfg), d_sd, "discriminator")
        ema = ckpt.get("ema_state_dict")
        models["generator"] = {
            "model": sd,
            "optimizer": _optimizer_state(ckpt, "adamw", args.drop_optimizer),
            "ema": ema,
            "step_count": int(ckpt.get("local_steps", 0)),
        }
        models["discriminator"] = {
            "model": d_sd,
            "optimizer": _optimizer_state(d_ckpt, "adamw", args.drop_optimizer),
            "ema": None,
            "step_count": int(d_ckpt.get("local_steps", 0)),
        }
        stage_state = {"disc_is_good": bool(d_ckpt.get("is_in_good_state", False))}
        print(
            f"generator: {gen_cfg.num_layers} layers, EMA {'present' if ema else 'absent'}; "
            f"discriminator: {disc_cfg.num_layers} layers"
        )

    out = (
        Path(args.output)
        if args.output
        else (
            run_dir_for(cfg.project.runs_dir, args.stage, "legacy")
            / "checkpoints"
            / f"step_{progress['step']:08d}.pt"
        )
    )
    if (
        not args.output
    ):  # default layout runs/<stage>/legacy/checkpoints/: give the run its config.yaml for --resume
        save_config(cfg, out.parent.parent / "config.yaml")
    save_checkpoint(
        out,
        {
            "stage": args.stage,
            "config": to_dict(cfg),
            "stats": stats.to_dict(),
            "models": models,
            "progress": progress,
            "stage_state": stage_state,
        },
    )
    print(f"Wrote {out}  (epoch {progress['epoch']}, step {progress['step']})")


def convert_dataset(args: argparse.Namespace) -> None:
    codes = torch.load(args.codes, map_location="cpu", weights_only=False)
    if not torch.is_tensor(codes) or codes.dim() != 4:
        raise SystemExit(f"{args.codes}: expected a [V, N, H, W] tensor")
    latents = None
    if args.latents:
        latents = torch.load(args.latents, map_location="cpu", weights_only=False)
        if not torch.is_tensor(latents) or latents.dim() != 5 or latents.dtype != torch.int8:
            raise SystemExit(f"{args.latents}: expected an int8 [V, N, C, H, W] tensor")
    cfg = load_config(args.config, args.set)
    stats = ImageStats.load(args.stats or cfg.data.stats_file)
    codebook_size = args.codebook_size or cfg.vqvae.model.codebook_size
    if int(codes.max()) >= codebook_size:
        raise SystemExit(f"codes go up to {int(codes.max())} but codebook_size is {codebook_size}")
    meta = {
        "codebook_size": codebook_size,
        "image_size": args.image_size,
        "patch_size": args.image_size // codes.shape[-1],
        "latent_dim": None if latents is None else latents.shape[2],
        "files": [],
        "source": "convert_legacy",
    }
    save_encoded(args.output, codes, latents, stats, meta)
    print(
        f"Wrote {args.output}: codes {tuple(codes.shape)}, latents {None if latents is None else tuple(latents.shape)}, "
        f"codebook {codebook_size}"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("checkpoint", help="convert a legacy .pth checkpoint")
    c.add_argument("--stage", choices=STAGES, required=True)
    c.add_argument("--input", required=True, help="legacy .pth (for cgan: the generator file)")
    c.add_argument("--discriminator", default=None, help="legacy discriminator .pth (cgan only)")
    c.add_argument("--output", default=None, help="default: runs/<stage>/legacy/checkpoints/step_<N>.pt")
    c.add_argument("--config", default="configs/ffhq512.yaml", help="config describing the model sizes")
    c.add_argument("--set", nargs="*", action="extend", default=[], metavar="KEY=VALUE")
    c.add_argument("--drop-optimizer", action="store_true", help="do not carry the optimizer state over")
    c.add_argument(
        "--num-heads", type=int, default=16, help="cgan only: heads are not recoverable from weights"
    )
    c.add_argument("--grid", type=int, default=32, help="cgan only: token grid side")
    c.set_defaults(func=convert_checkpoint)

    d = sub.add_parser("dataset", help="merge legacy all_vq_codes.pt / all_latents.pt into one encoded file")
    d.add_argument("--codes", required=True)
    d.add_argument("--latents", default=None)
    d.add_argument("--output", default="data/encoded.pt")
    d.add_argument("--config", default="configs/ffhq512.yaml", help="config the codes were made with")
    d.add_argument("--set", nargs="*", action="extend", default=[], metavar="KEY=VALUE")
    d.add_argument("--stats", default=None, help="stats JSON used when encoding (default: from --config)")
    d.add_argument(
        "--codebook-size", type=int, default=None, help="default: vqvae.model.codebook_size from --config"
    )
    d.add_argument("--image-size", type=int, default=512)
    d.set_defaults(func=convert_dataset)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
