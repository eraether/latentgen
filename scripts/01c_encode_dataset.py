#!/usr/bin/env python
"""
Stage 1c -- run every image through the trained VQ-VAE and autoencoder once and store the results.

    python scripts/01c_encode_dataset.py --config configs/ffhq512.yaml
    python scripts/01c_encode_dataset.py --set encode.vqvae_checkpoint=runs/vqvae/20240101_120000_ab12cd

Writes data.encoded_file (default data/encoded.pt) with, for every image (and its horizontal flip):
    codes    int16 [V, N, 32, 32]       the VQ-VAE code grid       -> stage 2 (MaskGIT) and stage 3
    latents  int8  [V, N, 32, 32, 32]   the AE latent * 127        -> stage 3 (cGAN)
Stages 2 and 3 never touch the images again, which is why they are so fast.
"""

import sys
from pathlib import Path

# make `latentgen` importable without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from tqdm import tqdm  # noqa: E402

from latentgen.cli import build_parser, checkpoint_for, load_cfg, setup  # noqa: E402
from latentgen.data import LATENT_SCALE, image_dataset_from_config, save_encoded  # noqa: E402
from latentgen.device import autocast, maybe_compile  # noqa: E402
from latentgen.pretrained import load_autoencoder, load_vqvae  # noqa: E402


def main() -> None:
    args = build_parser(__doc__, training=False).parse_args()
    cfg = load_cfg(args)
    device = setup(cfg)

    vqvae_path = checkpoint_for(cfg, cfg.encode.vqvae_checkpoint, "vqvae")
    ae_path = checkpoint_for(cfg, cfg.encode.autoencoder_checkpoint, "autoencoder")
    vqvae, stats = load_vqvae(vqvae_path, device)
    autoencoder, ae_stats = load_autoencoder(ae_path, device)
    if ae_stats != stats:
        raise SystemExit("the VQ-VAE and the autoencoder were trained with different normalisation stats")
    if vqvae.cfg.patch_size != autoencoder.cfg.patch_size:
        raise SystemExit(
            f"the VQ-VAE (patch_size {vqvae.cfg.patch_size}) and the autoencoder (patch_size "
            f"{autoencoder.cfg.patch_size}) must produce the same grid"
        )
    encode_codes = maybe_compile(vqvae.encode, cfg.project.compile)
    encode_latents = maybe_compile(autoencoder.encode, cfg.project.compile)

    dataset = image_dataset_from_config(
        cfg, stats, horizontal_flip=False
    )  # flips are stored explicitly below
    loader = DataLoader(
        dataset,
        batch_size=cfg.encode.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=cfg.data.num_workers,
        pin_memory=device.type == "cuda",
    )

    N = len(dataset)
    V = 2 if cfg.encode.include_flipped else 1
    audio = cfg.data.kind == "audio"
    codes_out = latents_out = None  # allocated from the first batch's shapes (grid is 1 x T/p for audio)
    print(f"Encoding {N} items x {V} variants")

    write = 0
    with torch.no_grad():
        for inputs, _ids in tqdm(loader, unit="batch"):
            inputs = inputs.to(device, non_blocking=True)
            # the stored augmentation: mirrored image, or inverted polarity for audio
            variants = [inputs, -inputs if audio else inputs.flip(-1)] if V == 2 else [inputs]
            for v, batch in enumerate(variants):
                with autocast(device):
                    codes = encode_codes(batch)
                    latents = encode_latents(batch)
                if codes_out is None:
                    codes_out = torch.zeros((V, N, *codes.shape[1:]), dtype=torch.int16)
                    latents_out = torch.zeros((V, N, *latents.shape[1:]), dtype=torch.int8)
                    print(f"codes {tuple(codes_out.shape)}, latents {tuple(latents_out.shape)}")
                codes_out[v, write : write + len(batch)] = codes.to(torch.int16).cpu()
                latents_out[v, write : write + len(batch)] = (
                    (latents.float() * LATENT_SCALE).round().to(torch.int8).cpu()
                )
            write += len(inputs)

    meta = {
        "codebook_size": vqvae.cfg.codebook_size,
        "kind": cfg.data.kind,
        "image_size": cfg.data.image_size,
        "audio_length": cfg.data.audio_length,
        "sample_rate": cfg.data.sample_rate,
        "patch_size": vqvae.cfg.patch_size,
        "latent_dim": autoencoder.cfg.bottleneck_dim,
        "files": [str(p) for p in dataset.files],
        "vqvae_checkpoint": str(vqvae_path),
        "autoencoder_checkpoint": str(ae_path),
    }
    save_encoded(cfg.data.encoded_file, codes_out, latents_out, stats, meta)
    used = codes_out.unique().numel()
    print(f"Saved {cfg.data.encoded_file}  ({used}/{vqvae.cfg.codebook_size} codes in use)")


if __name__ == "__main__":
    main()
