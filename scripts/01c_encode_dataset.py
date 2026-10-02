#!/usr/bin/env python
"""
Stage 1c -- run every image through the trained VQ-VAE and autoencoder once and store the results.

    python scripts/01c_encode_dataset.py --config configs/ffhq512.yaml
    python scripts/01c_encode_dataset.py --set encode.vqvae_checkpoint=runs/vqvae/20240101_120000_ab12cd

Writes two files to data.encoded_dir (default data/encoded/), for every image and its horizontal flip:
    coarse_encoded.pt   int16 [N, V, 32, 32]       the VQ-VAE code grids   -> stage 2 (MaskGIT) and stage 3
    fine_encoded.pt     int8  [N, V, 32, 32, 32]   the AE latents * 127    -> stage 3 (cGAN) only
When the fine latents would exceed encode.max_file_gb, both are written as chunks of ~encode.chunk_gb
instead (items in random order), and stages 2/3 stream them from disk. Either way stages 2 and 3 never
touch the images again, which is why they are so fast.
"""

import sys
from pathlib import Path

# make `latentgen` importable without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from tqdm import tqdm  # noqa: E402

from latentgen.cli import build_parser, checkpoint_for, load_cfg, setup  # noqa: E402
from latentgen.data import LATENT_SCALE, EncodedWriter, image_dataset_from_config  # noqa: E402
from latentgen.data.encoded import GB  # noqa: E402
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
        raise SystemExit("the VQ-VAE and the autoencoder were trained with different normalization stats")
    if vqvae.cfg.patch_size != autoencoder.cfg.patch_size:
        raise SystemExit(
            f"the VQ-VAE (patch_size {vqvae.cfg.patch_size}) and the autoencoder (patch_size "
            f"{autoencoder.cfg.patch_size}) must produce the same grid"
        )
    encode_codes = maybe_compile(vqvae.encode, cfg.project.compile)
    encode_latents = maybe_compile(autoencoder.encode, cfg.project.compile)

    dataset = image_dataset_from_config(cfg, stats, horizontal_flip=False)  # flips are stored explicitly
    N = len(dataset)
    V = 2 if cfg.encode.include_flipped else 1
    audio = cfg.data.kind == "audio"

    # single file or chunks? decided up front from the latent size, because chunks want shuffled input
    p = vqvae.cfg.patch_size
    grid = (1, cfg.data.audio_length // p) if audio else (cfg.data.image_size // p,) * 2
    item_bytes = V * autoencoder.cfg.bottleneck_dim * grid[0] * grid[1]  # int8 fine latent per item
    fine_gb = N * item_bytes / GB
    chunk_items = None
    order = None
    if fine_gb > cfg.encode.max_file_gb:
        chunk_items = max(1, int(cfg.encode.chunk_gb * GB // item_bytes))
        # random order: every chunk becomes a random subset, so streaming chunk by chunk stays well mixed
        order = torch.randperm(N, generator=torch.Generator().manual_seed(0)).tolist()
        print(
            f"Fine latents {fine_gb:.1f} GB > encode.max_file_gb={cfg.encode.max_file_gb}: writing "
            f"{-(-N // chunk_items)} chunks of {chunk_items} items (stages 2/3 will stream them)"
        )
    loader = DataLoader(
        dataset,
        batch_size=cfg.encode.batch_size,
        sampler=order,  # None = file order
        drop_last=False,
        num_workers=cfg.data.num_workers,
        pin_memory=device.type == "cuda",
    )
    writer = EncodedWriter(cfg.data.encoded_dir, N, chunk_items)
    print(f"Encoding {N} items x {V} variants ({fine_gb:.2f} GB of fine latents) into {cfg.data.encoded_dir}")

    used = torch.zeros(vqvae.cfg.codebook_size, dtype=torch.bool)
    with torch.no_grad():
        for inputs, ids in tqdm(loader, unit="batch"):
            inputs = inputs.to(device, non_blocking=True)
            # the stored augmentation: mirrored image, or inverted polarity for audio
            variants = [inputs, -inputs if audio else inputs.flip(-1)] if V == 2 else [inputs]
            codes, latents = [], []
            for batch in variants:
                with autocast(device):
                    codes.append(encode_codes(batch).to(torch.int16).cpu())
                    latents.append(
                        (encode_latents(batch).float() * LATENT_SCALE).round().to(torch.int8).cpu()
                    )
            codes = torch.stack(codes, dim=1)  # [b, V, H, W]
            used[codes.flatten().long()] = True
            writer.add(ids, codes, torch.stack(latents, dim=1))  # latents [b, V, C, H, W]

    meta = {
        "codebook_size": vqvae.cfg.codebook_size,
        "kind": cfg.data.kind,
        "image_size": cfg.data.image_size,
        "audio_length": cfg.data.audio_length,
        "sample_rate": cfg.data.sample_rate,
        "patch_size": p,
        "latent_dim": autoencoder.cfg.bottleneck_dim,
        "files": [str(f) for f in dataset.files],
        "vqvae_checkpoint": str(vqvae_path),
        "autoencoder_checkpoint": str(ae_path),
    }
    summary = writer.finish(stats, meta)
    print(
        f"Saved {cfg.data.encoded_dir}: {summary}  ({int(used.sum())}/{vqvae.cfg.codebook_size} codes in use)"
    )


if __name__ == "__main__":
    main()
