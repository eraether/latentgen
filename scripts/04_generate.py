#!/usr/bin/env python
"""
Generate images: sample code grids with MaskGIT, decode them with the VQ-VAE and/or GAN + AE.

    python scripts/04_generate.py --config configs/ffhq512.yaml
    python scripts/04_generate.py --set generate.num_images=64 generate.decode=gan generate.batch_size=8
    python scripts/04_generate.py --set generate.num_images=0        # run until Ctrl-C

Each batch is saved as one PNG in generate.out_dir. With decode=both every row is [VQ | GAN] pairs,
so you can see what the GAN adds on top of the blurry VQ reconstruction of the same code grid.
For audio datasets (data.kind: audio) every sample is written as .wav files instead, plus a waveform PNG.
"""

import sys
from datetime import datetime
from pathlib import Path

# make `latentgen` importable without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

from latentgen.cli import build_parser, checkpoint_for, load_cfg, setup  # noqa: E402
from latentgen.data import save_wav, waveform_image  # noqa: E402
from latentgen.device import maybe_compile  # noqa: E402
from latentgen.pretrained import load_autoencoder, load_generator, load_maskgit, load_vqvae  # noqa: E402
from latentgen.sampling import decode_gan, decode_vq, fill_plan, forward_passes, generate_codes  # noqa: E402
from latentgen.training.images import image_grid, save_image, tile  # noqa: E402


def main() -> None:
    args = build_parser(__doc__, training=False).parse_args()
    cfg = load_cfg(args)
    g = cfg.generate
    if g.decode not in ("vq", "gan", "both"):
        raise SystemExit("generate.decode must be 'vq', 'gan' or 'both'")
    device = setup(cfg)

    maskgit, stats = load_maskgit(checkpoint_for(cfg, g.maskgit_checkpoint, "maskgit"), device)
    vqvae = generator = autoencoder = None
    if g.decode in ("vq", "both"):
        vqvae, _ = load_vqvae(checkpoint_for(cfg, g.vqvae_checkpoint, "vqvae"), device)
    if g.decode in ("gan", "both"):
        generator, _ = load_generator(
            checkpoint_for(cfg, g.gan_checkpoint, "cgan"), device, use_ema=g.use_ema
        )
        autoencoder, _ = load_autoencoder(
            checkpoint_for(cfg, g.autoencoder_checkpoint, "autoencoder"), device
        )
        if generator.cfg.codebook_size < maskgit.cfg.codebook_size:
            raise SystemExit("MaskGIT can emit codes the GAN generator has no embedding for")
        if (generator.cfg.grid_h, generator.cfg.grid_w) != (maskgit.cfg.grid_h, maskgit.cfg.grid_w):
            raise SystemExit("MaskGIT and GAN generator grids differ")
    maskgit.forward = maybe_compile(maskgit.forward, cfg.project.compile)

    plan = fill_plan(maskgit.cfg.num_positions, g.sample_fraction, g.keep_schedule)
    passes = forward_passes(plan)
    print(f"{len(plan)} fills per batch, {passes} forward passes per batch:")
    for keep, n_sample, n_argmax in plan:
        print(
            f"  keep {100 * keep:5.1f}% -> sample {n_sample} slots one at a time, argmax the remaining {n_argmax}"
        )

    out_dir = Path(g.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = g.num_images if g.num_images > 0 else None
    produced, batch_idx = 0, 0
    overall = tqdm(total=target, desc="images", unit="img", position=0)
    try:
        while target is None or produced < target:
            batch = g.batch_size if target is None else min(g.batch_size, target - produced)
            step_bar = tqdm(total=passes, desc=f"batch {batch_idx}", unit="fwd", position=1, leave=False)
            codes = generate_codes(
                maskgit, batch, g.sample_fraction, g.keep_schedule, on_pass=lambda bar=step_bar: bar.update(1)
            )
            step_bar.close()

            columns = []
            if vqvae is not None:
                columns.append(decode_vq(codes, vqvae, stats))
            if generator is not None:
                columns.append(decode_gan(codes, generator, autoencoder, stats))
            path = out_dir / f"{run}_{batch_idx:05d}.png"
            if cfg.data.kind == "audio":
                # one .wav per sample and decoder, plus a waveform picture of the first sample per decoder
                names = [n for n, m in (("vq", vqvae), ("gan", generator)) if m is not None]
                for i in range(batch):
                    for name, out in zip(names, columns, strict=True):
                        save_wav(
                            out_dir / f"{run}_{batch_idx:05d}_{i}_{name}.wav", out[i], cfg.data.sample_rate
                        )
                save_image(torch.cat([waveform_image(c[0]) for c in columns], dim=-2), path)
            else:
                # one tile per sampled grid: [VQ | GAN] side by side, `pairs_per_row` tiles per row
                tiles = [image_grid([c[i : i + 1] for c in columns], max_rows=1) for i in range(batch)]
                save_image(tile(tiles, max(1, g.pairs_per_row)), path)
            torch.save(
                codes.cpu(), path.with_suffix(".codes.pt")
            )  # the code grids, in case you want to re-decode

            produced += batch
            batch_idx += 1
            overall.update(batch)
            overall.set_postfix(file=path.name)
    except KeyboardInterrupt:
        pass
    overall.close()
    print(f"\nWrote {produced} images in {batch_idx} files to {out_dir}")


if __name__ == "__main__":
    main()
