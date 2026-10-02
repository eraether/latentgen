#!/usr/bin/env python
"""
Stage 3 -- train the conditional GAN that maps VQ codes to a detailed AE latent.

    python scripts/03_train_cgan.py --config configs/ffhq512.yaml
    python scripts/03_train_cgan.py --resume runs/cgan/latest

Reads both coarse_encoded.pt and fine_encoded.pt from data.encoded_dir. The frozen autoencoder (cgan.autoencoder_checkpoint)
is only used to decode progress images: runs/cgan/<run>/images/ shows real | generator | EMA generator.
Independent of stage 2: trains at the same time as MaskGIT if you have the GPU for it.
"""

import sys
from pathlib import Path

# make `latentgen` importable without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from latentgen.cli import build_parser, checkpoint_for, choose_run_dir, load_cfg, setup  # noqa: E402
from latentgen.config import fill_from_data  # noqa: E402
from latentgen.data import open_encoded  # noqa: E402
from latentgen.pretrained import load_autoencoder  # noqa: E402
from latentgen.stages import CGANStage  # noqa: E402
from latentgen.training import Trainer  # noqa: E402


def main() -> None:
    args = build_parser(__doc__).parse_args()
    cfg = load_cfg(args)
    device = setup(cfg)

    data = open_encoded(
        cfg.data.encoded_dir,
        device,
        need_latents=True,
        storage=cfg.data.encoded_device,
        loading=cfg.data.encoded_loading,
        prefetch_batches=cfg.data.prefetch_batches,
    )
    shape = dict(
        codebook_size=data.codebook_size,
        grid_h=data.grid_h,
        grid_w=data.grid_w,
        bottleneck_dim=data.latent_dim,
    )
    cfg.cgan.generator = fill_from_data(cfg.cgan.generator, **shape)
    cfg.cgan.discriminator = fill_from_data(cfg.cgan.discriminator, **shape)

    autoencoder = None
    if cfg.cgan.train.image_every:
        try:
            autoencoder, _ = load_autoencoder(
                checkpoint_for(cfg, cfg.cgan.autoencoder_checkpoint, "autoencoder"), device
            )
        except (FileNotFoundError, ValueError) as e:
            print(
                f"WARNING: cgan.autoencoder_checkpoint={cfg.cgan.autoencoder_checkpoint!r} could not be loaded "
                f"({e}). Training continues WITHOUT progress images; fix the path or set cgan.train.image_every=0."
            )

    stage = CGANStage(cfg, device, data, data.stats, autoencoder)
    run_dir, resume_from = choose_run_dir(cfg, stage.name, args)
    trainer = Trainer(cfg, stage, run_dir, device, pdb_on_interrupt=args.pdb)
    if resume_from:
        trainer.resume(resume_from)
    trainer.run()


if __name__ == "__main__":
    main()
