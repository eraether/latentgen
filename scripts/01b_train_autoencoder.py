#!/usr/bin/env python
"""
Stage 1b -- train the continuous autoencoder (image <-> 32x32x32 latent in [-1, 1]).

    python scripts/01b_train_autoencoder.py --config configs/ffhq512.yaml
    python scripts/01b_train_autoencoder.py --resume runs/autoencoder/latest

Independent of stage 1a: the two can train at the same time on two GPUs / machines.
Outputs: runs/autoencoder/<run>/ (same layout as stage 1a).
"""

import sys
from pathlib import Path

# make `latentgen` importable without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from latentgen.cli import build_parser, choose_run_dir, load_cfg, setup  # noqa: E402
from latentgen.data import ImageBatches, ImageStats, image_dataset_from_config  # noqa: E402
from latentgen.stages import AutoencoderStage  # noqa: E402
from latentgen.training import Trainer  # noqa: E402


def main() -> None:
    args = build_parser(__doc__).parse_args()
    cfg = load_cfg(args)
    device = setup(cfg)

    stats = ImageStats.load(cfg.data.stats_file, cfg.data.kind)
    dataset = image_dataset_from_config(cfg, stats)
    data = ImageBatches(dataset, device, num_workers=cfg.data.num_workers)
    print(f"{len(dataset)} images in {cfg.data.image_dir}")

    stage = AutoencoderStage(cfg, device, data, stats)
    run_dir, resume_from = choose_run_dir(cfg, stage.name, args)
    trainer = Trainer(cfg, stage, run_dir, device, pdb_on_interrupt=args.pdb)
    if resume_from:
        trainer.resume(resume_from)
    trainer.run()


if __name__ == "__main__":
    main()
