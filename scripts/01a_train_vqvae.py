#!/usr/bin/env python
"""
Stage 1a -- train the VQ-VAE (image <-> 32x32 grid of discrete codes).

    python scripts/01a_train_vqvae.py --config configs/ffhq512.yaml
    python scripts/01a_train_vqvae.py --resume runs/vqvae/latest
    python scripts/01a_train_vqvae.py --set vqvae.train.epochs=3 vqvae.model.codebook_size=512

Outputs: runs/vqvae/<run>/checkpoints/step_*.pt, progress images in runs/vqvae/<run>/images/
(left: input, right: reconstruction). Watch PSNR in TensorBoard; ~25 dB is where FFHQ ended up.
"""

import sys
from pathlib import Path

# make `latentgen` importable without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from latentgen.cli import build_parser, choose_run_dir, load_cfg, setup  # noqa: E402
from latentgen.data import ImageBatches, ImageStats, image_dataset_from_config  # noqa: E402
from latentgen.stages import VQVAEStage  # noqa: E402
from latentgen.training import Trainer  # noqa: E402


def main() -> None:
    parser = build_parser(__doc__)
    parser.add_argument(
        "--cutover-now",
        action="store_true",
        help="perform the codebook cutover immediately (e.g. when resuming a pre-cutover run)",
    )
    args = parser.parse_args()
    cfg = load_cfg(args)
    device = setup(cfg)

    stats = ImageStats.load(cfg.data.stats_file, cfg.data.kind)
    dataset = image_dataset_from_config(cfg, stats)
    data = ImageBatches(dataset, device, num_workers=cfg.data.num_workers)
    print(f"{len(dataset)} images in {cfg.data.image_dir}")

    stage = VQVAEStage(cfg, device, data, stats)
    run_dir, resume_from = choose_run_dir(cfg, stage.name, args)
    trainer = Trainer(cfg, stage, run_dir, device, pdb_on_interrupt=args.pdb)
    if resume_from:
        trainer.resume(resume_from)
    if args.cutover_now:
        if not resume_from:
            raise SystemExit(
                "--cutover-now only makes sense together with --resume (cutting over an untrained model is pointless)"
            )
        stage.cutover()
    trainer.run()


if __name__ == "__main__":
    main()
