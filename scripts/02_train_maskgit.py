#!/usr/bin/env python
"""
Stage 2 -- train MaskGIT, the generative model over VQ code grids.

    python scripts/02_train_maskgit.py --config configs/ffhq512.yaml
    python scripts/02_train_maskgit.py --resume runs/maskgit/latest

Reads data.encoded_file (stage 1c). The grid size and vocabulary come from that file, so the only
things to choose are the transformer size (maskgit.model.*) and the training knobs.
The console shows cross-entropy / accuracy per mask-ratio bucket at every log.
"""

import sys
from pathlib import Path

# make `latentgen` importable without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from latentgen.cli import build_parser, choose_run_dir, load_cfg, setup  # noqa: E402
from latentgen.config import fill_from_data  # noqa: E402
from latentgen.data import EncodedDataset  # noqa: E402
from latentgen.stages import MaskGITStage  # noqa: E402
from latentgen.training import Trainer  # noqa: E402


def main() -> None:
    args = build_parser(__doc__).parse_args()
    cfg = load_cfg(args)
    device = setup(cfg)

    data = EncodedDataset(cfg.data.encoded_file, device, storage=cfg.data.encoded_device, need_latents=False)
    # grid / vocabulary are properties of the data, not choices: fill them in from the encoded file
    cfg.maskgit.model = fill_from_data(
        cfg.maskgit.model, codebook_size=data.codebook_size, grid_h=data.grid_h, grid_w=data.grid_w
    )

    stage = MaskGITStage(cfg, device, data, data.stats)
    run_dir, resume_from = choose_run_dir(cfg, stage.name, args)
    trainer = Trainer(cfg, stage, run_dir, device, pdb_on_interrupt=args.pdb)
    if resume_from:
        trainer.resume(resume_from)
    trainer.run()


if __name__ == "__main__":
    main()
