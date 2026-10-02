#!/usr/bin/env python
"""
Optional stage 0 -- per-channel mean / std of your image (or audio) folder, for normalization.

    python scripts/00_compute_dataset_stats.py --config configs/my_dataset.yaml
    python scripts/00_compute_dataset_stats.py --set data.image_dir=/path/to/images --out data/my_stats.json

Then set `data.stats_file: data/my_stats.json` in the config. Skipping this and using the FFHQ
defaults is fine for any natural-image dataset; it only matters when the color statistics are
far from a typical photo (e.g. medical scans, line art).
"""

import sys
from pathlib import Path

# make `latentgen` importable without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from tqdm import tqdm  # noqa: E402

from latentgen.cli import build_parser, load_cfg  # noqa: E402
from latentgen.data import ImageStats, image_dataset_from_config  # noqa: E402


def main() -> None:
    parser = build_parser(__doc__, training=False)
    parser.add_argument("--out", default="data/dataset_stats.json", help="where to write the JSON")
    parser.add_argument("--max-images", type=int, default=10000, help="subsample for speed (0 = all)")
    args = parser.parse_args()
    cfg = load_cfg(args)

    channels = 1 if cfg.data.kind == "audio" else 3
    identity = ImageStats(mean=(0.0,) * channels, std=(1.0,) * channels)  # read raw pixels / samples
    dataset = image_dataset_from_config(cfg, identity, horizontal_flip=False)
    n = len(dataset)
    if args.max_images and n > args.max_images:
        keep = torch.randperm(n)[: args.max_images]
        dataset.files = [dataset.files[i] for i in keep.tolist()]
    loader = DataLoader(dataset, batch_size=32, num_workers=cfg.data.num_workers)

    total = torch.zeros(channels, dtype=torch.float64)
    total_sq = torch.zeros(channels, dtype=torch.float64)
    count = 0
    for batch, _ in tqdm(loader, desc="items", unit="batch"):
        x = batch.double()
        dims = (0, *range(2, x.dim()))  # everything except the channel dim
        total += x.sum(dim=dims)
        total_sq += x.pow(2).sum(dim=dims)
        count += x.numel() // x.shape[1]
    mean = total / count
    std = (total_sq / count - mean.pow(2)).sqrt()
    stats = ImageStats(mean=tuple(mean.tolist()), std=tuple(std.tolist()))
    stats.save(args.out)
    print(f"mean {stats.mean}\nstd  {stats.std}\nwritten to {args.out} -> set data.stats_file: {args.out}")


if __name__ == "__main__":
    main()
