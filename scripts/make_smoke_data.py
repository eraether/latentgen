#!/usr/bin/env python
"""
Write a handful of synthetic images so configs/smoke_test.yaml can run without a real dataset.

    python scripts/make_smoke_data.py                 # 64 images, 64x64, into data/smoke/images
    python scripts/make_smoke_data.py --count 200 --size 128 --out data/smoke/images
"""

import argparse
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw


def make_audio(args: argparse.Namespace) -> None:
    """A few hundred samples of decaying sine tones with noise: structured enough to reconstruct."""
    import struct
    import wave

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    length = 1024
    for i in range(args.count):
        freq = rng.uniform(0.01, 0.1)  # cycles per sample
        samples = [
            0.6 * math.sin(2 * math.pi * freq * t) * math.exp(-t / length) + rng.gauss(0, 0.02)
            for t in range(length)
        ]
        with wave.open(str(out / f"smoke_{i:04d}.wav"), "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(16000)
            f.writeframes(b"".join(struct.pack("<h", int(max(-1, min(1, s)) * 32767)) for s in samples))
    print(f"wrote {args.count} clips to {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="data/smoke/images")
    p.add_argument("--count", type=int, default=64)
    p.add_argument("--size", type=int, default=64)
    p.add_argument(
        "--audio",
        action="store_true",
        help="write 16-bit wav clips (for configs/smoke_test_audio.yaml) instead",
    )
    args = p.parse_args()
    if args.audio:
        return make_audio(args)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    for i in range(args.count):
        # smooth background + a few shapes: structured enough that reconstruction loss means something
        img = Image.new("RGB", (args.size, args.size), tuple(rng.randrange(256) for _ in range(3)))
        draw = ImageDraw.Draw(img)
        for _ in range(rng.randrange(2, 6)):
            x, y = rng.randrange(args.size), rng.randrange(args.size)
            r = rng.randrange(args.size // 8, args.size // 2)
            color = tuple(rng.randrange(256) for _ in range(3))
            if rng.random() < 0.5:
                draw.ellipse((x - r, y - r, x + r, y + r), fill=color)
            else:
                draw.rectangle((x - r, y - r, x + r, y + r), fill=color)
        px = img.load()
        for yy in range(args.size):  # a soft gradient so neighbouring pixels are correlated
            shade = int(40 * math.sin(yy / args.size * math.pi))
            for xx in range(args.size):
                r, g, b = px[xx, yy]
                px[xx, yy] = (min(255, r + shade), g, max(0, b - shade))
        img.save(out / f"smoke_{i:04d}.png")
    print(f"wrote {args.count} images to {out}")


if __name__ == "__main__":
    main()
