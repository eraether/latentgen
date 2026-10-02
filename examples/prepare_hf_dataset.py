#!/usr/bin/env python
"""
Download a Hugging Face dataset into the plain folder layout the pipeline reads.

    pip install "datasets[audio]"          # only needed for this script

Images  (writes <out>/<index>.png, resized + centre-cropped to --size):
    python examples/prepare_hf_dataset.py images --dataset nuwandaa/ffhq128 --out data/ffhq128 --size 128
    python examples/prepare_hf_dataset.py images --dataset ILSVRC/imagenet-1k --split train --out data/imagenet256 \\
        --size 256 --max-items 100000                 # gated: accept the terms on the HF page + `huggingface-cli login`

Audio   (writes <out>/<index>.wav, mono 16-bit PCM at --sample-rate):
    python examples/prepare_hf_dataset.py audio --dataset google/speech_commands --config v0.02 \\
        --out data/speech_commands --sample-rate 16000 --max-items 50000

Then point a config at the folder (see configs/examples/*.yaml) and run the pipeline as usual.
Streaming mode is used, so --max-items stops the download early instead of fetching the whole dataset.
"""

import argparse
from pathlib import Path

import numpy as np


def _first_column_of_type(features, type_name: str) -> str:
    for name, feature in features.items():
        if type(feature).__name__ == type_name:
            return name
    raise SystemExit(f"no {type_name} column in this dataset; pass --column (columns: {list(features)})")


def prepare_images(args: argparse.Namespace) -> None:
    from datasets import load_dataset
    from PIL import Image

    ds = load_dataset(args.dataset, args.config, split=args.split, streaming=True)
    column = args.column or _first_column_of_type(ds.features, "Image")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for example in ds:
        img = example[column].convert("RGB")
        w, h = img.size
        scale = args.size / min(w, h)
        img = img.resize((max(args.size, round(w * scale)), max(args.size, round(h * scale))), Image.BICUBIC)
        w, h = img.size
        left, top = (w - args.size) // 2, (h - args.size) // 2
        img.crop((left, top, left + args.size, top + args.size)).save(out / f"{n:07d}.png")
        n += 1
        if n % 1000 == 0:
            print(f"{n} images written")
        if args.max_items and n >= args.max_items:
            break
    print(
        f"wrote {n} images to {out}  ->  data.kind: image, data.image_dir: {out}, data.image_size: {args.size}"
    )


def prepare_audio(args: argparse.Namespace) -> None:
    import wave

    from datasets import Audio, load_dataset

    ds = load_dataset(args.dataset, args.config, split=args.split, streaming=True)
    column = args.column or _first_column_of_type(ds.features, "Audio")
    ds = ds.cast_column(column, Audio(sampling_rate=args.sample_rate))  # resamples on the fly
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for example in ds:
        audio = np.asarray(example[column]["array"], dtype=np.float32)
        if audio.ndim == 2:  # [channels, T] -> mono
            audio = audio.mean(axis=0)
        pcm = np.clip(audio, -1, 1) * 32767
        with wave.open(str(out / f"{n:07d}.wav"), "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(args.sample_rate)
            f.writeframes(pcm.astype(np.int16).tobytes())
        n += 1
        if n % 1000 == 0:
            print(f"{n} clips written")
        if args.max_items and n >= args.max_items:
            break
    print(
        f"wrote {n} clips to {out}  ->  data.kind: audio, data.image_dir: {out}, data.sample_rate: {args.sample_rate}"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="kind", required=True)
    for kind in ("images", "audio"):
        q = sub.add_parser(kind)
        q.add_argument("--dataset", required=True, help="Hugging Face dataset id, e.g. nuwandaa/ffhq128")
        q.add_argument("--config", default=None, help="dataset configuration name, if it has several")
        q.add_argument("--split", default="train")
        q.add_argument(
            "--column", default=None, help="image / audio column (default: first one of that type)"
        )
        q.add_argument("--out", required=True, help="output folder")
        q.add_argument("--max-items", type=int, default=0, help="stop after this many items (0 = all)")
    sub.choices["images"].add_argument("--size", type=int, default=256, help="resize + centre-crop to this")
    sub.choices["audio"].add_argument("--sample-rate", type=int, default=16000)
    args = p.parse_args()
    (prepare_images if args.kind == "images" else prepare_audio)(args)


if __name__ == "__main__":
    main()
