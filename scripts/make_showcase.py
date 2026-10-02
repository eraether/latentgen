#!/usr/bin/env python
"""
Render the README showcase (docs/showcase/*.jpg) from your trained models.

    python scripts/make_showcase.py --config configs/ffhq512.yaml          # generate + compose
    python scripts/make_showcase.py --config configs/ffhq512.yaml --count 400 --batch-size 8
    python scripts/make_showcase.py --compose-only                          # re-lay-out existing samples
    python scripts/make_showcase.py --demo                                  # placeholder art, no torch / models

Two steps:

1. **generate** (needs the trained MaskGIT, VQ-VAE, GAN and autoencoder; ``generate.*_checkpoint``
   pick them, like scripts/04_generate.py) samples ``--count`` code grids and writes PNGs plus a
   ``meta.json`` to ``--work`` (default ``showcase_work/``): every sample through the GAN, a few
   through both decoders, one sample captured after every refine round, one code grid through
   eight noise draws, and two grids partially redrawn by MaskGIT.
2. **compose** (Pillow only) lays those out as the README graphics in ``--out`` (default
   ``docs/showcase/``): the hero banner, a gallery, the two decoders side by side, the refinement
   filmstrip, GAN variations and inpainting.

Generation is the slow part (~5 k MaskGIT passes per batch with the default sampler; 250 samples
at batch 8 is roughly an hour on a recent GPU). ``--compose-only`` re-runs just the layout, so
tweaking the graphics is instant. ``--demo`` fills the layouts with drawn placeholder faces to
preview them without any models.
"""

from __future__ import annotations

import argparse
import colorsys
import json
import random
import sys
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # make `latentgen` importable without `pip install -e .`

FONTS = ROOT / "docs" / "showcase" / "fonts"

# palette: the stage colors of the diagrams in docs/, on a near-black ground
BG = (13, 15, 20)
PANEL = (24, 27, 35)
TEXT = (236, 238, 243)
MUTED = (150, 156, 170)
VQ = (96, 150, 255)
MASKGIT = (167, 129, 255)
GAN = (255, 138, 76)
AE = (45, 212, 191)

REFINE_SAMPLES = 2
PAIR_SAMPLES = 3
VARIATIONS = 8
INPAINT_DRAWS = 5


# ============================================================================= generation (torch)


def generate(args: argparse.Namespace) -> None:
    import torch
    from tqdm import tqdm

    from latentgen.cli import checkpoint_for, setup
    from latentgen.config import load_config
    from latentgen.data import to_pil
    from latentgen.device import maybe_compile
    from latentgen.pretrained import load_autoencoder, load_generator, load_maskgit, load_vqvae
    from latentgen.sampling import decode_gan, decode_vq, fill_plan, forward_passes, generate_codes, resample

    cfg = load_config(args.config, args.set)
    g = cfg.generate
    if cfg.data.kind != "image":
        raise SystemExit("the showcase is laid out for images; for audio use scripts/04_generate.py")
    if min(args.batch_size, args.count) < PAIR_SAMPLES:
        raise SystemExit(
            f"--batch-size and --count must be at least {PAIR_SAMPLES} (the extras reuse the first batch)"
        )
    device = setup(cfg)
    maskgit, stats = load_maskgit(checkpoint_for(cfg, g.maskgit_checkpoint, "maskgit"), device)
    vqvae, _ = load_vqvae(checkpoint_for(cfg, g.vqvae_checkpoint, "vqvae"), device)
    generator, _ = load_generator(checkpoint_for(cfg, g.gan_checkpoint, "cgan"), device, use_ema=g.use_ema)
    autoencoder, _ = load_autoencoder(checkpoint_for(cfg, g.autoencoder_checkpoint, "autoencoder"), device)
    maskgit.forward = maybe_compile(maskgit.forward, cfg.project.compile)

    work = Path(args.work)
    for sub in ("gallery", "pairs", "refine", "variations", "inpaint"):
        (work / sub).mkdir(parents=True, exist_ok=True)

    def save(images: torch.Tensor, names: list[str]) -> None:
        for image, name in zip(images, names, strict=True):
            to_pil(image).save(work / name)

    passes = forward_passes(fill_plan(maskgit.cfg.num_positions, g.sample_fraction, g.keep_schedule))
    meta: dict = {
        "grid": [maskgit.cfg.grid_h, maskgit.cfg.grid_w],
        "codebook_size": maskgit.cfg.codebook_size,
        "image_size": cfg.data.image_size,
        "keep_schedule": list(g.keep_schedule),
        "forward_passes": passes,
        "codes": [],
    }
    bar = tqdm(total=args.count, desc="samples", unit="img")
    made = 0
    while made < args.count:
        b = min(args.batch_size, args.count - made)
        rounds: list[torch.Tensor] = []
        on_fill = (
            (lambda _i, _keep, c, rounds=rounds: rounds.append(c[:REFINE_SAMPLES])) if made == 0 else None
        )
        codes = generate_codes(maskgit, b, g.sample_fraction, g.keep_schedule, on_fill=on_fill)
        save(
            decode_gan(codes, generator, autoencoder, stats),
            [f"gallery/{made + i:04d}.png" for i in range(b)],
        )

        if made == 0:  # the extra showcases all reuse the first batch
            n = min(PAIR_SAMPLES, b)
            save(decode_vq(codes[:n], vqvae, stats), [f"pairs/{i}_vq.png" for i in range(n)])
            meta["codes"] = codes[:n].cpu().tolist()
            for r, grid in enumerate(rounds):  # the sample after every fill / refine round
                save(
                    decode_gan(grid, generator, autoencoder, stats),
                    [f"refine/{s}_{r}.png" for s in range(len(grid))],
                )
            meta["refine_rounds"] = len(rounds)
            one = codes[:1].expand(VARIATIONS, -1, -1)  # one code grid, eight noise draws
            save(
                decode_gan(one, generator, autoencoder, stats),
                [f"variations/{k}.png" for k in range(VARIATIONS)],
            )
            save(decode_vq(codes[:1], vqvae, stats), ["variations/vq.png"])
            H, W = codes.shape[1:]
            regions = ("top", "left")[: min(2, b)]
            for s, region in enumerate(regions):
                keep = torch.zeros(INPAINT_DRAWS, H, W, dtype=torch.bool, device=device)
                if region == "top":
                    keep[:, : H // 2] = True
                else:
                    keep[:, :, : W // 2] = True
                source = codes[s : s + 1].expand(INPAINT_DRAWS, -1, -1)
                redrawn = resample(maskgit, source, keep, g.sample_fraction)
                save(decode_gan(codes[s : s + 1], generator, autoencoder, stats), [f"inpaint/{s}_source.png"])
                save(
                    decode_gan(redrawn, generator, autoencoder, stats),
                    [f"inpaint/{s}_{k}.png" for k in range(INPAINT_DRAWS)],
                )
            meta["inpaint_regions"] = list(regions)
        made += b
        bar.update(b)
    bar.close()
    meta["count"] = args.count
    (work / "meta.json").write_text(json.dumps(meta))
    print(f"Wrote {args.count} samples and the extras to {work}")


# ============================================================================= demo samples (no torch)


def demo(args: argparse.Namespace) -> None:
    """Placeholder samples in the same layout as :func:`generate`: drawn faces, blurred for "VQ"."""
    rng = random.Random(0)
    work = Path(args.work)
    for sub in ("gallery", "pairs", "refine", "variations", "inpaint"):
        (work / sub).mkdir(parents=True, exist_ok=True)
    size = 256
    params = [_face_params(rng) for _ in range(args.count)]
    for i, p in enumerate(params):
        _draw_face(p, size).save(work / "gallery" / f"{i:04d}.png")
    keep_schedule = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64]
    meta = {
        "grid": [32, 32],
        "codebook_size": 256,
        "image_size": 512,
        "keep_schedule": keep_schedule,
        "forward_passes": 5207,
        "codes": [],
        "refine_rounds": len(keep_schedule) + 1,
        "inpaint_regions": ["top", "left"],
        "count": args.count,
        "demo": True,
    }
    for i in range(PAIR_SAMPLES):
        sharp = Image.open(work / "gallery" / f"{i:04d}.png")
        sharp.filter(ImageFilter.GaussianBlur(5)).save(work / "pairs" / f"{i}_vq.png")
        small = sharp.convert("RGB").resize((32, 32), Image.BILINEAR)
        meta["codes"].append(
            [[(r // 40) * 36 + (g // 40) * 6 + b // 40 for r, g, b in _row(small, y)] for y in range(32)]
        )
    for s in range(REFINE_SAMPLES):
        for r in range(len(keep_schedule) + 1):
            q = dict(params[s])
            if r < 3:  # early rounds: a different face entirely, then it settles
                q = _face_params(random.Random(1000 * s + r))
            _draw_face(q, size).filter(ImageFilter.GaussianBlur(max(0, 3 - r))).save(
                work / "refine" / f"{s}_{r}.png"
            )
    base = params[0]
    for k in range(VARIATIONS):
        q = dict(
            base,
            hair=_jitter(base["hair"], rng, 40),
            eyes=rng.uniform(0.9, 1.1),
            mouth=rng.uniform(-0.4, 0.6),
        )
        _draw_face(q, size).save(work / "variations" / f"{k}.png")
    _draw_face(base, size).filter(ImageFilter.GaussianBlur(5)).save(work / "variations" / "vq.png")
    for s, region in enumerate(("top", "left")):
        src = _draw_face(params[s], size)
        src.save(work / "inpaint" / f"{s}_source.png")
        for k in range(INPAINT_DRAWS):
            other = _draw_face(_face_params(random.Random(500 + 10 * s + k)), size)
            box = (0, size // 2, size, size) if region == "top" else (size // 2, 0, size, size)
            out = src.copy()
            out.paste(other.crop(box), box[:2])
            out.save(work / "inpaint" / f"{s}_{k}.png")
    (work / "meta.json").write_text(json.dumps(meta))


def _row(img: Image.Image, y: int):
    return [img.getpixel((x, y)) for x in range(img.width)]


def _jitter(color, rng, amount):
    return tuple(max(0, min(255, c + rng.randint(-amount, amount))) for c in color)


def _face_params(rng: random.Random) -> dict:
    hue = rng.random()
    bg = tuple(int(255 * c) for c in colorsys.hls_to_rgb(hue, rng.uniform(0.25, 0.6), rng.uniform(0.2, 0.5)))
    skin = tuple(
        int(255 * c) for c in colorsys.hls_to_rgb(rng.uniform(0.04, 0.09), rng.uniform(0.3, 0.8), 0.45)
    )
    hair = tuple(
        int(255 * c) for c in colorsys.hls_to_rgb(rng.uniform(0.0, 0.12), rng.uniform(0.08, 0.6), 0.5)
    )
    return {"bg": bg, "skin": skin, "hair": hair, "w": rng.uniform(0.27, 0.34), "eyes": rng.uniform(0.9, 1.1),
            "mouth": rng.uniform(-0.3, 0.7), "tilt": rng.uniform(-0.04, 0.04)}  # fmt: skip


def _draw_face(p: dict, size: int) -> Image.Image:
    s = 4 * size  # supersampled, then downscaled: smooth edges
    img = Image.new("RGB", (s, s), p["bg"])
    d = ImageDraw.Draw(img)
    cx, cy, w = s * (0.5 + p["tilt"]), s * 0.53, s * p["w"]
    d.ellipse(
        (cx - 1.25 * w, cy + 0.9 * w, cx + 1.25 * w, cy + 2.6 * w), fill=_jitter(p["bg"], random.Random(1), 0)
    )
    d.ellipse((cx - 1.12 * w, cy - 1.45 * w, cx + 1.12 * w, cy + 0.9 * w), fill=p["hair"])
    d.ellipse((cx - w, cy - 1.15 * w, cx + w, cy + 1.25 * w), fill=p["skin"])
    d.chord((cx - 1.05 * w, cy - 1.45 * w, cx + 1.05 * w, cy - 0.1 * w), 180, 360, fill=p["hair"])
    ey, ex, er = cy - 0.05 * w, 0.4 * w, 0.11 * w * p["eyes"]
    for sx in (-1, 1):
        d.ellipse((cx + sx * ex - er, ey - er, cx + sx * ex + er, ey + er), fill=(30, 26, 28))
    m = 0.32 * w
    d.arc((cx - m, cy + 0.35 * w - m * p["mouth"], cx + m, cy + 0.55 * w + m * 0.4), 20, 160, fill=(120, 50, 55),
          width=max(2, int(0.05 * w)))  # fmt: skip
    return img.resize((size, size), Image.LANCZOS)


# ============================================================================= composition (Pillow)


def font(size: int, weight: int = 400, display: bool = False) -> ImageFont.FreeTypeFont:
    if display:
        return ImageFont.truetype(str(FONTS / "ArchivoBlack-Regular.ttf"), size)
    f = ImageFont.truetype(str(FONTS / "Inter.ttf"), size)
    f.set_variation_by_axes([min(32, max(14, size)), weight])
    return f


class Samples:
    """Read access to the generate/demo output directory."""

    def __init__(self, work: Path) -> None:
        self.work = work
        if not (work / "meta.json").is_file():
            raise SystemExit(f"{work}/meta.json not found: run without --compose-only first (or with --demo)")
        self.meta = json.loads((work / "meta.json").read_text())
        self.gallery = sorted((work / "gallery").glob("*.png"))
        self._cache: dict = {}

    def open(self, rel: str | Path, size: int | None = None) -> Image.Image:
        key = (str(rel), size)
        if key not in self._cache:
            img = Image.open(self.work / rel if isinstance(rel, str) else rel).convert("RGB")
            self._cache[key] = img.resize((size, size), Image.LANCZOS) if size else img
        return self._cache[key]

    def tile(self, i: int, size: int) -> Image.Image:
        return self.open(self.gallery[i % len(self.gallery)], size)


def rounded(img: Image.Image, radius: int) -> Image.Image:
    mask = Image.new("L", img.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, img.width - 1, img.height - 1), radius, fill=255)
    out = Image.new("RGBA", img.size)
    out.paste(img, (0, 0), mask)
    return out


def paste_card(
    canvas: Image.Image, img: Image.Image, xy: tuple[int, int], radius: int = 12, shadow: int = 14
):
    """Rounded image with a soft drop shadow."""
    x, y = xy
    if shadow:
        sh = Image.new("L", (img.width + 4 * shadow, img.height + 4 * shadow), 0)
        ImageDraw.Draw(sh).rounded_rectangle(
            (2 * shadow, 2 * shadow, 2 * shadow + img.width, 2 * shadow + img.height), radius, fill=170
        )
        sh = sh.filter(ImageFilter.GaussianBlur(shadow / 2))
        canvas.paste((0, 0, 0), (x - 2 * shadow, y - 2 * shadow + shadow // 3), sh)
    card = rounded(img, radius)
    canvas.paste(card, (x, y), card)


def text(draw, xy, s, size, color=TEXT, weight=400, anchor="la", display=False):
    draw.text(xy, s, font=font(size, weight, display), fill=color, anchor=anchor)


def pill(draw: ImageDraw.ImageDraw, xy, label: str, color, size: int = 20) -> int:
    """A small rounded label; returns its width."""
    f = font(size, 600)
    w = int(draw.textlength(label, font=f)) + size
    x, y = xy
    draw.rounded_rectangle((x, y, x + w, y + int(size * 1.6)), size, fill=tuple(int(c * 0.22) for c in color))
    draw.text((x + w / 2, y + size * 0.8), label, font=f, fill=color, anchor="mm")
    return w


def arrow(draw: ImageDraw.ImageDraw, x0, y, x1, color=MUTED, width=3):
    draw.line((x0, y, x1 - 10, y), fill=color, width=width)
    draw.polygon([(x1, y), (x1 - 14, y - 8), (x1 - 14, y + 8)], fill=color)


def demo_stamp(img: Image.Image, samples: Samples) -> None:
    if samples.meta.get("demo"):
        d = ImageDraw.Draw(img)
        text(d, (img.width - 18, img.height - 14), "placeholder art -- run scripts/make_showcase.py", 15,
             (120, 124, 135), 500, anchor="rs")  # fmt: skip


def code_map(codes: list[list[int]], size: int) -> Image.Image:
    """A code grid as color: every code id gets a fixed hue (golden-angle spacing), nearest-neighbor upscaled."""
    h, w = len(codes), len(codes[0])
    img = Image.new("RGB", (w, h))
    for y, row in enumerate(codes):
        for x, c in enumerate(row):
            hue = (c * 0.618033988749895) % 1.0
            light = 0.35 + 0.3 * ((c * 7) % 11) / 10
            img.putpixel((x, y), tuple(int(255 * v) for v in colorsys.hls_to_rgb(hue, light, 0.65)))
    return img.resize((size, size), Image.NEAREST)


# ----------------------------------------------------------------------------- hero


def compose_hero(samples: Samples, title: str, out: Path) -> None:
    """The title set in faces: a dim mosaic of every sample, full-color inside the letters."""
    W, H, t = 2000, 640, 32
    rng = random.Random(7)
    cols, rows = W // t + 2, H // t + 1
    n = len(samples.gallery)
    order = list(range(n))
    rng.shuffle(order)
    mosaic = Image.new("RGB", (W, H), BG)
    for r in range(rows):
        offset = (t // 2) * (r % 2)  # brick bond: rows shifted by half a tile
        for c in range(cols):
            i = order[(r * 37 + c * 11) % n]  # neighbors (left, up, diagonal) never repeat a face
            mosaic.paste(samples.tile(i, t), (c * t - offset, r * t))

    # the title mask: as large as fits, centered slightly above the middle
    size = 330
    f = font(size, display=True)
    while f.getlength(title) > W * 0.9:
        size -= 10
        f = font(size, display=True)
    mask = Image.new("L", (W, H), 0)
    md = ImageDraw.Draw(mask)
    md.text((W / 2, H * 0.44), title, font=f, fill=255, anchor="mm")

    dim = ImageEnhance.Color(mosaic).enhance(0.25)
    dim = ImageEnhance.Brightness(dim).enhance(0.22).filter(ImageFilter.GaussianBlur(1.2))
    vignette = Image.radial_gradient("L").resize((W, H)).point(lambda v: 255 - int(v * 0.85))
    dim = Image.composite(dim, Image.new("RGB", (W, H), BG), vignette)
    bright = ImageEnhance.Contrast(mosaic).enhance(1.08)
    hero = Image.composite(bright, dim, mask)

    # a thin glowing rim around the letters
    edge = ImageChops.subtract(mask.filter(ImageFilter.MaxFilter(7)), mask)
    glow = edge.filter(ImageFilter.GaussianBlur(6))
    hero.paste(MASKGIT, (0, 0), glow.point(lambda v: int(v * 0.9)))
    hero.paste((255, 255, 255), (0, 0), edge.point(lambda v: int(v * 0.85)))

    d = ImageDraw.Draw(hero)
    text(d, (W / 2, H * 0.80), "coarse codes sampled by MaskGIT  ·  detail painted by a conditional GAN", 34,
         TEXT, 500, anchor="mm")  # fmt: skip
    m = samples.meta
    sub = (
        f"{len(samples.gallery)} unedited {m['image_size']}px samples, "
        f"{m['grid'][0]}x{m['grid'][1]} codes from a {m['codebook_size']}-word vocabulary"
    )
    text(d, (W / 2, H * 0.88), sub, 24, MUTED, 400, anchor="mm")
    demo_stamp(hero, samples)
    hero.save(out / "hero.jpg", quality=90)


# ----------------------------------------------------------------------------- gallery


def compose_gallery(samples: Samples, out: Path) -> None:
    """A bento wall: a few large samples among many small ones."""
    u, gap, cols, rows, pad = 140, 8, 13, 5, 28
    big = [(0, 0, 3), (4, 2, 3), (8, 0, 2), (11, 3, 2), (2, 3, 2)]  # (col, row, span)
    taken = [[False] * cols for _ in range(rows)]
    cells = []
    for c, r, s in big:
        cells.append((c, r, s))
        for dr in range(s):
            for dc in range(s):
                taken[r + dr][c + dc] = True
    cells += [(c, r, 1) for r in range(rows) for c in range(cols) if not taken[r][c]]
    W = pad * 2 + cols * u + (cols - 1) * gap
    H = pad * 2 + rows * u + (rows - 1) * gap + 56
    img = Image.new("RGB", (W, H), BG)
    for i, (c, r, s) in enumerate(cells):
        side = s * u + (s - 1) * gap
        paste_card(
            img, samples.tile(i, side), (pad + c * (u + gap), pad + r * (u + gap)), radius=10, shadow=0
        )
    d = ImageDraw.Draw(img)
    y = H - pad - 18
    x = pad
    for label, color in (("MaskGIT", MASKGIT), ("cGAN", GAN), ("AE decoder", AE)):
        x += pill(d, (x, y - 14), label, color, 18) + 10
    text(d, (x + 8, y + 1), f"{len(cells)} of {len(samples.gallery)} samples, no cherry-picking, no editing", 20,
         MUTED, 400, anchor="lm")  # fmt: skip
    demo_stamp(img, samples)
    img.save(out / "gallery.jpg", quality=90)


# ----------------------------------------------------------------------------- the two decoders


def compose_decoders(samples: Samples, out: Path) -> None:
    """codes -> VQ-VAE decoder (soft) vs cGAN + AE decoder (sharp), with a zoomed detail."""
    s, gap, pad, zoom = 300, 70, 40, 2.4
    n = min(PAIR_SAMPLES, len(samples.meta["codes"]))
    W = pad * 2 + 3 * s + 2 * gap + 40 + int(s * 0.8)
    H = pad + 70 + n * (s + 36) + pad
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    xs = [pad, pad + s + gap, pad + 2 * (s + gap)]
    zx = xs[2] + s + 40
    heads = [
        ("coarse codes", f"{samples.meta['grid'][0]}x{samples.meta['grid'][1]}, one of {samples.meta['codebook_size']} each", MASKGIT),
        ("VQ-VAE decoder", "faithful to the codes, but soft", VQ),
        ("cGAN + AE decoder", "same codes, detail hallucinated", GAN),
    ]  # fmt: skip
    for x, (title, sub, color) in zip(xs, heads, strict=True):
        text(d, (x, pad), title, 26, color, 700)
        text(d, (x, pad + 34), sub, 18, MUTED)
    text(d, (zx, pad), "zoomed", 26, TEXT, 700)
    text(d, (zx, pad + 34), f"{zoom:.1f}x, VQ above GAN", 18, MUTED)
    for i in range(n):
        y = pad + 80 + i * (s + 36)
        vq = samples.open(f"pairs/{i}_vq.png", s)
        gan = samples.tile(i, s)
        paste_card(img, code_map(samples.meta["codes"][i], s), (xs[0], y), radius=10, shadow=0)
        paste_card(img, vq, (xs[1], y), radius=10, shadow=0)
        paste_card(img, gan, (xs[2], y), radius=10, shadow=0)
        arrow(d, xs[0] + s + 12, y + s // 2, xs[1] - 12, VQ)
        d.line((xs[0] + s + 12, y + s // 2, xs[0] + s + 30, y + s // 2), fill=MUTED, width=3)
        # the code path forks: VQ decoder above, GAN below the line
        d.line((xs[0] + s + 30, y + s // 2, xs[0] + s + 30, y + s + 18), fill=GAN, width=3)
        d.line((xs[0] + s + 30, y + s + 18, xs[2] - 30, y + s + 18), fill=GAN, width=3)
        d.line((xs[2] - 30, y + s + 18, xs[2] - 30, y + s // 2 + 40), fill=GAN, width=3)
        d.polygon(
            [(xs[2] - 12, y + s // 2 + 40), (xs[2] - 26, y + s // 2 + 32), (xs[2] - 26, y + s // 2 + 48)],
            fill=GAN,
        )
        d.line((xs[2] - 30, y + s // 2 + 40, xs[2] - 26, y + s // 2 + 40), fill=GAN, width=3)
        # detail crop: around the eyes
        cw = int(s / zoom)
        box = ((s - cw) // 2, int(s * 0.36), (s - cw) // 2 + cw, int(s * 0.36) + cw // 2)
        zw = int(s * 0.8)
        for k, src in enumerate((vq, gan)):
            crop = src.crop(box).resize((zw, zw // 2), Image.LANCZOS)
            paste_card(img, crop, (zx, y + k * (zw // 2 + 8)), radius=8, shadow=0)
        d.rounded_rectangle(
            (xs[2] + box[0], y + box[1], xs[2] + box[2], y + box[3]), 4, outline=TEXT, width=2
        )
    demo_stamp(img, samples)
    img.save(out / "two_decoders.jpg", quality=90)


# ----------------------------------------------------------------------------- refinement


def compose_refinement(samples: Samples, out: Path) -> None:
    """One sample after every fill: early rounds redraw almost everything, late ones polish."""
    keeps = [0.0, *samples.meta["keep_schedule"]][: samples.meta["refine_rounds"]]
    n = len(keeps)
    s, gap, pad = 190, 14, 36
    rows = REFINE_SAMPLES
    W = pad * 2 + n * s + (n - 1) * gap
    H = pad + 130 + rows * (s + gap) + 90
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    text(d, (pad, pad), "Drafted, then redrafted", 30, TEXT, 700)
    text(d, (pad, pad + 40), "after the first fill, each round keeps a random few percent of the grid and redraws "
         "the rest around it", 19, MUTED)  # fmt: skip
    top = pad + 130
    for r, keep in enumerate(keeps):
        x = pad + r * (s + gap)
        label = "first fill" if r == 0 else f"keep {round(100 * keep)}%"
        text(d, (x + s / 2, top - 14), label, 18, MASKGIT if r else TEXT, 600, anchor="ms")
        for row in range(rows):
            paste_card(img, samples.open(f"refine/{row}_{r}.png", s), (x, top + row * (s + gap)), 8, 0)
    # progress bar: how much of the grid survives into each round
    y = top + rows * (s + gap) + 18
    d.rounded_rectangle((pad, y, W - pad, y + 10), 5, fill=PANEL)
    for r in range(n):
        cx = pad + r * (s + gap) + s / 2
        d.ellipse((cx - 7, y - 2, cx + 7, y + 12), fill=MASKGIT if r else TEXT)
    passes = samples.meta.get("forward_passes")
    note = f"{passes:,} MaskGIT forward passes per batch, one sampled slot per pass" if passes else ""
    text(d, (pad, y + 38), note, 17, MUTED)
    demo_stamp(img, samples)
    img.save(out / "refinement.jpg", quality=90)


# ----------------------------------------------------------------------------- GAN variations


def compose_variations(samples: Samples, out: Path) -> None:
    """One code grid through the generator with eight different noise draws."""
    big, s, gap, pad = 420, 200, 12, 40
    W = pad * 2 + big + 90 + 4 * s + 3 * gap
    H = pad * 2 + 110 + max(big, 2 * s + gap)
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    text(d, (pad, pad), "One sketch, many finishes", 30, TEXT, 700)
    text(d, (pad, pad + 40), "the cGAN's noise decides hair strands, skin texture and lighting; the codes keep "
         "identity and pose", 19, MUTED)  # fmt: skip
    top = pad + 110
    paste_card(img, samples.open("variations/vq.png", big), (pad, top), 14, 0)
    pill(d, (pad + 14, top + big - 46), "VQ-VAE view of the codes", VQ, 18)
    gx = pad + big + 90
    arrow(d, pad + big + 18, top + big // 2, gx - 18, GAN, 4)
    for k in range(VARIATIONS):
        x = gx + (k % 4) * (s + gap)
        y = top + (k // 4) * (s + gap)
        paste_card(img, samples.open(f"variations/{k}.png", s), (x, y), 10, 0)
    demo_stamp(img, samples)
    img.save(out / "variations.jpg", quality=90)


# ----------------------------------------------------------------------------- inpainting


def _hatch(size: int, region: str) -> Image.Image:
    """Mask (L) of the redrawn half, with diagonal stripes."""
    m = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(m)
    box = (0, size // 2, size, size) if region == "top" else (size // 2, 0, size, size)
    for k in range(-size, 2 * size, 14):
        d.line((k, 0, k + size, size), fill=255, width=4)
    keep = Image.new("L", (size, size), 0)
    ImageDraw.Draw(keep).rectangle(box, fill=255)
    return ImageChops.multiply(m, keep), keep


def compose_inpainting(samples: Samples, out: Path) -> None:
    """Keep half of a code grid, let MaskGIT redraw the other half: no extra training needed."""
    s, gap, pad = 220, 12, 40
    regions = samples.meta.get("inpaint_regions", ["top", "left"])
    W = pad * 2 + s + 60 + INPAINT_DRAWS * s + (INPAINT_DRAWS - 1) * gap
    H = pad * 2 + 100 + len(regions) * (s + 44)
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    text(d, (pad, pad), "Keep half, reimagine the rest", 30, TEXT, 700)
    text(d, (pad, pad + 40), "MaskGIT redraws the hatched codes conditioned on the kept ones; same model, "
         "no inpainting training", 19, MUTED)  # fmt: skip
    for row, region in enumerate(regions):
        y = pad + 110 + row * (s + 44)
        src = samples.open(f"inpaint/{row}_source.png", s).copy()
        stripes, redraw = _hatch(s, region)
        src = Image.composite(ImageEnhance.Brightness(src).enhance(0.35), src, redraw)
        src.paste(MASKGIT, (0, 0), stripes.point(lambda v: int(v * 0.6)))
        paste_card(img, src, (pad, y), 10, 0)
        caption = "keep the top half" if region == "top" else "keep the left half"
        text(d, (pad, y + s + 10), caption, 17, MUTED)
        x0 = pad + s + 60
        arrow(d, pad + s + 12, y + s // 2, x0 - 12, MASKGIT)
        for k in range(INPAINT_DRAWS):
            paste_card(img, samples.open(f"inpaint/{row}_{k}.png", s), (x0 + k * (s + gap), y), 10, 0)
    demo_stamp(img, samples)
    img.save(out / "inpainting.jpg", quality=90)


def compose(args: argparse.Namespace) -> None:
    samples = Samples(Path(args.work))
    if len(samples.gallery) < 60:
        print(
            f"note: only {len(samples.gallery)} samples; the hero and gallery repeat faces (250+ looks best)"
        )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for fn in (compose_hero, compose_gallery, compose_decoders, compose_refinement, compose_variations,
               compose_inpainting):  # fmt: skip
        fn(samples, args.title, out) if fn is compose_hero else fn(samples, out)
        print(f"  {fn.__name__.removeprefix('compose_')} done")
    print(f"Showcase written to {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(ROOT / "configs" / "ffhq512.yaml"))
    p.add_argument("--set", nargs="*", action="extend", default=[], metavar="KEY=VALUE")
    p.add_argument(
        "--count", type=int, default=250, help="samples to generate (the hero and gallery use them all)"
    )
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--work", default="showcase_work", help="where generated samples are kept")
    p.add_argument("--out", default=str(ROOT / "docs" / "showcase"), help="where the graphics go")
    p.add_argument("--title", default="latentgen", help="text of the hero banner")
    p.add_argument("--compose-only", action="store_true", help="skip generation, lay out existing samples")
    p.add_argument("--demo", action="store_true", help="placeholder samples instead of models (no torch)")
    args = p.parse_args()
    if args.demo:
        demo(args)
    elif not args.compose_only:
        generate(args)
    compose(args)


if __name__ == "__main__":
    main()
