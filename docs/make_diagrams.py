#!/usr/bin/env python
"""
Generates every diagram in docs/ (SVG, plus PNG when `cairosvg` is installed).

    python docs/make_diagrams.py

The diagrams are drawn with the tiny helper below rather than by hand so they stay consistent and
are easy to edit: change a label or a box here, re-run, done. No torch needed.
"""

from __future__ import annotations

import html
from pathlib import Path

OUT = Path(__file__).resolve().parent

# ----------------------------------------------------------------------------- palette
INK = "#111827"
MUTED = "#4b5563"
LINE = "#6b7280"
BG = "#ffffff"
STAGE = {  # fill, stroke, per stage
    "vqvae": ("#dbeafe", "#2563eb"),
    "ae": ("#ccfbf1", "#0d9488"),
    "encode": ("#e5e7eb", "#4b5563"),
    "maskgit": ("#ede9fe", "#7c3aed"),
    "gan": ("#ffedd5", "#ea580c"),
    "infer": ("#dcfce7", "#16a34a"),
    "data": ("#fef9c3", "#ca8a04"),
    "plain": ("#f9fafb", "#9ca3af"),
}
FONT = "Inter, 'Helvetica Neue', Helvetica, Arial, sans-serif"
MONO = "'DejaVu Sans Mono', Menlo, Consolas, monospace"


class Canvas:
    def __init__(self, width: int, height: int, title: str) -> None:
        self.w, self.h, self.title = width, height, title
        self.parts: list[str] = []

    # -- primitives -------------------------------------------------------------------
    def text(self, x, y, s, size=13, anchor="middle", color=INK, weight="normal", mono=False, italic=False):
        style = f"font-family:{MONO if mono else FONT};font-size:{size}px;font-weight:{weight};fill:{color}"
        if italic:
            style += ";font-style:italic"
        self.parts.append(
            f'<text x="{x}" y="{y}" text-anchor="{anchor}" xml:space="preserve" style="{style}">{html.escape(s)}</text>'
        )

    def lines(self, x, y, items, size=12, anchor="middle", color=MUTED, gap=16, mono=False):
        for i, s in enumerate(items):
            self.text(x, y + i * gap, s, size=size, anchor=anchor, color=color, mono=mono)

    def box(
        self,
        x,
        y,
        w,
        h,
        title=None,
        body=(),
        kind="plain",
        r=10,
        title_size=14,
        body_size=12,
        mono_body=False,
        dashed=False,
        align="middle",
    ):
        fill, stroke = STAGE[kind]
        dash = ' stroke-dasharray="6 4"' if dashed else ""
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="1.6"{dash}/>'
        )
        cy = y + 20
        if title:
            self.text(x + w / 2, cy, title, size=title_size, weight="600", color=INK)
            cy += 18
        if align == "start":
            self.lines(x + 16, cy, body, size=body_size, mono=mono_body, anchor="start")
        else:
            self.lines(x + w / 2, cy, body, size=body_size, mono=mono_body)
        return (x, y, w, h)

    def pill(self, x, y, s, kind="plain", size=11):
        fill, stroke = STAGE[kind]
        w = len(s) * size * 0.62 + 18
        self.parts.append(
            f'<rect x="{x - w / 2}" y="{y - 11}" width="{w}" height="22" rx="11" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="1.2"/>'
        )
        self.text(x, y + 4, s, size=size, color=INK, weight="600")

    def arrow(self, x1, y1, x2, y2, label=None, color=LINE, dashed=False, label_dy=-6, width=1.8, mono=False):
        dash = ' stroke-dasharray="5 4"' if dashed else ""
        self.parts.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}" '
            f'marker-end="url(#arrow)"{dash}/>'
        )
        if label:
            self.text((x1 + x2) / 2, (y1 + y2) / 2 + label_dy, label, size=11, color=MUTED, mono=mono)

    def path(self, d, color=LINE, dashed=False, arrow=True, width=1.8):
        dash = ' stroke-dasharray="5 4"' if dashed else ""
        mk = ' marker-end="url(#arrow)"' if arrow else ""
        self.parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}"{mk}{dash}/>')

    def grid(self, x, y, n, cell, values=None, masked=None, kind="maskgit", label=None):
        """An n x n token grid. `values` colours cells by code id, `masked` set of (r, c) drawn hatched."""
        fill, stroke = STAGE[kind]
        palette = ["#bfdbfe", "#fde68a", "#bbf7d0", "#fecaca", "#ddd6fe", "#fbcfe8", "#a5f3fc", "#fed7aa"]
        for r in range(n):
            for c in range(n):
                cx, cy = x + c * cell, y + r * cell
                if masked and (r, c) in masked:
                    self.parts.append(
                        f'<rect x="{cx}" y="{cy}" width="{cell}" height="{cell}" fill="url(#hatch)" '
                        f'stroke="{stroke}" stroke-width="0.8"/>'
                    )
                else:
                    col = palette[(values[r][c] if values else (r * 3 + c * 5)) % len(palette)]
                    self.parts.append(
                        f'<rect x="{cx}" y="{cy}" width="{cell}" height="{cell}" fill="{col}" '
                        f'stroke="{stroke}" stroke-width="0.8"/>'
                    )
        if label:
            self.text(x + n * cell / 2, y + n * cell + 16, label, size=11, color=MUTED)

    def image_icon(self, x, y, w, h, label=None):
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="#fde68a" stroke="#ca8a04" stroke-width="1.4"/>'
        )
        self.parts.append(
            f'<circle cx="{x + w * 0.5}" cy="{y + h * 0.42}" r="{h * 0.18}" fill="#fbbf24" stroke="#ca8a04"/>'
        )
        self.parts.append(
            f'<path d="M{x + w * 0.2},{y + h * 0.9} Q{x + w * 0.5},{y + h * 0.55} {x + w * 0.8},{y + h * 0.9}" '
            f'fill="#f59e0b" stroke="#ca8a04"/>'
        )
        if label:
            self.text(x + w / 2, y + h + 16, label, size=11, color=MUTED)

    # -- output -----------------------------------------------------------------------
    def save(self, name: str) -> None:
        defs = (
            "<defs>"
            f'<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">'
            f'<path d="M0,0 L10,5 L0,10 z" fill="{LINE}"/></marker>'
            '<pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
            '<rect width="6" height="6" fill="#f3f4f6"/><line x1="0" y1="0" x2="0" y2="6" stroke="#9ca3af" stroke-width="2"/>'
            "</pattern></defs>"
        )
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" viewBox="0 0 {self.w} {self.h}">'
            f'<title>{html.escape(self.title)}</title>{defs}<rect width="{self.w}" height="{self.h}" fill="{BG}"/>'
            + "".join(self.parts)
            + "</svg>"
        )
        (OUT / f"{name}.svg").write_text(svg)
        try:
            import cairosvg

            cairosvg.svg2png(
                bytestring=svg.encode(), write_to=str(OUT / f"{name}.png"), output_width=self.w * 2
            )
        except ImportError:
            pass
        print("wrote", name)


# ============================================================================================
# 1. pipeline overview
# ============================================================================================
def pipeline_overview():
    c = Canvas(1180, 620, "latentgen pipeline overview")
    c.text(
        590, 30, "The whole pipeline: four training stages, one generation step (4)", size=20, weight="700"
    )
    c.text(
        590, 52, "arrows carry data; each box is one script in scripts/", size=12, color=MUTED, italic=True
    )

    # ---- images source
    c.image_icon(40, 150, 90, 70, "your image folder")
    c.text(85, 250, "512 x 512 x 3", size=11, color=MUTED, mono=True)

    # ---- stage 1a / 1b
    c.box(
        200,
        100,
        230,
        110,
        "Stage 1a  VQ-VAE",
        ["image -> 32x32 grid of codes", "codebook of 256 vectors", "scripts/01a_train_vqvae.py"],
        kind="vqvae",
    )
    c.box(
        200,
        240,
        230,
        110,
        "Stage 1b  Autoencoder",
        [
            "image -> 32x32x32 latent in [-1,1]",
            "keeps ~32x more detail than VQ",
            "scripts/01b_train_autoencoder.py",
        ],
        kind="ae",
    )
    c.arrow(135, 170, 198, 155)
    c.arrow(135, 200, 198, 295)
    c.text(85, 268, "(1a and 1b are trained", size=10, color=MUTED, italic=True)
    c.text(85, 281, "independently)", size=10, color=MUTED, italic=True)

    # ---- stage 1c
    c.box(
        500,
        160,
        220,
        110,
        "Stage 1c  Encode",
        ["run every image (and its flip)", "through both frozen models", "scripts/01c_encode_dataset.py"],
        kind="encode",
    )
    c.arrow(432, 155, 498, 195)
    c.arrow(432, 295, 498, 235)

    # ---- encoded file
    c.box(
        790,
        150,
        190,
        130,
        "data/encoded.pt",
        [
            "codes   int16 [2, N, 32, 32]",
            "latents int8  [2, N, 32, 32, 32]",
            "",
            "stages 2 and 3 only read this",
        ],
        kind="data",
        mono_body=False,
    )
    c.arrow(722, 215, 788, 215)

    # ---- stage 2 / 3
    c.box(
        500,
        360,
        220,
        110,
        "Stage 2  MaskGIT",
        ["learns p(code grid)", "masked-token transformer, 24 layers", "scripts/02_train_maskgit.py"],
        kind="maskgit",
    )
    c.box(
        790,
        360,
        220,
        110,
        "Stage 3  cGAN",
        ["codes -> AE latent", "generator 12 / discriminator 16 layers", "scripts/03_train_cgan.py"],
        kind="gan",
    )
    c.path("M860,282 L860,320 L610,320 L610,358")
    c.text(735, 312, "codes only", size=11, color=MUTED)
    c.arrow(885, 282, 885, 358)
    c.text(895, 340, "codes + latents", size=11, color=MUTED, anchor="start")

    # ---- inference
    c.box(
        260,
        500,
        720,
        95,
        "Generate   scripts/04_generate.py",
        [
            "MaskGIT samples a 32x32 code grid from nothing  ->  VQ-VAE decoder: faithful but soft image",
            "                                                 ->  GAN generator -> AE decoder: sharp image",
        ],
        kind="infer",
    )
    c.arrow(610, 472, 610, 498)
    c.arrow(900, 472, 900, 498)
    c.path("M260,352 L260,470 L330,470 L330,498", dashed=True)
    c.path("M198,195 L180,195 L180,480 L300,480 L300,498", dashed=True)
    c.text(185, 492, "frozen decoders", size=11, color=MUTED, italic=True, anchor="start")
    c.save("pipeline_overview")


# ============================================================================================
# 2. stage 1 codecs
# ============================================================================================
def stage1_codecs():
    c = Canvas(1180, 560, "stage 1 image codecs")
    c.text(
        590,
        30,
        "Stage 1: two image codecs with the same encoder / decoder, different bottlenecks",
        size=19,
        weight="700",
    )
    c.text(
        590,
        52,
        "“everything is an MLP”: tokens are refined independently; the only spatial mixing is one 3x3 conv in the decoder",
        size=12,
        color=MUTED,
        italic=True,
    )

    y = 90
    # encoder chain
    c.image_icon(40, y + 40, 80, 64, "input image")
    c.text(80, y + 140, "[B, 3, 512, 512]", size=10, color=MUTED, mono=True)
    c.box(
        160, y + 30, 150, 90, "PixelUnshuffle(16)", ["fold each 16x16 patch", "into channels"], kind="plain"
    )
    c.text(235, y + 140, "[B, 768, 32, 32]", size=10, color=MUTED, mono=True)
    c.box(350, y + 30, 130, 90, "1x1 conv", ["patch -> hidden", "(768 -> 256)"], kind="plain")
    c.text(415, y + 140, "[B, 256, 32, 32]", size=10, color=MUTED, mono=True)
    c.box(
        520, y + 30, 170, 90, "6 x MLPBlock2D", ["RMSNorm2D -> SwiGLU", "residual, per token"], kind="plain"
    )
    c.text(605, y + 140, "[B, 256, 32, 32]", size=10, color=MUTED, mono=True)
    c.arrow(125, y + 72, 158, y + 72)
    c.arrow(312, y + 72, 348, y + 72)
    c.arrow(482, y + 72, 518, y + 72)
    c.text(400, y + 12, "PatchEncoder  (latentgen/nn/codec.py)", size=12, weight="600", color=MUTED)

    # bottlenecks
    bx = 740
    c.box(
        bx,
        y - 10,
        400,
        118,
        "VQ-VAE bottleneck  (stage 1a)",
        [
            "RMSNorm2D -> 1x1 conv to 64 channels",
            "snap each of the 1024 positions to its nearest codebook vector",
            "codebook: hypernetwork(noise) until cutover, then a learned table",
            "out: codes [B, 32, 32] int + straight-through latent [B, 64, 32, 32]",
        ],
        kind="vqvae",
        body_size=11,
    )
    c.box(
        bx,
        y + 124,
        400,
        70,
        "AE bottleneck  (stage 1b)",
        [
            "RMSNorm2D -> 1x1 conv to 32 channels -> tanh",
            "out: latent [B, 32, 32, 32] in [-1, 1]",
        ],
        kind="ae",
        body_size=11,
    )
    c.arrow(692, y + 60, bx - 2, y + 50)
    c.arrow(692, y + 85, bx - 2, y + 160)

    # decoder chain
    y2 = 340
    c.text(
        590,
        y2 - 20,
        "PatchDecoder  (same class, separate weights per model: the VQ-VAE's decodes the quantized latent, the AE's the tanh latent)",
        size=12,
        weight="600",
        color=MUTED,
    )
    c.box(140, y2, 170, 90, "3x3 conv", ["bottleneck -> hidden", "the ONLY spatial mixing"], kind="plain")
    c.box(350, y2, 170, 90, "6 x MLPBlock2D", ["same block as the encoder"], kind="plain")
    c.box(560, y2, 150, 90, "1x1 conv", ["hidden -> 768"], kind="plain")
    c.box(750, y2, 160, 90, "PixelShuffle(16)", ["channels -> 16x16 pixels"], kind="plain")
    c.image_icon(960, y2 + 12, 80, 64, "reconstruction")
    c.arrow(312, y2 + 45, 348, y2 + 45)
    c.arrow(522, y2 + 45, 558, y2 + 45)
    c.arrow(712, y2 + 45, 748, y2 + 45)
    c.arrow(912, y2 + 45, 958, y2 + 45)
    c.path(f"M{bx + 190},{y + 196} L{bx + 190},{y2 - 40} L80,{y2 - 40} L80,{y2 + 45} L138,{y2 + 45}")
    c.text(225, y2 + 110, "[B, 64 or 32, 32, 32]", size=10, color=MUTED, mono=True)
    c.text(830, y2 + 110, "[B, 3, 512, 512]", size=10, color=MUTED, mono=True)

    c.box(
        140,
        470,
        900,
        70,
        None,
        [
            "Loss: smooth-L1(10 * reconstruction, 10 * input)   + commitment loss for the VQ-VAE (pulls encoder outputs onto their codes)",
            "Metric: PSNR on [0,1] images.  FFHQ: VQ-VAE ~25 dB after 5 epochs (768x compression); the AE is much sharper (24x compression).",
        ],
        kind="plain",
        body_size=11,
        dashed=True,
    )
    c.save("stage1_codecs")


# ============================================================================================
# 3. hypernetwork codebook + cutover
# ============================================================================================
def codebook_cutover():
    c = Canvas(1000, 380, "hypernetwork codebook and cutover")
    c.text(
        500, 30, "The VQ-VAE codebook: generated from noise, then frozen (“cutover”)", size=19, weight="700"
    )

    c.box(
        40,
        80,
        280,
        150,
        "Before cutover (step < cutover_step)",
        [
            "every forward pass:",
            "noise [256, hidden] ~ N(0, 1)",
            "-> 6 transformer layers -> RMSNorm -> Linear",
            "-> codebook [256, 64]   (new every pass!)",
            "",
            "acts as a strong regulariser: no dead codes",
        ],
        kind="vqvae",
        body_size=11,
    )
    c.box(
        380,
        80,
        240,
        150,
        "Cutover (once)",
        [
            "sample the codebook one last time",
            "copy it into `learned_codebook`",
            "set the `cutover` flag (saved in ckpt)",
            "reset the optimizer",
            "",
            "config: vqvae.cutover_step",
        ],
        kind="encode",
        body_size=11,
    )
    c.box(
        680,
        80,
        280,
        150,
        "After cutover",
        [
            "codebook = learned_codebook [256, 64]",
            "a plain nn.Parameter, trained directly",
            "hypernetwork weights kept but unused",
            "",
            "the model is now a normal VQ-VAE;",
            "only now are codes stable enough to encode",
        ],
        kind="vqvae",
        body_size=11,
    )
    c.arrow(322, 155, 378, 155)
    c.arrow(622, 155, 678, 155)

    c.box(
        40,
        260,
        920,
        95,
        None,
        [
            "FFHQ timeline from the original notes:  PSNR 24.4 dB at cutover (2 epochs, 2180 steps, 30 min)  ->  25.2 dB after 5 epochs (6020 steps, 1 h)",
            "Resuming a pre-cutover checkpoint past the step?  The trainer cuts over right after the next optimizer step.",
            "Force it:  python scripts/01a_train_vqvae.py --resume runs/vqvae/latest --cutover-now",
        ],
        kind="plain",
        body_size=11,
        dashed=True,
    )
    c.save("codebook_cutover")


# ============================================================================================
# 4. stage 2 MaskGIT
# ============================================================================================
def stage2_maskgit():
    c = Canvas(1180, 520, "stage 2 MaskGIT")
    c.text(590, 30, "Stage 2: MaskGIT learns the distribution of code grids", size=19, weight="700")
    c.text(
        590,
        52,
        "a bidirectional transformer: hide part of the grid, predict what was there",
        size=12,
        color=MUTED,
        italic=True,
    )

    masked = {
        (0, 1),
        (0, 3),
        (1, 0),
        (1, 2),
        (1, 4),
        (2, 1),
        (2, 3),
        (3, 0),
        (3, 2),
        (3, 4),
        (4, 1),
        (4, 3),
        (0, 0),
        (2, 0),
        (4, 4),
    }
    c.grid(60, 110, 5, 26, label="code grid from encoded.pt")
    c.text(125, 272, "[B, 32, 32] ints in 0..255", size=10, color=MUTED, mono=True)
    c.arrow(200, 175, 250, 175)
    c.text(225, 100, "mask r ~ U[0.3, 1.0]", size=11, color=MUTED)
    c.grid(260, 110, 5, 26, masked=masked, label="masked positions = [MASK] embedding")
    c.arrow(400, 175, 450, 175)

    c.box(
        455,
        95,
        300,
        160,
        "MaskGIT  (latentgen/nn/maskgit.py)",
        [
            "code_embedding [256, 512] (+ learned mask token)",
            "2D RoPE: head dims rotate by row and by column",
            "24 x TransformerLayer (attention + SwiGLU)",
            "final RMSNorm -> Linear to 256 logits per position",
            "",
            "out: logits [B, 32, 32, 256]",
        ],
        kind="maskgit",
        body_size=11,
    )
    c.arrow(757, 175, 805, 175)
    c.box(
        810,
        95,
        320,
        160,
        "Loss",
        [
            "cross-entropy ONLY at masked positions",
            "(the visible ones are given, not predicted)",
            "",
            "reported per mask-ratio bucket:",
            "100-91% masked  (global structure, hard)",
            "...  40-30% masked  (local detail, easy)",
        ],
        kind="plain",
        body_size=11,
    )

    c.box(
        60,
        300,
        1070,
        95,
        "Why mask ratios 30 %..100 %?",
        [
            "Generation starts from a fully masked grid and every refine round re-masks 36 %..99 % of it, so the hard,",
            "high-density states dominate. Inside a fill the mask thins out further; those easier states are left to generalise.",
        ],
        kind="plain",
        body_size=11,
        dashed=True,
    )
    c.box(
        60,
        415,
        1070,
        85,
        "Training data path",
        [
            "the whole encoded dataset (int16 codes, 2 x 70k x 32 x 32 = 290 MB) sits in CPU RAM or VRAM (data.encoded_device);",
            "a batch is a random gather of rows + a random flip variant; no image decoding, no DataLoader, no disk reads.",
        ],
        kind="data",
        body_size=11,
    )
    c.save("stage2_maskgit")


# ============================================================================================
# 5. stage 3 cGAN
# ============================================================================================
def stage3_cgan():
    c = Canvas(1180, 600, "stage 3 conditional GAN")
    c.text(
        590,
        30,
        "Stage 3: the conditional GAN adds back the detail the VQ codes threw away",
        size=19,
        weight="700",
    )
    c.text(
        590,
        52,
        "codes (768x compressed) -> AE latent (24x compressed); the frozen AE decoder turns that into pixels",
        size=12,
        color=MUTED,
        italic=True,
    )

    # generator
    c.grid(50, 110, 4, 24, label="codes [B,32,32]")
    c.arrow(150, 158, 200, 158)
    c.box(
        205,
        95,
        300,
        125,
        "Generator",
        [
            "code embedding + Linear(noise ~ N(0,1)) per position",
            "12 x TransformerLayer with 2D RoPE",
            "RMSNorm -> Linear(32) -> tanh",
            "out: fake latent [B, 32, 32, 32] in [-1, 1]",
        ],
        kind="gan",
        body_size=11,
    )
    c.arrow(507, 158, 560, 158, "fake")
    # real latent
    c.box(
        205,
        250,
        300,
        70,
        "Real latent from encoded.pt",
        ["int8 / 127 + dequantisation noise, clamp [-1, 1]"],
        kind="data",
        body_size=11,
    )
    c.arrow(507, 285, 560, 200, "real", label_dy=12)
    # discriminator
    c.box(
        565,
        95,
        300,
        160,
        "Discriminator",
        [
            "code embedding + Linear(latent) per position",
            "16 x TransformerLayer with 2D RoPE",
            "RMSNorm -> Linear(1) -> mean over positions",
            "out: one real/fake logit per image",
            "",
            "sees the codes too: “does this latent fit these codes?”",
        ],
        kind="gan",
        body_size=11,
    )
    c.arrow(867, 175, 920, 175)
    c.box(
        925,
        95,
        215,
        160,
        "Losses (BCE)",
        [
            "D: real -> 1, fake -> 0",
            "G: fake -> 1 (D frozen,",
            "    only input grads flow)",
            "",
            "logit beyond +-0.5 counts",
            "as a confident call",
        ],
        kind="plain",
        body_size=11,
    )

    # phase schedule state machine
    y = 360
    c.text(590, y - 15, "Who trains this step?  (CGANStage.step)", size=14, weight="600")
    c.box(
        150,
        y,
        330,
        90,
        "Discriminator phase",
        ["D steps on real + fake", "G only produces fakes (no grads)", "stays here while D accuracy <= 85 %"],
        kind="gan",
        body_size=11,
    )
    c.box(
        700,
        y,
        330,
        90,
        "Generator phase",
        [
            "G steps against the frozen D",
            "D keeps scoring the new fakes",
            "stays here while D accuracy > 85 %",
        ],
        kind="gan",
        body_size=11,
    )
    c.path(f"M482,{y + 30} L698,{y + 30}")
    c.text(590, y + 22, "D accuracy on real AND fake > 85 %", size=11, color=MUTED)
    c.path(f"M698,{y + 65} L482,{y + 65}")
    c.text(590, y + 82, "D fooled: fake accuracy <= 85 %", size=11, color=MUTED)

    c.box(
        150,
        480,
        880,
        100,
        None,
        [
            "Generator EMA: every 10 successful generator steps, ema = 0.99 * ema + 0.01 * weights  (half-life ~690 generator steps).",
            "Inference samples from the EMA copy (generate.use_ema). Progress images show  real | generator | EMA  decoded by the frozen AE.",
            "Optimizer: plain AdamW (betas 0.5 / 0.999, no weight decay) with a 100-step warm-up -- not schedule-free, as GANs need the live weights.",
        ],
        kind="plain",
        body_size=11,
        dashed=True,
    )
    c.save("stage3_cgan")


# ============================================================================================
# 6. sampling
# ============================================================================================
def sampling():
    c = Canvas(1180, 560, "generation / sampling")
    c.text(
        590,
        30,
        "Generation: fill a masked grid, then refine it in rounds, then decode two ways",
        size=19,
        weight="700",
    )
    c.text(
        590,
        52,
        "latentgen/sampling.py -- one slot per forward pass, random order, plain softmax sampling",
        size=12,
        color=MUTED,
        italic=True,
    )

    all_masked = {(r, cc) for r in range(4) for cc in range(4)}
    some = {(0, 1), (1, 3), (2, 0), (3, 2), (0, 3), (2, 2), (1, 0), (3, 3), (0, 0), (2, 3), (1, 2), (3, 0)}
    few = {(1, 1), (2, 3), (0, 2)}
    gx = [60, 212, 364, 516, 834, 986]  # grid x positions, 88 px wide each
    c.grid(gx[0], 100, 4, 22, masked=all_masked, label="1. all masked")
    c.arrow(gx[0] + 93, 144, gx[1] - 5, 144)
    c.grid(gx[1], 100, 4, 22, label="fill: sample 75 % of slots")
    c.text(gx[1] + 44, 215, "one at a time, argmax the rest", size=10, color=MUTED)
    c.arrow(gx[1] + 93, 144, gx[2] - 5, 144, "keep 1 %", label_dy=-10)
    c.grid(gx[2], 100, 4, 22, masked=some, label="2. re-mask 99 %, fill")
    c.arrow(gx[2] + 93, 144, gx[3] - 5, 144, "keep 2 %", label_dy=-10)
    c.grid(gx[3], 100, 4, 22, masked=some, label="3. re-mask 98 %, fill")
    c.text((gx[3] + 88 + gx[4] - 60) / 2, 148, "... 4, 8, 16, 32 % ...", size=12, color=MUTED)
    c.arrow(gx[4] - 60, 144, gx[4] - 5, 144, "keep 64 %", label_dy=-10)
    c.grid(gx[4], 100, 4, 22, masked=few, label="8. re-mask 36 %, fill")
    c.arrow(gx[4] + 93, 144, gx[5] - 5, 144)
    c.grid(gx[5], 100, 4, 22, label="final code grid")

    # bar chart of forward passes per fill (keep schedule)
    c.text(300, 270, "Forward passes per fill (1024 positions, sample_fraction 0.75)", size=13, weight="600")
    keeps = [0.0, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64]
    bx, by, bw, bh = 80, 290, 440, 150
    c.parts.append(f'<line x1="{bx}" y1="{by + bh}" x2="{bx + bw}" y2="{by + bh}" stroke="{LINE}"/>')
    maxv = 1024 * 0.75 + 1
    for i, k in enumerate(keeps):
        n_masked = 1024 - round(k * 1024)
        passes = round(0.75 * n_masked) + 1
        h = bh * passes / maxv
        xx = bx + 12 + i * 54
        c.parts.append(
            f'<rect x="{xx}" y="{by + bh - h}" width="40" height="{h}" fill="{STAGE["infer"][0]}" stroke="{STAGE["infer"][1]}"/>'
        )
        c.text(xx + 20, by + bh - h - 5, str(passes), size=10, color=MUTED)
        c.text(xx + 20, by + bh + 14, f"{int(k * 100)}%", size=10, color=MUTED)
    c.text(
        bx + bw / 2,
        by + bh + 30,
        "fraction of the grid kept going into each fill  (total ~5.2k passes per batch)",
        size=10,
        color=MUTED,
    )

    # decode
    c.box(
        600,
        290,
        250,
        90,
        "VQ-VAE decoder",
        ["codes -> codebook lookup -> PatchDecoder", "faithful to the codes, soft"],
        kind="vqvae",
        body_size=11,
    )
    c.box(
        880,
        290,
        250,
        90,
        "GAN generator + AE decoder",
        ["codes -> fake latent [32,32,32]", "-> PatchDecoder: sharp"],
        kind="gan",
        body_size=11,
    )
    c.path("M1030,220 L1030,250 L725,250 L725,288")
    c.path("M1030,220 L1030,250 L1005,250 L1005,288")
    c.text(860, 244, "same code grid, decoded twice", size=11, color=MUTED)
    c.box(
        600,
        410,
        530,
        60,
        None,
        [
            "output PNG: every row is  [ VQ image | GAN image ]  for one sampled grid,",
            "plus the raw code grids as <name>.codes.pt so you can re-decode later",
        ],
        kind="infer",
        body_size=11,
    )
    c.arrow(725, 382, 725, 408)
    c.arrow(1005, 382, 1005, 408)
    c.text(
        590,
        515,
        "Why re-mask so aggressively?  The first fill is drawn from an all-masked grid, which the model finds hardest;",
        size=11,
        color=MUTED,
        italic=True,
    )
    c.text(
        590,
        532,
        "keeping just 1 % as anchors and redrawing the rest fixes global structure cheaply; later rounds polish locally.",
        size=11,
        color=MUTED,
        italic=True,
    )
    c.save("sampling")


# ============================================================================================
# 7. training loop
# ============================================================================================
def training_loop():
    c = Canvas(1180, 500, "training loop")
    c.text(590, 30, "One training loop for all stages  (latentgen/training/loop.py)", size=19, weight="700")
    c.text(
        590,
        52,
        "a Stage says what to do with one micro-batch and one step; the Trainer does everything else",
        size=12,
        color=MUTED,
        italic=True,
    )

    c.box(
        40,
        90,
        250,
        110,
        "data.batches(microbatch_size)",
        [
            "ImageBatches (stages 1a/1b)",
            "or EncodedDataset (stages 2/3)",
            "",
            "only the ids NOT seen this epoch",
        ],
        kind="data",
        body_size=11,
    )
    c.arrow(292, 145, 340, 145)
    c.box(
        345,
        90,
        250,
        110,
        "stage.microbatch(batch, scale)",
        [
            "forward under bf16 autocast",
            "(loss * scale).backward()",
            "scale = 1 / microbatches_per_step",
            "logger.log(...) -- no GPU sync",
        ],
        kind="plain",
        body_size=11,
    )
    c.path("M470,202 L470,230 L420,230 L420,202", arrow=True)
    c.text(
        545,
        232,
        "x microbatches_per_step  (batch_size / microbatch_size)",
        size=11,
        color=MUTED,
        anchor="start",
    )
    c.arrow(597, 145, 645, 145)
    c.box(
        650,
        90,
        230,
        110,
        "stage.step()",
        [
            "clip_grad_norm",
            "skip the step if non-finite",
            "optimizer.step(), EMA",
            "stage.on_step(step)  e.g. cutover",
        ],
        kind="plain",
        body_size=11,
    )
    c.arrow(882, 145, 930, 145)
    c.box(
        935,
        70,
        210,
        150,
        "every N steps",
        [
            "log_every: console + TensorBoard",
            "image_every: progress PNG",
            "save_every_seconds / _steps:",
            "   checkpoint",
            "",
            "Ctrl-C: save + exit",
        ],
        kind="plain",
        body_size=11,
    )

    # run dir
    c.box(
        40,
        280,
        440,
        200,
        "runs/<stage>/<run_name>/",
        [
            "config.yaml                   full config of this run",
            "checkpoints/step_00001234.pt  models, optimizers, EMA,",
            "                              epoch progress, seen ids",
            "tensorboard/                  tensorboard --logdir runs",
            "images/step_00000020.png      progress images",
            "",
            "--resume runs/<stage>/latest  newest run, newest step",
            "--resume runs/<stage>/<run>   that run, newest step",
            "--resume path/to/step_*.pt    that file",
        ],
        kind="encode",
        body_size=10.5,
        mono_body=True,
        align="start",
    )

    c.box(
        500,
        280,
        645,
        200,
        "What a checkpoint restores",
        [
            "weights (schedule-free runs save the averaged weights, the ones to sample with)",
            "optimizer state (dropped automatically if you changed a layer size)",
            "generator EMA, stage state (e.g. which GAN phase we were in)",
            "epoch, step, micro-batch count, and the set of image ids already seen this epoch",
            "",
            "so a resumed run finishes the SAME epoch on the images it had not seen yet,",
            "and `--set` can still change learning rate, batch size, epochs ... on resume.",
            "",
            "ModelManager = model + compiled forward + optimizer + EMA  (latentgen/training/manager.py)",
        ],
        kind="plain",
        body_size=11,
    )
    c.save("training_loop")


# ============================================================================================
# 8. tensor shapes / compression
# ============================================================================================
def data_shapes():
    c = Canvas(1180, 420, "tensor shapes through the pipeline")
    c.text(590, 30, "What one image becomes (FFHQ 512 settings)", size=19, weight="700")
    c.text(
        590,
        52,
        "arrows: what is produced from what, and what each model reads / writes",
        size=12,
        color=MUTED,
        italic=True,
    )
    c.image_icon(60, 120, 90, 72, "image")
    c.text(105, 226, "3 x 512 x 512", size=10, color=MUTED, mono=True)
    c.text(105, 240, "786,432 values", size=10, color=MUTED, mono=True)
    c.box(
        260,
        80,
        230,
        100,
        "VQ codes  (stage 1a)",
        ["32 x 32 = 1,024 ints in 0..255", "stored as int16", "768x smaller"],
        kind="vqvae",
        body_size=11,
    )
    c.box(
        260,
        210,
        230,
        100,
        "AE latent  (stage 1b)",
        ["32 x 32 x 32 = 32,768 values", "tanh, stored as int8", "24x smaller"],
        kind="ae",
        body_size=11,
    )
    c.arrow(155, 150, 258, 125)
    c.arrow(155, 170, 258, 255)
    c.box(
        600,
        80,
        260,
        100,
        "MaskGIT  (stage 2)",
        ["in: a code grid with holes", "out: 256 logits per position", "[32, 32, 256]"],
        kind="maskgit",
        body_size=11,
    )
    c.box(
        600,
        210,
        260,
        100,
        "GAN generator  (stage 3)",
        ["in: a code grid (+ noise)", "out: a latent of the AE's shape", "[32, 32, 32] in [-1, 1]"],
        kind="gan",
        body_size=11,
    )
    c.arrow(492, 130, 598, 130, "models p(codes)", label_dy=-8)
    c.arrow(492, 130, 598, 255, "conditions on", label_dy=14)
    c.arrow(598, 258, 492, 258, "imitates", label_dy=-8)
    c.box(
        920,
        120,
        220,
        120,
        "Decoding",
        ["codes -> VQ decoder -> image", "latent -> AE decoder -> image", "", "both produce 3 x 512 x 512"],
        kind="infer",
        body_size=11,
    )
    c.arrow(862, 260, 918, 200)
    c.box(
        60,
        340,
        1080,
        65,
        None,
        [
            "patch_size (16) is the one knob behind every shape: grid = image_size / patch_size. 256 px images give 16 x 16 grids (4x fewer tokens).",
            "codebook_size (256) is MaskGIT's vocabulary; the AE's bottleneck_dim (32) is what the GAN must produce per position.",
        ],
        kind="plain",
        body_size=11,
        dashed=True,
    )
    c.save("data_shapes")


if __name__ == "__main__":
    pipeline_overview()
    stage1_codecs()
    codebook_cutover()
    stage2_maskgit()
    stage3_cgan()
    sampling()
    training_loop()
    data_shapes()
