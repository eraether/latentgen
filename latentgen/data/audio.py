"""
Stage 1 input for audio: a folder of 16-bit PCM ``.wav`` files (mono; stereo is averaged).

Each clip is cropped / zero-padded to ``audio_length`` samples and returned as a ``[1, T]`` float
waveform in ``[-1, 1]``, normalized with the dataset stats. Files are read with the standard
library only, so there is no extra dependency; ``examples/prepare_hf_dataset.py`` writes this format
from any Hugging Face audio dataset (resampled to one rate).
"""

from __future__ import annotations

import wave
from pathlib import Path

import torch
from torch.utils.data import Dataset

from latentgen.data.normalization import ImageStats


def list_audio(audio_dir: str | Path) -> list[Path]:
    audio_dir = Path(audio_dir)
    if not audio_dir.is_dir():
        raise FileNotFoundError(f"audio folder not found: {audio_dir} (set data.image_dir in your config)")
    files = sorted(audio_dir.rglob("*.wav"))
    if not files:
        raise FileNotFoundError(f"no .wav files under {audio_dir}")
    return files


def load_wav(path: str | Path, length: int | None = None) -> torch.Tensor:
    """Read a 16-bit PCM wav as a mono ``[1, T]`` float tensor in ``[-1, 1]`` (cropped / padded to ``length``)."""
    with wave.open(str(path), "rb") as f:
        if f.getsampwidth() != 2:
            raise ValueError(
                f"{path}: only 16-bit PCM wav files are supported (got {8 * f.getsampwidth()}-bit)"
            )
        channels = f.getnchannels()
        data = torch.frombuffer(bytearray(f.readframes(f.getnframes())), dtype=torch.int16)
    wave_ = data.float().div_(32768.0).view(-1, channels).mean(dim=1)  # [T], stereo -> mono
    if length is not None:
        wave_ = wave_[:length]
        if wave_.numel() < length:
            wave_ = torch.nn.functional.pad(wave_, (0, length - wave_.numel()))
    return wave_[None]


def save_wav(path: str | Path, wave_: torch.Tensor, sample_rate: int) -> None:
    """Write a ``[1, T]`` (or ``[T]``) float waveform in ``[-1, 1]`` as 16-bit PCM wav."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pcm = (wave_.detach().float().flatten().clamp(-1, 1) * 32767).round().to(torch.int16).cpu()
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(pcm.numpy().tobytes())


class AudioFolderDataset(Dataset):
    """Yields ``(normalized waveform [1, audio_length], index)``. Optional random sign flip as augmentation."""

    def __init__(
        self, audio_dir: str | Path, audio_length: int, stats: ImageStats, flip: bool = True
    ) -> None:
        self.files = list_audio(audio_dir)
        self.audio_length = audio_length
        self.stats = stats
        self.flip = flip

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        x = self.stats.normalize(load_wav(self.files[index], self.audio_length), batched=False)
        if self.flip and torch.rand(()) < 0.5:
            x = -x  # polarity inversion: the audio analog of a horizontal flip
        return x, index


def waveform_image(wave_: torch.Tensor, height: int = 96, width: int = 1024) -> torch.Tensor:
    """Draw a ``[1, T]`` waveform in ``[-1, 1]`` as a ``[3, height, width]`` image (for progress images)."""
    x = wave_.detach().float().flatten().cpu().clamp(-1, 1)
    cols = max(1, x.numel() // width)
    x = x[: cols * width].view(width, cols)
    hi = ((1 - x.max(dim=1).values) * 0.5 * (height - 1)).long()
    lo = ((1 - x.min(dim=1).values) * 0.5 * (height - 1)).long()
    rows = torch.arange(height)[:, None]
    mask = (rows >= hi[None]) & (rows <= lo[None])  # [height, width]
    img = torch.ones(3, height, width)
    img[:, mask] = torch.tensor([0.15, 0.39, 0.92])[:, None]
    return img


# a perceptually ordered dark-to-bright palette (black -> indigo -> magenta -> orange -> pale yellow)
_PALETTE = torch.tensor(
    [
        [0.00, 0.00, 0.02],
        [0.23, 0.06, 0.44],
        [0.55, 0.13, 0.51],
        [0.87, 0.29, 0.38],
        [0.99, 0.62, 0.36],
        [0.99, 0.99, 0.75],
    ]
)


def colorize(x: torch.Tensor) -> torch.Tensor:
    """``[H, W]`` values in ``[0, 1]`` -> ``[3, H, W]`` RGB through :data:`_PALETTE`."""
    pos = x.clamp(0, 1) * (len(_PALETTE) - 1)
    lo = pos.floor().long().clamp_(max=len(_PALETTE) - 2)
    frac = (pos - lo).unsqueeze(-1)
    rgb = _PALETTE[lo] * (1 - frac) + _PALETTE[lo + 1] * frac  # [H, W, 3]
    return rgb.permute(2, 0, 1)


def spectrogram_image(
    wave_: torch.Tensor, height: int = 128, width: int = 1024, n_fft: int = 512, floor_db: float = -80.0
) -> torch.Tensor:
    """Log-magnitude spectrogram of a ``[1, T]`` waveform as a ``[3, height, width]`` image (low frequencies at the bottom)."""
    x = wave_.detach().float().flatten().cpu()
    spec = torch.stft(
        x, n_fft, hop_length=n_fft // 4, window=torch.hann_window(n_fft), return_complex=True
    ).abs()
    db = 20 * spec.clamp_min(1e-5).log10()
    ref = max(float(db.max()), -40.0)  # a silent clip stays dark instead of being stretched into noise
    db = (db - ref).clamp(floor_db, 0) / -floor_db + 1  # [0, 1], 0 = floor_db below the reference
    img = torch.nn.functional.interpolate(db.flip(0)[None, None], size=(height, width), mode="bilinear")[0, 0]
    return colorize(img)


def audio_panel(waves: list[torch.Tensor], width: int = 1024) -> torch.Tensor:
    """Waveform over spectrogram for each ``[1, T]`` clip, stacked top to bottom (progress / generation images)."""
    rows = []
    for w in waves:
        rows += [
            waveform_image(w.cpu(), width=width),
            spectrogram_image(w, width=width),
            torch.ones(3, 6, width),
        ]
    return torch.cat(rows[:-1], dim=-2)
