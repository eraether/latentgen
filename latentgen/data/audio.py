"""
Stage 1 input for audio: a folder of 16-bit PCM ``.wav`` files (mono; stereo is averaged).

Each clip is cropped / zero-padded to ``audio_length`` samples and returned as a ``[1, T]`` float
waveform in ``[-1, 1]``, normalised with the dataset stats. Files are read with the standard
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
    """Yields ``(normalised waveform [1, audio_length], index)``. Optional random sign flip as augmentation."""

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
            x = -x  # polarity inversion: the audio analogue of a horizontal flip
        return x, index


def waveform_image(wave_: torch.Tensor, height: int = 96, width: int = 1024) -> torch.Tensor:
    """Draw a ``[1, T]`` waveform in ``[-1, 1]`` as a ``[3, height, width]`` image (for progress images)."""
    x = wave_.detach().float().flatten().clamp(-1, 1)
    cols = max(1, x.numel() // width)
    x = x[: cols * width].view(width, cols)
    hi = ((1 - x.max(dim=1).values) * 0.5 * (height - 1)).long()
    lo = ((1 - x.min(dim=1).values) * 0.5 * (height - 1)).long()
    rows = torch.arange(height)[:, None]
    mask = (rows >= hi[None]) & (rows <= lo[None])  # [height, width]
    img = torch.ones(3, height, width)
    img[:, mask] = torch.tensor([0.15, 0.39, 0.92])[:, None]
    return img
