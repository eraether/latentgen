"""
Device, precision and speed helpers shared by every stage.

* :func:`get_device` picks the GPU (and refuses to silently fall back to CPU).
* :func:`autocast` is the bf16 mixed-precision context every forward pass runs under.
* :func:`attention` runs scaled-dot-product attention with FlashAttention when the hardware
  supports it and falls back (once, with a warning) otherwise.
* :func:`maybe_compile` wraps a module/function in ``torch.compile`` when enabled.
"""

from __future__ import annotations

import contextlib
import random
import warnings
from collections.abc import Callable

import torch
import torch.nn.functional as F

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except ImportError:  # torch < 2.3
    SDPBackend = sdpa_kernel = None

_flash_ok = False  # set by probe_flash_attention()


def seed_everything(seed: int | None) -> int:
    """Seed python and torch (CPU + all GPUs). Returns the seed actually used."""
    if seed is None:
        seed = random.randrange(2**31)
    random.seed(seed)
    torch.manual_seed(seed)
    return seed


def get_device(allow_cpu: bool = False) -> torch.device:
    """Return ``cuda:0`` if available.

    Training these models on CPU is impractically slow, so without ``allow_cpu`` a missing GPU
    is an error with a pointer to the install instructions rather than a silent slowdown.
    """
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"Using GPU: {name} ({total_gb:.1f} GB)")
        return torch.device("cuda:0")
    if allow_cpu:
        warnings.warn(
            "No CUDA GPU found: running on CPU (only sensible for the tiny smoke-test config).", stacklevel=2
        )
        return torch.device("cpu")
    raise SystemExit(
        "No CUDA GPU detected. This pipeline needs a GPU. Install the PyTorch build that matches your "
        "driver (run ./install.sh or see README 'Installation'), or pass project.allow_cpu=true for a "
        "CPU smoke test."
    )


def configure_torch(matmul_precision: str = "high") -> None:
    """One-time global torch settings."""
    torch.set_float32_matmul_precision(matmul_precision)
    if hasattr(torch, "_dynamo"):
        torch._dynamo.config.cache_size_limit = 1024


def autocast(device: torch.device, enabled: bool = True):
    """bf16 autocast on CUDA; a no-op context on CPU."""
    if device.type != "cuda" or not enabled:
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def probe_flash_attention(device: torch.device) -> bool:
    """Decide once whether FlashAttention can be used (called from ``cli.setup`` before any model runs).

    FlashAttention needs CUDA, bf16/fp16 inputs and a supported GPU (Ampere or newer). Deciding
    here keeps :func:`attention` free of try/except, which matters because it runs inside
    ``torch.compile``-d models.
    """
    global _flash_ok
    _flash_ok = False
    if device.type == "cuda" and sdpa_kernel is not None:
        try:
            q = torch.randn(1, 2, 8, 16, device=device, dtype=torch.bfloat16)
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                F.scaled_dot_product_attention(q, q, q)
            _flash_ok = True
        except RuntimeError:
            _flash_ok = False
    if not _flash_ok:
        warnings.warn(
            "FlashAttention is unavailable (CPU, or a GPU older than Ampere). Falling back to PyTorch's default "
            "attention kernel: this works but is slower and uses more memory.",
            stacklevel=2,
        )
    return _flash_ok


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Non-causal scaled-dot-product attention on ``[B, heads, S, head_dim]`` tensors.

    Forces the FlashAttention kernel when :func:`probe_flash_attention` found it usable and the
    inputs are half precision (every training forward runs under bf16 autocast, so they are).
    """
    if _flash_ok and q.is_cuda and q.dtype in (torch.float16, torch.bfloat16):
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return F.scaled_dot_product_attention(q, k, v, is_causal=False)
    return F.scaled_dot_product_attention(q, k, v, is_causal=False)


def maybe_compile(fn: Callable, enabled: bool, **kwargs) -> Callable:
    """``torch.compile(fn)`` when enabled (and supported), else ``fn`` unchanged."""
    if not enabled or not hasattr(torch, "compile"):
        return fn
    return torch.compile(fn, **kwargs)


def count_parameters(model: torch.nn.Module, verbose: bool = False) -> int:
    """Number of trainable parameters; optionally prints a per-module breakdown."""
    total = 0
    rows = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        rows.append((name, p.numel()))
        total += p.numel()
    if verbose:
        width = max(len(n) for n, _ in rows) if rows else 10
        for name, n in rows:
            print(f"  {name:<{width}}  {n:>12,}")
    print(f"Trainable parameters: {total:,} ({total / 1e6:.1f} M)")
    return total
