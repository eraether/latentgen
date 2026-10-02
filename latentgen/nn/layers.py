"""
Building blocks shared by all four models.

Two families live here:

* **Sequence blocks** (``[B, S, C]`` tensors) for the transformers: :class:`Attention`,
  :class:`FusedSwiGLU`, :class:`TransformerLayer`, :class:`RoPE`.
* **Feature-map blocks** (``[B, C, H, W]`` tensors) for the convolutional-free image codecs:
  :class:`RMSNorm2D`, :class:`FusedSwiGLU2D`, :class:`MLPBlock2D`.

Attribute names (``qkv_proj``, ``fused_proj``, ``premlp_norm`` ...) are part of the checkpoint
format: they are what ``state_dict()`` keys are made of, so renaming one breaks loading every
checkpoint written before the rename. Rename with care.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from latentgen.device import attention

# ----------------------------------------------------------------------------- sequence blocks


class FusedSwiGLU(nn.Module):
    """SwiGLU MLP: ``down(silu(gate(x)) * up(x))`` with gate+up fused into one Linear."""

    def __init__(self, hidden_size: int, intermediate_size: int, output_size: int | None = None) -> None:
        super().__init__()
        self.fused_proj = nn.Linear(hidden_size, 2 * intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, output_size or hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.fused_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class RoPE(nn.Module):
    """Rotary position embedding for a ``grid_h x grid_w`` token grid.

    * 2-D grids (images): axial RoPE -- half of each head's dimensions rotate with the row index,
      the other half with the column index, so attention can tell "two tokens up" from "two tokens
      left".
    * 1-D sequences (audio, ``grid_h == 1``): every dimension rotates with the position. Using the
      axial layout here would waste half of each head on a row index that is always 0.

    ``base`` sets the longest wavelength: about the length of the longest axis is right, so ~100 for a
    32-wide image grid and ~10000 for a 1024-long audio sequence. The tables are non-persistent
    buffers: rebuilt from the config, never stored in checkpoints.
    """

    def __init__(self, head_dim: int, grid_h: int, grid_w: int, base: float = 100.0) -> None:
        super().__init__()
        one_d = grid_h == 1
        if head_dim % (2 if one_d else 4):
            raise ValueError(f"head_dim={head_dim} must be divisible by {2 if one_d else 4} for RoPE")
        if one_d:
            inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
            freqs = torch.arange(grid_w, dtype=torch.float32)[:, None] * inv_freq  # [S, head_dim/2]
        else:
            d_axis = head_dim // 2
            inv_freq = 1.0 / (base ** (torch.arange(0, d_axis, 2, dtype=torch.float32) / d_axis))
            ys, xs = torch.meshgrid(
                torch.arange(grid_h, dtype=torch.float32),
                torch.arange(grid_w, dtype=torch.float32),
                indexing="ij",
            )
            freqs = torch.cat(
                [ys.reshape(-1, 1) * inv_freq, xs.reshape(-1, 1) * inv_freq], dim=-1
            )  # [S, head_dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [S, head_dim]
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)
        self.num_positions = grid_h * grid_w

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.cos, self.sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """``x: [B, heads, S, head_dim]``, ``cos/sin: [S, head_dim]``."""
    cos = cos.to(x.dtype)[None, None]
    sin = sin.to(x.dtype)[None, None]
    return x * cos + _rotate_half(x) * sin


class Attention(nn.Module):
    """Multi-head non-causal self-attention with a fused QKV projection and optional RoPE."""

    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError(f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor | None = None, sin: torch.Tensor | None = None
    ) -> torch.Tensor:
        B, S, C = x.shape
        q, k, v = self.qkv_proj(x).view(B, S, 3, self.num_heads, self.head_dim).unbind(2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # [B, heads, S, head_dim]
        if cos is not None:
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)
        out = attention(q, k, v)
        return self.out_proj(out.transpose(1, 2).reshape(B, S, C))


class TransformerLayer(nn.Module):
    """Pre-norm transformer block: ``x + attn(norm(x))`` then ``x + mlp(norm(x))``."""

    def __init__(self, hidden_size: int, intermediate_size: int, num_attention_heads: int) -> None:
        super().__init__()
        self.self_attn = Attention(hidden_size, num_attention_heads)
        self.mlp = FusedSwiGLU(hidden_size, intermediate_size)
        self.input_layernorm = nn.RMSNorm(hidden_size)
        self.post_attention_layernorm = nn.RMSNorm(hidden_size)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor | None = None, sin: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


# ----------------------------------------------------------------------------- feature-map blocks


class RMSNorm2D(nn.Module):
    """RMSNorm over the channel dimension of a ``[B, C, H, W]`` tensor.

    ``nn.RMSNorm`` normalizes the *last* dimension, which is wrong for NCHW feature maps.
    The reduction is done in fp32 so it is stable under bf16 autocast.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * x.pow(2).mean(dim=1, keepdim=True).add(self.eps).rsqrt()
        return x.to(dtype) * self.weight.view(1, -1, 1, 1)


class FusedSwiGLU2D(nn.Module):
    """:class:`FusedSwiGLU` applied per pixel of a ``[B, C, H, W]`` tensor via 1x1 convolutions."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.fused_proj = nn.Conv2d(hidden_size, 2 * intermediate_size, kernel_size=1, bias=False)
        self.down_proj = nn.Conv2d(intermediate_size, hidden_size, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.fused_proj(x).chunk(2, dim=1)
        return self.down_proj(F.silu(gate) * up)


class MLPBlock2D(nn.Module):
    """Pre-norm residual per-pixel MLP block: ``x + mlp(norm(x))``. No spatial mixing."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.premlp_norm = RMSNorm2D(hidden_size)
        self.mlp = FusedSwiGLU2D(hidden_size, intermediate_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(self.premlp_norm(x))


# ----------------------------------------------------------------------------- helpers


def run_layers(
    layers: nn.ModuleList, x: torch.Tensor, *args: torch.Tensor, checkpointing: bool = False
) -> torch.Tensor:
    """Apply a stack of layers in order, optionally with activation checkpointing.

    Activation checkpointing drops each layer's intermediate activations after the forward
    pass and recomputes them during backward. It cuts activation memory roughly by the depth
    of the stack at the cost of ~30% more compute. Only active when gradients are enabled.
    """
    use_ckpt = checkpointing and torch.is_grad_enabled()
    for layer in layers:
        if use_ckpt:
            x = activation_checkpoint(layer, x, *args, use_reentrant=False)
        else:
            x = layer(x, *args)
    return x


def transformer_stack(
    num_layers: int, hidden_size: int, intermediate_size: int, num_heads: int
) -> nn.ModuleList:
    return nn.ModuleList(
        TransformerLayer(hidden_size, intermediate_size, num_heads) for _ in range(num_layers)
    )


def mlp2d_stack(num_layers: int, hidden_size: int, intermediate_size: int) -> nn.ModuleList:
    return nn.ModuleList(MLPBlock2D(hidden_size, intermediate_size) for _ in range(num_layers))
