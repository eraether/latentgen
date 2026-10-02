"""MaskGIT sampling on a tiny untrained model: the mechanics, not the quality."""

import torch

from latentgen.config import TransformerModelConfig
from latentgen.nn import MaskGIT
from latentgen.sampling import fill_plan, forward_passes, generate_codes, resample


def test_fill_plan_counts():
    plan = fill_plan(16, sample_fraction=0.75, keep_schedule=(0.25, 0.5))
    assert plan == [(0.0, 12, 4), (0.25, 9, 3), (0.5, 6, 2)]
    assert forward_passes(plan) == 12 + 1 + 9 + 1 + 6 + 1


def test_generate_fills_every_slot():
    cfg = TransformerModelConfig(
        hidden_size=32,
        intermediate_size=64,
        num_layers=1,
        num_attention_heads=4,
        codebook_size=16,
        grid_h=4,
        grid_w=4,
    )
    model = MaskGIT(cfg).eval()
    passes = []
    codes = generate_codes(
        model, batch_size=3, sample_fraction=0.5, keep_schedule=(0.5,), on_pass=lambda: passes.append(1)
    )
    assert codes.shape == (3, 4, 4) and codes.dtype == torch.long
    assert codes.min() >= 0 and codes.max() < 16
    assert len(passes) == forward_passes(fill_plan(16, 0.5, (0.5,)))


def test_on_fill_and_resample():
    cfg = TransformerModelConfig(
        hidden_size=32,
        intermediate_size=64,
        num_layers=1,
        num_attention_heads=4,
        codebook_size=16,
        grid_h=4,
        grid_w=4,
    )
    model = MaskGIT(cfg).eval()
    rounds = []
    codes = generate_codes(model, 2, 0.5, (0.25, 0.5), on_fill=lambda i, keep, c: rounds.append((i, keep, c)))
    assert [(i, keep) for i, keep, _ in rounds] == [(0, 0.0), (1, 0.25), (2, 0.5)]
    assert torch.equal(rounds[-1][2], codes)
    keep = torch.zeros(2, 4, 4, dtype=torch.bool)
    keep[:, :2] = True  # keep the top half
    redrawn = resample(model, codes, keep, sample_fraction=0.5)
    assert torch.equal(redrawn[:, :2], codes[:, :2]) and redrawn.shape == codes.shape
