"""Model shapes and checkpoint-compatible parameter names. Runs on CPU with tiny models."""

import torch

from latentgen.config import AutoencoderModelConfig, TransformerModelConfig, VQVAEModelConfig
from latentgen.nn import VQVAE, Autoencoder, Discriminator, Generator, MaskGIT
from latentgen.nn.maskgit import masked_cross_entropy, random_mask

TINY_VQ = VQVAEModelConfig(
    hidden_size=32,
    intermediate_size=64,
    num_encoder_layers=2,
    num_decoder_layers=2,
    bottleneck_dim=8,
    codebook_size=16,
    num_codebook_layers=1,
    num_attention_heads=4,
    patch_size=16,
)
TINY_AE = AutoencoderModelConfig(
    hidden_size=32,
    intermediate_size=64,
    num_encoder_layers=2,
    num_decoder_layers=2,
    bottleneck_dim=4,
    patch_size=16,
)
TINY_TF = TransformerModelConfig(
    hidden_size=32,
    intermediate_size=64,
    num_layers=2,
    num_attention_heads=4,
    codebook_size=16,
    grid_h=4,
    grid_w=4,
    bottleneck_dim=4,
)


def test_vqvae_shapes_and_cutover():
    model = VQVAE(TINY_VQ)
    x = torch.randn(2, 3, 64, 64)
    recon, codes, commitment = model(x)
    assert recon.shape == x.shape and codes.shape == (2, 4, 4) and commitment.ndim == 0
    assert codes.min() >= 0 and codes.max() < 16
    assert not model.bottleneck.codebook.is_cut_over
    model.bottleneck.codebook.cutover_to_learned_codebook()
    assert model.bottleneck.codebook.is_cut_over and bool(model.bottleneck.codebook.cutover)
    # after cutover encode/decode are deterministic
    assert torch.equal(model.encode(x), model.encode(x))
    assert model.decode(codes).shape == x.shape
    # the flag survives a state_dict round trip
    fresh = VQVAE(TINY_VQ)
    fresh.load_state_dict(model.state_dict())
    assert fresh.bottleneck.codebook.is_cut_over


def test_autoencoder_shapes():
    model = Autoencoder(TINY_AE)
    x = torch.randn(2, 3, 64, 64)
    recon, latents = model(x)
    assert recon.shape == x.shape and latents.shape == (2, 4, 4, 4)
    assert latents.min() >= -1 and latents.max() <= 1


def test_maskgit_shapes_and_loss():
    model = MaskGIT(TINY_TF)
    codes = torch.randint(0, 16, (3, 4, 4))
    mask, ratio = random_mask(3, 4, 4, 0.3, 1.0, codes.device)
    assert mask.shape == (3, 4, 4) and ratio.shape == (3,)
    assert torch.all(mask.flatten(1).sum(1) == (ratio * 16).round().long())
    logits = model(codes, mask)
    assert logits.shape == (3, 4, 4, 16)
    loss, ce_sum, correct_sum, n_masked = masked_cross_entropy(logits, codes, mask)
    assert loss.ndim == 0 and ce_sum.shape == (3,)
    assert torch.all(n_masked == mask.flatten(1).sum(1).float())


def test_gan_shapes():
    gen, disc = Generator(TINY_TF), Discriminator(TINY_TF)
    codes = torch.randint(0, 16, (2, 4, 4))
    fake = gen(codes)
    assert fake.shape == (2, 4, 4, 4) and fake.abs().max() <= 1
    assert disc(codes, fake).shape == (2, 1)


def test_parameter_names_are_stable():
    """State-dict keys are the checkpoint format: renaming any of these breaks loading saved checkpoints."""
    vq = set(VQVAE(TINY_VQ).state_dict())
    for key in [
        "encoder.patch_proj.weight",
        "encoder.layers.0.premlp_norm.weight",
        "encoder.layers.0.mlp.fused_proj.weight",
        "encoder.layers.0.mlp.down_proj.weight",
        "bottleneck.norm.weight",
        "bottleneck.down_proj_global.weight",
        "bottleneck.codebook.layers.0.self_attn.qkv_proj.weight",
        "bottleneck.codebook.layers.0.self_attn.out_proj.weight",
        "bottleneck.codebook.layers.0.mlp.fused_proj.weight",
        "bottleneck.codebook.layers.0.input_layernorm.weight",
        "bottleneck.codebook.layers.0.post_attention_layernorm.weight",
        "bottleneck.codebook.final_norm.weight",
        "bottleneck.codebook.final_out.weight",
        "bottleneck.codebook.cutover",
        "bottleneck.codebook.learned_codebook",
        "decoder.receptive_field_expansion.weight",
        "decoder.layers.0.premlp_norm.weight",
        "decoder.output_proj.weight",
    ]:
        assert key in vq, key
    ae = set(Autoencoder(TINY_AE).state_dict())
    for key in [
        "encoder.patch_proj.weight",
        "bottleneck.norm.weight",
        "bottleneck.down_proj_global.weight",
        "decoder.receptive_field_expansion.weight",
        "decoder.layers.1.mlp.down_proj.weight",
    ]:
        assert key in ae, key
    mg = set(MaskGIT(TINY_TF).state_dict())
    for key in [
        "code_embedding.weight",
        "mask_token",
        "layers.0.self_attn.qkv_proj.weight",
        "final_norm.weight",
        "final_out.weight",
    ]:
        assert key in mg, key
    assert not any(k.startswith("rope.") for k in mg), "RoPE tables must stay non-persistent"
    g = set(Generator(TINY_TF).state_dict())
    for key in ["lowres_quantized_code_embedding.weight", "noise_embedding.weight", "final_out.weight"]:
        assert key in g, key
    d = set(Discriminator(TINY_TF).state_dict())
    for key in ["lowres_quantized_code_embedding.weight", "highres_code_proj.weight", "final_out.weight"]:
        assert key in d, key


def test_parameter_order_is_stable():
    """Optimizer state is matched to parameters by POSITION, so registration order is part of the format too."""
    vq = [k for k, _ in VQVAE(TINY_VQ).named_parameters()]
    assert vq.index("decoder.layers.0.premlp_norm.weight") < vq.index("decoder.output_proj.weight")
    assert vq.index("decoder.output_proj.weight") < vq.index("decoder.receptive_field_expansion.weight")
    assert (
        vq.index("encoder.patch_proj.weight")
        < vq.index("bottleneck.norm.weight")
        < vq.index("decoder.layers.0.premlp_norm.weight")
    )
    cb = [k for k in vq if k.startswith("bottleneck.codebook.")]
    assert (
        cb[-1] == "bottleneck.codebook.learned_codebook" and cb[-2] == "bottleneck.codebook.final_out.weight"
    )
    layer = [k for k in vq if k.startswith("bottleneck.codebook.layers.0.")]
    assert layer == [
        "bottleneck.codebook.layers.0.self_attn.qkv_proj.weight",
        "bottleneck.codebook.layers.0.self_attn.out_proj.weight",
        "bottleneck.codebook.layers.0.mlp.fused_proj.weight",
        "bottleneck.codebook.layers.0.mlp.down_proj.weight",
        "bottleneck.codebook.layers.0.input_layernorm.weight",
        "bottleneck.codebook.layers.0.post_attention_layernorm.weight",
    ]
    g = [k for k, _ in Generator(TINY_TF).named_parameters()]
    assert g[0] == "lowres_quantized_code_embedding.weight" and g[-3:] == [
        "noise_embedding.weight",
        "final_norm.weight",
        "final_out.weight",
    ]
    d = [k for k, _ in Discriminator(TINY_TF).named_parameters()]
    assert d[:2] == ["lowres_quantized_code_embedding.weight", "highres_code_proj.weight"]
    m = [k for k, _ in MaskGIT(TINY_TF).named_parameters()]
    assert m[:2] == ["code_embedding.weight", "mask_token"] and m[-2:] == [
        "final_norm.weight",
        "final_out.weight",
    ]


def test_one_dimensional_codecs():
    """Audio: [B, 1, T] in, a 1 x T/p grid of codes / latents inside, [B, 1, T] out."""
    from latentgen.config import replace

    vq = VQVAE(replace(TINY_VQ, channels=1, dims=1))
    x = torch.randn(2, 1, 64)
    recon, codes, _ = vq(x)
    assert recon.shape == x.shape and codes.shape == (2, 1, 4)
    ae = Autoencoder(replace(TINY_AE, channels=1, dims=1))
    recon, latents = ae(x)
    assert recon.shape == x.shape and latents.shape == (2, 4, 1, 4)
    # 1-D models keep the same parameter names and order as 2-D ones (only the patch projection widths differ)
    assert [k for k, _ in vq.named_parameters()] == [k for k, _ in VQVAE(TINY_VQ).named_parameters()]


def test_rope_is_one_dimensional_for_audio_grids():
    """A 1 x T grid rotates every head dimension with the position (none wasted on the constant row)."""
    from latentgen.nn.layers import RoPE

    rope_1d = RoPE(head_dim=8, grid_h=1, grid_w=64, base=10000.0)
    _, sin = rope_1d()
    assert sin.shape == (64, 8) and bool((sin[1:].abs() > 0).all())
    rope_2d = RoPE(head_dim=8, grid_h=4, grid_w=4)
    _, sin = rope_2d()
    assert bool((sin[:4, :2] == 0).all())  # first row: the row-index half does not rotate


def test_audio_losses():
    from latentgen.data import ImageStats
    from latentgen.training.losses import psnr, stft_loss

    x = torch.randn(2, 1, 4096) * 0.1
    assert stft_loss(x, x).item() < 1e-5
    assert stft_loss(x * 0.5, x).item() > 0.1
    assert torch.isfinite(psnr(x * 0.9, x, ImageStats(mean=(0.0,), std=(1.0,))))
    assert stft_loss(x[..., :1024], x[..., :1024] * 0.5).item() > 0  # short clips skip the long FFTs
