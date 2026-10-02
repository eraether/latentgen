"""
latentgen -- a four-stage latent image generation pipeline.

    1a  VQ-VAE          image  <-> 32x32 grid of discrete codes          (latentgen.nn.vqvae)
    1b  Autoencoder     image  <-> 32x32x32 continuous latent            (latentgen.nn.autoencoder)
    1c  Encode dataset  every image -> (codes, latent) stored once       (scripts/01c_encode_dataset.py)
    2   MaskGIT         learns the distribution of code grids            (latentgen.nn.maskgit)
    3   cGAN            codes -> AE latent, adds the detail VQ lost      (latentgen.nn.cgan)
    gen MaskGIT samples a code grid, decoded via VQ-VAE and via GAN + AE (latentgen.sampling)

Start with README.md; every module docstring explains its piece.
"""

__version__ = "2.0.0"
