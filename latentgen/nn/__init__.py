"""Model definitions. One file per model, shared blocks in ``layers.py`` / ``codec.py``."""

from latentgen.nn.autoencoder import Autoencoder
from latentgen.nn.cgan import Discriminator, Generator
from latentgen.nn.maskgit import MaskGIT
from latentgen.nn.vqvae import VQVAE

__all__ = ["Autoencoder", "Discriminator", "Generator", "MaskGIT", "VQVAE"]
