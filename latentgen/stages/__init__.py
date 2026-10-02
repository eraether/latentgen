"""One :class:`~latentgen.training.loop.Stage` per training script."""

from latentgen.stages.cgan import CGANStage
from latentgen.stages.codec import AutoencoderStage, VQVAEStage
from latentgen.stages.maskgit import MaskGITStage

__all__ = ["AutoencoderStage", "CGANStage", "MaskGITStage", "VQVAEStage"]
