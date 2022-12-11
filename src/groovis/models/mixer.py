from einops.layers.torch import EinMix
from hydra_zen.typing import Partial
from torch import nn

from groovis.types import SequenceTensor, SequenceToSequence, StrictInt, torchtyped


class MixerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: StrictInt = 1024,
    ) -> None:
        super().__init__()

        self.projection: SequenceToSequence = EinMix(
            "b n d_in -> b n d_out",
            weight_shape="d_in d_out",
            bias_shape="d_out",
            d_in=embed_dim,
            d_out=embed_dim,
        )

    @torchtyped
    def forward(self, representation: SequenceTensor) -> SequenceTensor:
        return representation + self.projection(representation)


class Mixer(nn.Module):
    def __init__(
        self,
        block: Partial[nn.Module],
        depth: StrictInt = 24,
    ) -> None:
        super().__init__()

        self.blocks = nn.ModuleList([block() for _ in range(depth)])

    @torchtyped
    def forward(self, representation: SequenceTensor) -> SequenceTensor:
        for block in self.blocks:
            representation = block(representation)
        return representation
