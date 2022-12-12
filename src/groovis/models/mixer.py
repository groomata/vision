from einops.layers.torch import EinMix
from hydra_zen.typing import Partial
from torch import nn

from groovis.models.components.layer_norm import NormType
from groovis.types import SequenceTensor, SequenceToSequence, StrictInt, torchtyped


class MixerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: StrictInt = 1024,
        act_layer: Partial[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()

        self.block: SequenceToSequence = nn.Sequential(
            EinMix(
                "b n d_in -> b n d_out",
                weight_shape="d_in d_out",
                bias_shape="d_out",
                d_in=embed_dim,
                d_out=embed_dim,
            ),
            act_layer(),
        )

    @torchtyped
    def forward(self, representation: SequenceTensor) -> SequenceTensor:
        return self.block(representation)


class Mixer(nn.Module):
    def __init__(
        self,
        block: Partial[nn.Module],
        norm: Partial[NormType],
        depth: StrictInt = 24,
    ) -> None:
        super().__init__()

        self.blocks = nn.ModuleList([norm(block=block()) for _ in range(depth)])

    @torchtyped
    def forward(self, representation: SequenceTensor) -> SequenceTensor:
        for block in self.blocks:
            representation = block(representation)
        return representation
