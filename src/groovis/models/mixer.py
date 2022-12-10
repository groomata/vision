from einops.layers.torch import EinMix
from torch import nn

from groovis.types import SequenceTensor, SequenceToSequence, StrictInt, torchtyped


class Mixer(nn.Module):
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
