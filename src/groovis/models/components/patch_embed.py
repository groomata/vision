import torch
from einops.layers.torch import EinMix
from torch import nn

from groovis.types import (
    ImageTensor,
    ImageToSequence,
    SequenceTensor,
    StrictInt,
    torchtyped,
)


class PatchEmbed(nn.Module):
    def __init__(
        self,
        image_size: StrictInt = 224,
        patch_size: StrictInt = 16,
        channels: StrictInt = 3,
        embed_dim: StrictInt = 1024,
    ) -> None:
        super().__init__()

        self.patch_embed: ImageToSequence = EinMix(
            "b c (h ph) (w pw) -> b (h w) d",
            weight_shape="c ph pw d",
            bias_shape="d",
            c=channels,
            ph=patch_size,
            pw=patch_size,
            d=embed_dim,
        )

        self.position_embedding = nn.Parameter(
            torch.randn((image_size // patch_size) ** 2, embed_dim) * 0.02
        )

    @torchtyped
    def forward(self, images: ImageTensor) -> SequenceTensor:
        return self.patch_embed(images) + self.position_embedding
