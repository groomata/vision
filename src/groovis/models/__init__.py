from typing import Optional

import torch
from einops.layers.torch import Reduce
from torch import nn

from groovis.types import (
    ImageTensor,
    ImageToSequence,
    PooledTensor,
    SequenceToPooled,
    SequenceToSequence,
)


class Architecture(nn.Module):
    def __init__(
        self,
        patch_embed: nn.Module,
        backbone: Optional[nn.Module],
    ) -> None:
        super().__init__()
        self.patch_embed: ImageToSequence = patch_embed
        self.backbone: SequenceToSequence = backbone or nn.Identity()
        self.pool: SequenceToPooled = Reduce("b n d -> b d", "mean")

    def forward(self, images: ImageTensor) -> PooledTensor:
        representation = self.patch_embed(images)
        representation = self.backbone(representation)
        representation = self.pool(representation)
        return representation
