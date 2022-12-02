from typing import Optional

import torch
from einops.layers.torch import Reduce
from torch import nn


class Architecture(nn.Module):
    def __init__(
        self,
        patch_embed: nn.Module,
        backbone: Optional[nn.Module],
    ) -> None:
        super().__init__()
        self.patch_embed = patch_embed
        self.backbone = backbone or nn.Identity()
        self.pool = Reduce("b n d -> b d", "mean")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        representation = self.patch_embed(images)
        representation = self.backbone(representation)
        representation = self.pool(representation)
        return representation
