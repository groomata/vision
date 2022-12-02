import torch
from einops.layers.torch import EinMix
from torch import nn


class PatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int = 16,
        channels: int = 3,
        embed_dim: int = 1024,
    ) -> None:
        super().__init__()

        self.patch_embed = EinMix(
            "b c (h ph) (w pw) -> b (h w) d",
            weight_shape="c ph pw d",
            bias_shape="d",
            c=channels,
            ph=patch_size,
            pw=patch_size,
            d=embed_dim,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.patch_embed(images)
