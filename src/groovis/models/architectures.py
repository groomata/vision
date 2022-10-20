import torch
from einops.layers.torch import Rearrange, Reduce
from torch import nn


class PatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int = 16,
        channels: int = 3,
        embed_dim: int = 1024,
    ) -> None:
        super().__init__()

        self.split_images = Rearrange(
            "b c (h ph) (w pw) -> b (h w) (c ph pw)",
            ph=patch_size,
            pw=patch_size,
        )

        self.projection = nn.Linear(
            in_features=channels * patch_size**2,
            out_features=embed_dim,
            bias=True,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        patches = self.split_images(images)
        representation = self.projection(patches)

        # Below code is equivalent to self.projection(patches)
        #
        # representation = torch.einsum(
        #     "b n p, d p -> b n d", patches, self.weight
        # )

        # representation += repeat(
        #     self.bias,
        #     "d -> b n d",
        #     b=representation.shape[0],
        #     n=representation.shape[1],
        # )

        return representation


class Architecture(nn.Module):
    def __init__(
        self,
        patch_size: int = 16,
        channels: int = 3,
        embed_dim: int = 1024,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            channels=channels,
            embed_dim=embed_dim,
        )
        self.pool = Reduce("b n d -> b d", "mean")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        representation = self.patch_embed(images)
        representation = self.pool(representation)
        return representation
