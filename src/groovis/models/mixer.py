import torch
from einops.layers.torch import EinMix
from torch import nn


class Mixer(nn.Module):
    def __init__(
        self,
        embed_dim: int = 1024,
    ) -> None:
        super().__init__()

        self.projection = EinMix(
            "b n d_in -> b n d_out",
            weight_shape="d_in d_out",
            bias_shape="d_out",
            d_in=embed_dim,
            d_out=embed_dim,
        )

    def forward(self, representation: torch.Tensor) -> torch.Tensor:
        return representation + self.projection(representation)
