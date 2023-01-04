from einops import einsum, rearrange, unpack
from einops.layers.torch import EinMix
from torch import nn

from groovis.types import SequenceTensor


class SelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int = 1024,
        num_heads: int = 8,
        head_dim: int = 64,
    ) -> None:
        super().__init__()

        self.head_dim = head_dim
        self.num_heads = num_heads
        self.inner_dim = num_heads * head_dim

        self.to_qkv = EinMix(
            "b n d_in -> b n d_out",
            weight_shape="d_in d_out",
            bias_shape="d_out",
            d_in=embed_dim,
            d_out=3 * self.inner_dim,
        )

        self.proj = EinMix(
            "b n d_out -> b n d_in",
            weight_shape="d_out d_in",
            bias_shape="d_in",
            d_in=embed_dim,
            d_out=self.inner_dim,
        )

    def forward(self, representation: SequenceTensor) -> SequenceTensor:
        query, key, value = unpack(
            self.to_qkv(representation),
            [[self.inner_dim], [self.inner_dim], [self.inner_dim]],
            "b n *",
        )

        query, key, value = map(
            lambda x: rearrange(x, "... n (h d) -> ... h n d", h=self.num_heads),
            (query, key, value),
        )

        dots = einsum(query, key, "... q d, ... k d -> ... q k") * self.head_dim**-0.5
        attention = dots.softmax(dim=-1)
        out = einsum(attention, value, "... q k, ... k d -> ... q d")
        out = rearrange(out, "... h n d -> ... n (h d)")
        out = self.proj(out)

        return out
