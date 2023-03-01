import torch
from einops import einsum, rearrange, unpack, pack
from einops.layers.torch import EinMix
from hydra_zen.typing import Partial
from torch import nn

from groovis.models.components.layer_norm import NormType
from groovis.types import (
    SequenceTensor,
    SequenceToSequence,
    StrictFloat,
    StrictInt,
    torchtyped,
)


class SelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: StrictInt = 1024,
        num_heads: StrictInt = 8,
    ) -> None:
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim

        self.to_qkv = EinMix(
            "b n d_in -> b n d_out",
            weight_shape="d_in d_out",
            bias_shape="d_out",
            d_in=embed_dim,
            d_out=3 * self.embed_dim,
        )

        self.proj = EinMix(
            "b n d_out -> b n d_in",
            weight_shape="d_out d_in",
            bias_shape="d_in",
            d_in=embed_dim,
            d_out=self.embed_dim,
        )

        # # WARNING: Incorrect!
        # self.relative_position_bias = nn.Parameter(torch.randn(num_heads, 196, 196))

    def forward(self, representation: SequenceTensor) -> SequenceTensor:
        query, key, value = unpack(
            self.to_qkv(representation),
            [[self.embed_dim], [self.embed_dim], [self.embed_dim]],
            "b n *",
        )

        query, key, value = map(
            lambda x: rearrange(x, "... n (h d) -> ... h n d", h=self.num_heads),
            (query, key, value),
        )

        dots = einsum(query, key, "... q d, ... k d -> ... q k") * self.head_dim**-0.5

        # dots += self.relative_position_bias

        attention = dots.softmax(dim=-1)
        out = einsum(attention, value, "... q k, ... k d -> ... q d")
        out = rearrange(out, "... h n d -> ... n (h d)")
        out = self.proj(out)

        return out


class Feedforward(nn.Module):
    """
    Block for per-location mixing operations
    """

    def __init__(
        self,
        embed_dim: StrictInt = 1024,
        expansion_factor: StrictFloat = 4,
        act_layer: Partial[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()

        self.block: SequenceToSequence = nn.Sequential(
            EinMix(
                "b n d_in -> b n d_out",
                weight_shape="d_in d_out",
                bias_shape="d_out",
                d_in=embed_dim,
                d_out=int(expansion_factor * embed_dim),
            ),
            act_layer(),
            EinMix(
                "b n d_out -> b n d_in",
                weight_shape="d_out d_in",
                bias_shape="d_in",
                d_in=embed_dim,
                d_out=int(expansion_factor * embed_dim),
            ),
        )

    @torchtyped
    def forward(self, representation: SequenceTensor) -> SequenceTensor:
        return self.block(representation)


class FusedTransformerBlock(nn.Module):
    """
    Implements [ViT-22B](https://arxiv.org/abs/2302.05442)
    """

    def __init__(
        self,
        embed_dim: StrictInt = 1024,
        expansion_factor: StrictFloat = 4,
        num_heads: StrictInt = 8,
        act_layer: Partial[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()

        self.expanded_dim = int(expansion_factor * embed_dim)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.proj_in = EinMix(
            "b n d_in -> b n d_out",
            weight_shape="d_in d_out",
            d_in=embed_dim,
            d_out=self.expanded_dim + 3 * embed_dim,
        )
        self.ff_bias = nn.Parameter(torch.zeros(self.expanded_dim))

        self.proj_out = EinMix(
            "b n d_out -> b n d_in",
            weight_shape="d_out d_in",
            bias_shape="d_in",
            d_in=embed_dim,
            d_out=self.expanded_dim + embed_dim,
        )

        self.act_layer = act_layer()

        self.query_norm = nn.LayerNorm(embed_dim)
        self.key_norm = nn.LayerNorm(embed_dim)

        # Implements Sub-LN in [Foundation Transformers](https://arxiv.org/abs/2210.06423)
        self.attention_norm = nn.LayerNorm(embed_dim)
        self.ff_norm = nn.LayerNorm(self.expanded_dim)

    @torchtyped
    def forward(self, representation: SequenceTensor) -> SequenceTensor:
        ff, query, key, value = unpack(
            self.proj_in(representation),
            [[self.expanded_dim], [self.embed_dim], [self.embed_dim], [self.embed_dim]],
            "b n *",
        )

        ff_out = self.act_layer(ff + self.ff_bias)

        # WARNING: Should be done head-wise!
        query, key, value = map(
            lambda x: rearrange(x, "... n (h d) -> ... h n d", h=self.num_heads),
            (self.query_norm(query), self.key_norm(key), value),
        )
        dots = einsum(query, key, "... q d, ... k d -> ... q k") * self.head_dim**-0.5
        attention = dots.softmax(dim=-1)
        attention_out = einsum(attention, value, "... q k, ... k d -> ... q d")
        attention_out = rearrange(attention_out, "... h n d -> ... n (h d)")

        out, _packed_shape = pack(
            [
                self.ff_norm(ff_out),
                self.attention_norm(attention_out),
            ],
            "b n *",
        )
        out = self.proj_out(out)
        return out


class CrossTokenMixerBlock(nn.Module):
    """
    Block for cross-location mixing operations
    """

    def __init__(
        self,
        seq_length: StrictInt = 196,
        expansion_factor: StrictFloat = 0.5,
        act_layer: Partial[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()

        self.block: SequenceToSequence = nn.Sequential(
            EinMix(
                "b n_in d -> b n_out d",
                weight_shape="n_in n_out",
                bias_shape="n_out",
                n_in=seq_length,
                n_out=int(expansion_factor * seq_length),
            ),
            act_layer(),
            EinMix(
                "b n_out d -> b n_in d",
                weight_shape="n_out n_in",
                bias_shape="n_in",
                n_in=seq_length,
                n_out=int(expansion_factor * seq_length),
            ),
        )

    @torchtyped
    def forward(self, representation: SequenceTensor) -> SequenceTensor:
        return self.block(representation)


class AlternatingBackbone(nn.Module):
    def __init__(
        self,
        cross_location_block: Partial[nn.Module],
        per_location_block: Partial[nn.Module],
        norm: Partial[NormType],
        depth: StrictInt = 24,
    ) -> None:
        super().__init__()

        self.blocks = nn.ModuleList([])
        for _ in range(depth):
            self.blocks.append(norm(block=cross_location_block()))
            self.blocks.append(norm(block=per_location_block()))

    @torchtyped
    def forward(self, representation: SequenceTensor) -> SequenceTensor:
        for block in self.blocks:
            representation = block(representation)
        return representation


class HomogeneousBackbone(nn.Module):
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
