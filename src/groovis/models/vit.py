from einops import einsum, rearrange, unpack
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
        embed_dim: int = 1024,
        num_heads: int = 8,
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
        attention = dots.softmax(dim=-1)
        out = einsum(attention, value, "... q k, ... k d -> ... q d")
        out = rearrange(out, "... h n d -> ... n (h d)")
        out = self.proj(out)

        return out


class PerTokenMixerBlock(nn.Module):
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
