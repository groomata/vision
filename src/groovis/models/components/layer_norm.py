import sys
from typing import Union

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias

from torch import nn

from groovis.types import AnyTensor, StrictInt


class PreNorm(nn.Module):
    def __init__(self, block: nn.Module, embed_dim: StrictInt = 1024) -> None:
        super().__init__()

        self.block = block
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, representation: AnyTensor) -> AnyTensor:
        return representation + self.block(self.norm(representation))


class PostNorm(nn.Module):
    def __init__(self, block: nn.Module, embed_dim: StrictInt = 1024) -> None:
        super().__init__()

        self.block = block
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, representation: AnyTensor) -> AnyTensor:
        return self.norm(representation + self.block(representation))


NormType: TypeAlias = Union[PreNorm, PostNorm]
