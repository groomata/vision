import sys
from typing import Annotated, Callable

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias

import torch
from beartype import beartype
from beartype.vale import Is
from jaxtyping import Float, jaxtyped

StrictInt = Annotated[int, Is[lambda x: x > 0]]
StrictFloat = Annotated[float, Is[lambda x: x > 0]]


def torchtyped(fn):
    return jaxtyped(beartype(fn))


AnyTensor: TypeAlias = Float[torch.Tensor, "*shape"]
ImageTensor: TypeAlias = Float[torch.Tensor, "batch channel height width"]
SequenceTensor: TypeAlias = Float[torch.Tensor, "batch sequence feature"]
PooledTensor: TypeAlias = Float[torch.Tensor, "batch feature"]

ImageToSequence: TypeAlias = Callable[[ImageTensor], SequenceTensor]
SequenceToSequence: TypeAlias = Callable[[SequenceTensor], SequenceTensor]
SequenceToPooled: TypeAlias = Callable[[SequenceTensor], PooledTensor]
