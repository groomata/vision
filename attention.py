import torch
from einops import einsum
from einops.layers.torch import EinMix

x = torch.randn(196, 1000)
inner_dim = 100

query_generator = EinMix(
    "n d_in -> n d_out",
    weight_shape="d_in d_out",
    bias_shape="d_out",
    d_in=1000,
    d_out=inner_dim,
)

key_generator = EinMix(
    "n d_in -> n d_out",
    weight_shape="d_in d_out",
    bias_shape="d_out",
    d_in=1000,
    d_out=inner_dim,
)

value_generator = EinMix(
    "n d_in -> n d_out",
    weight_shape="d_in d_out",
    bias_shape="d_out",
    d_in=1000,
    d_out=inner_dim,
)


query = query_generator(x)
key = key_generator(x)
value = value_generator(x)

dots = einsum(query, key, "q d, k d -> q k") * inner_dim**-0.5

attention = dots.softmax(dim=-1)

out = einsum(attention, value, "q k, k d -> q d")

print(out.shape)
