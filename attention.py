import torch
from einops import einsum, rearrange, unpack
from einops.layers.torch import EinMix

x = torch.randn(196, 1000)
num_heads = 5
head_dim = 20
inner_dim = num_heads * head_dim


to_qkv = EinMix(
    "n d_in -> n d_out",
    weight_shape="d_in d_out",
    bias_shape="d_out",
    d_in=1000,
    d_out=3 * inner_dim,
)

query, key, value = unpack(
    to_qkv(x),
    [[inner_dim], [inner_dim], [inner_dim]],
    "n *",
)

query, key, value = map(
    lambda x: rearrange(x, "n (h d) -> h n d", h=num_heads),
    (query, key, value),
)

dots = einsum(query, key, "... q d, ... k d -> ... q k") * head_dim**-0.5

attention = dots.softmax(dim=-1)

out = einsum(attention, value, "... q k, ... k d -> ... q d")

out = rearrange(out, "h n d -> n (h d)")

print(out.shape)
