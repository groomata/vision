import torch
from einops import einsum, pack, unpack
from einops.layers.torch import EinMix

x = torch.randn(196, 1000)
num_heads = 5
head_dim = 20

outs = []
for i in range(num_heads):
    to_qkv = EinMix(
        "n d_in -> n d_out",
        weight_shape="d_in d_out",
        bias_shape="d_out",
        d_in=1000,
        d_out=3 * head_dim,
    )

    query, key, value = unpack(
        to_qkv(x),
        [[head_dim], [head_dim], [head_dim]],
        "n *",
    )

    dots = einsum(query, key, "q d, k d -> q k") * head_dim**-0.5

    attention = dots.softmax(dim=-1)

    out = einsum(attention, value, "q k, k d -> q d")

    outs.append(out)

outs, packed_shape = pack(outs, "n *")
