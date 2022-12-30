from torch import nn

from groovis.configs import partial_builds

GELUConfig = partial_builds(nn.GELU, approximate="tanh")
ReLUConfig = partial_builds(nn.ReLU)
SiLUConfig = partial_builds(nn.SiLU)
