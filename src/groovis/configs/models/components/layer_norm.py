from groovis.configs import partial_builds
from groovis.models.components.layer_norm import PostNorm, PreNorm

PreNormConfig = partial_builds(PreNorm)
PostNormConfig = partial_builds(PostNorm)
