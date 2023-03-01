from hydra.core.config_store import ConfigStore

from groovis.configs import full_builds, partial_builds
from groovis.models.vit import HomogeneousBackbone, FusedTransformerBlock

from . import ArchitectureConfig, Depth, EmbedDim, PatchEmbedConfig
from .components.act_layer import GELUConfig
from .components.layer_norm import PreNormConfig

FusedTransformerBlockConfig = partial_builds(
    FusedTransformerBlock,
    expansion_factor=4,
    act_layer=GELUConfig,
)
HomogeneousBackboneConfig = full_builds(HomogeneousBackbone)


def _register_configs():
    cs = ConfigStore.instance()

    cs.store(
        group="architecture",
        name="parallel_vit_small",
        node=ArchitectureConfig(
            patch_embed=PatchEmbedConfig(
                embed_dim=EmbedDim.SMALL.value,
            ),
            backbone=HomogeneousBackboneConfig(
                block=FusedTransformerBlockConfig(
                    embed_dim=EmbedDim.SMALL.value,
                    num_heads=6,
                ),
                norm=PreNormConfig(
                    embed_dim=EmbedDim.SMALL.value,
                ),
                depth=Depth.SMALL.value,
            ),
        ),
    )
    cs.store(
        group="architecture",
        name="parallel_vit_base",
        node=ArchitectureConfig(
            patch_embed=PatchEmbedConfig(
                embed_dim=EmbedDim.BASE.value,
            ),
            backbone=HomogeneousBackboneConfig(
                block=FusedTransformerBlockConfig(
                    embed_dim=EmbedDim.BASE.value,
                    num_heads=12,
                ),
                norm=PreNormConfig(
                    embed_dim=EmbedDim.BASE.value,
                ),
                depth=Depth.BASE.value,
            ),
        ),
    )
    cs.store(
        group="architecture",
        name="parallel_vit_large",
        node=ArchitectureConfig(
            patch_embed=PatchEmbedConfig(
                embed_dim=EmbedDim.LARGE.value,
            ),
            backbone=HomogeneousBackboneConfig(
                block=FusedTransformerBlockConfig(
                    embed_dim=EmbedDim.LARGE.value,
                    num_heads=16,
                ),
                norm=PreNormConfig(
                    embed_dim=EmbedDim.LARGE.value,
                ),
                depth=Depth.LARGE.value,
            ),
        ),
    )
