from hydra.core.config_store import ConfigStore

from groovis.configs import full_builds, partial_builds
from groovis.models.vit import AlternatingBackbone, Feedforward, SelfAttention

from . import ArchitectureConfig, Depth, EmbedDim, PatchEmbedConfig
from .components.act_layer import GELUConfig
from .components.layer_norm import PreNormConfig

SelfAttentionConfig = partial_builds(
    SelfAttention,
)
FeedforwardConfig = partial_builds(
    Feedforward,
    expansion_factor=4,
    act_layer=GELUConfig,
)
AlternatingBackboneConfig = full_builds(AlternatingBackbone)


def _register_configs():
    cs = ConfigStore.instance()

    cs.store(
        group="architecture",
        name="vit_small",
        node=ArchitectureConfig(
            patch_embed=PatchEmbedConfig(
                embed_dim=EmbedDim.SMALL.value,
            ),
            backbone=AlternatingBackboneConfig(
                cross_location_block=SelfAttentionConfig(
                    embed_dim=EmbedDim.SMALL.value,
                    num_heads=6,
                ),
                per_location_block=FeedforwardConfig(
                    embed_dim=EmbedDim.SMALL.value,
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
        name="vit_base",
        node=ArchitectureConfig(
            patch_embed=PatchEmbedConfig(
                embed_dim=EmbedDim.BASE.value,
            ),
            backbone=AlternatingBackboneConfig(
                cross_location_block=SelfAttentionConfig(
                    embed_dim=EmbedDim.BASE.value,
                    num_heads=12,
                ),
                per_location_block=FeedforwardConfig(
                    embed_dim=EmbedDim.BASE.value,
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
        name="vit_large",
        node=ArchitectureConfig(
            patch_embed=PatchEmbedConfig(
                embed_dim=EmbedDim.LARGE.value,
            ),
            backbone=AlternatingBackboneConfig(
                cross_location_block=SelfAttentionConfig(
                    embed_dim=EmbedDim.LARGE.value,
                    num_heads=16,
                ),
                per_location_block=FeedforwardConfig(
                    embed_dim=EmbedDim.LARGE.value,
                ),
                norm=PreNormConfig(
                    embed_dim=EmbedDim.LARGE.value,
                ),
                depth=Depth.LARGE.value,
            ),
        ),
    )
