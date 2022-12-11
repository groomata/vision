from hydra.core.config_store import ConfigStore

from groovis.configs import full_builds, partial_builds
from groovis.models.mixer import Mixer, MixerBlock

from . import ArchitectureConfig, Depth, EmbedDim, PatchEmbedConfig
from .components.act_layer import GELUConfig

MixerBlockConfig = partial_builds(MixerBlock)
MixerConfig = full_builds(Mixer)


def _register_configs():
    cs = ConfigStore.instance()

    cs.store(
        group="architecture",
        name="mixer_small",
        node=ArchitectureConfig(
            patch_embed=PatchEmbedConfig(
                embed_dim=EmbedDim.SMALL.value,
            ),
            backbone=MixerConfig(
                block=MixerBlockConfig(
                    embed_dim=EmbedDim.SMALL.value,
                    act_layer=GELUConfig,
                ),
                depth=Depth.SMALL.value,
            ),
        ),
    )
    cs.store(
        group="architecture",
        name="mixer_base",
        node=ArchitectureConfig(
            patch_embed=PatchEmbedConfig(
                embed_dim=EmbedDim.BASE.value,
            ),
            backbone=MixerConfig(
                block=MixerBlockConfig(
                    embed_dim=EmbedDim.BASE.value,
                    act_layer=GELUConfig,
                ),
                depth=Depth.BASE.value,
            ),
        ),
    )
    cs.store(
        group="architecture",
        name="mixer_large",
        node=ArchitectureConfig(
            patch_embed=PatchEmbedConfig(
                embed_dim=EmbedDim.LARGE.value,
            ),
            backbone=MixerConfig(
                block=MixerBlockConfig(
                    embed_dim=EmbedDim.LARGE.value,
                    act_layer=GELUConfig,
                ),
                depth=Depth.LARGE.value,
            ),
        ),
    )
