from hydra.core.config_store import ConfigStore

from groovis.configs import full_builds
from groovis.models.mixer import Mixer

from . import ArchitectureConfig, EmbedDim, PatchEmbedConfig

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
                embed_dim=EmbedDim.SMALL.value,
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
                embed_dim=EmbedDim.BASE.value,
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
                embed_dim=EmbedDim.LARGE.value,
            ),
        ),
    )
