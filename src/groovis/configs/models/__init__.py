from enum import IntEnum

from hydra.core.config_store import ConfigStore

from groovis.configs import full_builds
from groovis.models import Architecture
from groovis.models.components.patch_embed import PatchEmbed


class Depth(IntEnum):
    SMALL = 8
    BASE = 12
    LARGE = 24


class EmbedDim(IntEnum):
    SMALL = 384
    BASE = 768
    LARGE = 1024


ArchitectureConfig = full_builds(Architecture)
PatchEmbedConfig = full_builds(PatchEmbed)


def _register_configs():
    cs = ConfigStore.instance()

    cs.store(
        group="architecture",
        name="simple_small",
        node=ArchitectureConfig(
            patch_embed=PatchEmbedConfig(
                embed_dim=EmbedDim.SMALL.value,
            ),
            backbone=None,
        ),
    )
    cs.store(
        group="architecture",
        name="simple_base",
        node=ArchitectureConfig(
            patch_embed=PatchEmbedConfig(
                embed_dim=EmbedDim.BASE.value,
            ),
            backbone=None,
        ),
    )
    cs.store(
        group="architecture",
        name="simple_large",
        node=ArchitectureConfig(
            patch_embed=PatchEmbedConfig(
                embed_dim=EmbedDim.LARGE.value,
            ),
            backbone=None,
        ),
    )
