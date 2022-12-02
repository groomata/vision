from hydra.core.config_store import ConfigStore

from groovis.configs import full_builds
from groovis.models import Architecture
from groovis.models.components.patch_embed import PatchEmbed

ArchitectureConfig = full_builds(Architecture)
PatchEmbedConfig = full_builds(PatchEmbed)

SmallArchitectureConfig = ArchitectureConfig(
    patch_embed=PatchEmbedConfig(
        embed_dim=384,
    ),
    backbone=None,
)
BaseArchitectureConfig = ArchitectureConfig(
    patch_embed=PatchEmbedConfig(
        embed_dim=768,
    ),
    backbone=None,
)
LargeArchitectureConfig = ArchitectureConfig(
    patch_embed=PatchEmbedConfig(
        embed_dim=1024,
    ),
    backbone=None,
)


def _register_configs():
    cs = ConfigStore.instance()

    cs.store(group="architecture", name="simple_small", node=SmallArchitectureConfig)
    cs.store(group="architecture", name="simple_base", node=BaseArchitectureConfig)
    cs.store(group="architecture", name="simple_large", node=LargeArchitectureConfig)
