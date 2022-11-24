from hydra.core.config_store import ConfigStore
from timm import create_model

from groovis.configs import full_builds
from groovis.models.architecture import Architecture

TimmModelConfig = full_builds(
    create_model,
    num_classes=0,
)

SmallViTTimmModelConfig = TimmModelConfig(model_name="vit_small_patch16_224")
BaseViTTimmModelConfig = TimmModelConfig(model_name="vit_base_patch16_224")
LargeViTTimmModelConfig = TimmModelConfig(model_name="vit_large_patch16_224")


ArchitectureConfig = full_builds(Architecture)

SmallArchitectureConfig = ArchitectureConfig(embed_dim=384)
BaseArchitectureConfig = ArchitectureConfig(embed_dim=768)
LargeArchitectureConfig = ArchitectureConfig(embed_dim=1024)


def _register_configs():
    cs = ConfigStore.instance()

    cs.store(group="architecture", name="small", node=SmallArchitectureConfig)
    cs.store(group="architecture", name="base", node=BaseArchitectureConfig)
    cs.store(group="architecture", name="large", node=LargeArchitectureConfig)
    cs.store(group="architecture", name="vit_small_timm", node=SmallViTTimmModelConfig)
    cs.store(group="architecture", name="vit_base_timm", node=BaseViTTimmModelConfig)
    cs.store(group="architecture", name="vit_large_timm", node=LargeViTTimmModelConfig)
