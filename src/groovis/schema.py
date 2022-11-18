from dataclasses import dataclass, field
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from timm import create_model

from groovis.models.architecture import Architecture


def import_path(x: Any):
    return f"{x.__module__}.{x.__name__}"


@dataclass
class TimmModelConfig:
    _target_: str = import_path(create_model)
    model_name: str = MISSING
    num_classes: int = 0


@dataclass
class SmallViTTimmModelConfig:
    model_name = "vit_small_patch16_224"


@dataclass
class BaseViTTimmModelConfig:
    model_name = "vit_base_patch16_224"


@dataclass
class LargeViTTimmModelConfig:
    model_name = "vit_large_patch16_224"


@dataclass
class ArchitectureConfig:
    _target_: str = import_path(Architecture)
    patch_size: int = 16
    channels: int = 3
    embed_dim: int = MISSING


@dataclass
class SmallArchitectureConfig(ArchitectureConfig):
    embed_dim: int = 384


@dataclass
class BaseArchitectureConfig(ArchitectureConfig):
    embed_dim: int = 768


@dataclass
class LargeArchitectureConfig(ArchitectureConfig):
    embed_dim: int = 1024


defaults = [
    "_self_",
    {"architecture": "base"},
]


@dataclass
class Config:
    defaults: list[Any] = field(default_factory=lambda: defaults)
    architecture: Any = MISSING
    base_lr: float = 0.005
    warmup_lr: float = 0.000005
    warmup_epochs: int = 0
    batch_size: int = 32
    epochs: int = 1
    patience: int = 4
    clip_grad: float = 0.5
    temperature: float = 0.1
    log_interval: int = 10
    save_top_k: int = 3
    run_name: str = "default-test"
    offline: bool = True


def register_configs():
    cs = ConfigStore.instance()

    cs.store(name="default", node=Config)
    cs.store(group="architecture", name="small", node=SmallArchitectureConfig)
    cs.store(group="architecture", name="base", node=BaseArchitectureConfig)
    cs.store(group="architecture", name="large", node=LargeArchitectureConfig)
    cs.store(group="architecture", name="vit_small_timm", node=SmallViTTimmModelConfig)
    cs.store(group="architecture", name="vit_base_timm", node=BaseViTTimmModelConfig)
    cs.store(group="architecture", name="vit_large_timm", node=LargeViTTimmModelConfig)
