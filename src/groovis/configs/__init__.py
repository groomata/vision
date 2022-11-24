from dataclasses import dataclass, field
from typing import Any

from hydra.core.config_store import ConfigStore
from hydra_zen import make_custom_builds_fn
from omegaconf import MISSING

full_builds = make_custom_builds_fn(populate_full_signature=True)


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
    from .architecture import _register_configs as register_architecture_configs

    cs = ConfigStore.instance()

    cs.store(name="default", node=Config)

    register_architecture_configs()
