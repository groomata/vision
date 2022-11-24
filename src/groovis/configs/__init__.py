import pkgutil
from dataclasses import dataclass, field
from typing import Any

from hydra.core.config_store import ConfigStore
from hydra_zen import make_custom_builds_fn, to_yaml
from omegaconf import MISSING
from rich import print


def print_yaml(config: Any):
    print(to_yaml(config))


partial_builds = make_custom_builds_fn(populate_full_signature=True, zen_partial=True)
full_builds = make_custom_builds_fn(populate_full_signature=True)


defaults = [
    "_self_",
    {"architecture": "base"},
    {"loss": "nt_xent_medium"},
    {"datamodule": "imagenet"},
    {"datamodule/dataset": "imagenette"},
    {"datamodule/dataset/transforms": "relaxed"},
    {"datamodule/dataloader": "base"},
    {"trainer": "auto"},
    {"trainer/logger": "wandb"},
    {"trainer/callbacks": "default"},
    {"optimizer": "adam"},
    {"scheduler": "onecycle"},
]


@dataclass
class Config:
    defaults: list[Any] = field(default_factory=lambda: defaults)
    architecture: Any = MISSING
    loss: Any = MISSING
    datamodule: Any = MISSING
    trainer: Any = MISSING
    optimizer: Any = MISSING
    scheduler: Any = MISSING


def register_configs():
    cs = ConfigStore.instance()

    cs.store(name="default", node=Config)

    for module_info in pkgutil.walk_packages(__path__):
        name = module_info.name
        module_finder = module_info.module_finder

        module = module_finder.find_module(name).load_module(name)
        module._register_configs()
