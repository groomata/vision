from dataclasses import dataclass

from omegaconf import OmegaConf


@dataclass
class Cfg:
    patch_size: int = 16
    channels: int = 3
    embed_dim: int = 128
    base_lr: float = 3e-4
    warmup_lr: float = 3e-7
    warmup_epochs: int = 5
    batch_size: int = 32
    epochs: int = 30
    patience: int = 5
    clip_grad: float = 3.0
    temperature: float = 0.1
    log_interval: int = 50


def load_config(path: str) -> Cfg:
    schema = OmegaConf.structured(Cfg())

    config: Cfg = OmegaConf.merge(
        schema,
        OmegaConf.load(path),
    )

    return config
