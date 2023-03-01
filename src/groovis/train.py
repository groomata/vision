from hydra_zen import instantiate
from hydra_zen.typing import Partial
from omegaconf import OmegaConf
from pytorch_lightning import LightningDataModule, Trainer
from pytorch_lightning.loggers.wandb import WandbLogger
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from groovis.configs import Config
from groovis.module import Vision


def train(config: Config):

    exp: Config = instantiate(config)  # WARNING: Abusing our types

    datamodule: LightningDataModule = exp.datamodule

    loss_fn: nn.Module = exp.loss

    architecture: nn.Module = exp.architecture

    optimizer: Partial[Optimizer] = exp.optimizer

    scheduler: Partial[_LRScheduler] = exp.scheduler

    model = Vision(
        architecture=architecture,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    trainer: Trainer = exp.trainer

    if trainer.is_global_zero:
        if isinstance(trainer.logger, WandbLogger):
            trainer.logger.experiment.config.update(OmegaConf.to_container(config))

    trainer.fit(
        model=model,
        datamodule=datamodule,
    )
