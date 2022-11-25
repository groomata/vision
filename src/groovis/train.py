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

    logger: WandbLogger = trainer.logger

    logger.experiment.config.update(OmegaConf.to_container(config))

    logger.watch(
        model=architecture,
        log="all",
        log_freq=config.trainer.log_every_n_steps,
    )

    trainer.fit(
        model=model,
        datamodule=datamodule,
    )
