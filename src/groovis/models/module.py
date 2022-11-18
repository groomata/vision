import torch
import torch_optimizer as optim
from pytorch_lightning import LightningModule
from torch import nn
from torchmetrics import MeanMetric

from groovis.schema import Config

TRAIN_LOSS = "train/loss"
VAL_LOSS = "val/loss"


class Vision(LightningModule):
    hparams: Config

    def __init__(
        self,
        architecture: nn.Module,
        loss_fn: nn.Module,
        config: Config,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(config)

        self.architecture = architecture
        self.loss_fn = loss_fn

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.architecture(images)

    def training_step(self, batch: list[torch.Tensor], _batch_idx: int) -> torch.Tensor:
        images_1, images_2 = batch

        representations_1 = self(images_1)
        representations_2 = self(images_2)

        loss = self.loss_fn(representations_1, representations_2)

        self.train_loss(loss)

        self.log(
            name=TRAIN_LOSS,
            value=self.train_loss,
            on_step=True,
            on_epoch=False,
            logger=True,
        )

        return loss

    def validation_step(self, batch: list[torch.Tensor], _batch_idx: int):
        images_1, images_2 = batch

        representations_1 = self(images_1)
        representations_2 = self(images_2)

        loss = self.loss_fn(representations_1, representations_2)

        self.val_loss(loss)

        self.log(
            name=VAL_LOSS,
            value=self.val_loss,
            on_step=False,
            on_epoch=True,
            logger=True,
        )

    def configure_optimizers(self):

        optimizer = optim.LARS(
            params=self.parameters(),
            lr=self.hparams.base_lr,
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=self.hparams.base_lr,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=self.hparams.warmup_epochs / self.hparams.epochs,
            anneal_strategy="linear",
            div_factor=self.hparams.base_lr / self.hparams.warmup_lr,
            final_div_factor=1e6,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def optimizer_zero_grad(
        self,
        _epoch: int,
        _batch_idx: int,
        optimizer: torch.optim.Optimizer,
        _optimizer_idx: int,
    ):
        optimizer.zero_grad(set_to_none=True)
