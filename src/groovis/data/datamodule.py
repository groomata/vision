from typing import Optional, Type

from pytorch_lightning import LightningDataModule, Trainer
from pytorch_lightning.accelerators import CPUAccelerator
from torch.utils.data import DataLoader

from groovis.data.augmentation import SIMCLR_AUG_RELAXED
from groovis.data.dataset import BaseImagenet
from groovis.schema import Config


class ImagenetModule(LightningDataModule):
    hparams: Config
    trainer: Trainer

    def __init__(self, config: Config, dataset: Type[BaseImagenet]) -> None:
        super().__init__()

        self.save_hyperparameters(config)

        self.dataset = dataset

    def setup(self, stage: Optional[str] = None) -> None:
        self.dataset(split="train")
        self.dataset(split="validation")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.dataset(
                transforms=SIMCLR_AUG_RELAXED,
                split="train",
            ),
            batch_size=self.hparams.batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=0 if self.on_cpu else self.trainer.num_devices * 4,
            prefetch_factor=2,
            persistent_workers=True,
            pin_memory=not self.on_cpu,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.dataset(
                transforms=SIMCLR_AUG_RELAXED,
                split="validation",
            ),
            batch_size=self.hparams.batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=0 if self.on_cpu else self.trainer.num_devices * 4,
            prefetch_factor=2,
            persistent_workers=True,
            pin_memory=not self.on_cpu,
        )

    @property
    def on_cpu(self) -> bool:
        return isinstance(self.trainer.accelerator, CPUAccelerator)
