from typing import Optional

from hydra_zen.typing import Partial
from pytorch_lightning import LightningDataModule, Trainer
from pytorch_lightning.accelerators import CPUAccelerator
from torch.utils.data import DataLoader

from groovis.data.dataset import BaseImagenet


class ImagenetModule(LightningDataModule):
    trainer: Trainer

    def __init__(
        self,
        dataloader: Partial[DataLoader],
        dataset: Partial[BaseImagenet],
    ) -> None:
        super().__init__()

        self.dataset = dataset
        self.dataloader = dataloader

    def setup(self, stage: Optional[str] = None) -> None:
        self.dataset(split="train")
        self.dataset(split="validation")

    def train_dataloader(self) -> DataLoader:
        return self.dataloader(
            dataset=self.dataset(
                split="train",
            ),
            num_workers=None if self.on_cpu else self.trainer.num_devices * 4,
            prefetch_factor=None if self.on_cpu else 2,
            persistent_workers=None if self.on_cpu else True,
            pin_memory=None if self.on_cpu else True,
        )

    def val_dataloader(self) -> DataLoader:
        return self.dataloader(
            dataset=self.dataset(
                split="validation",
            ),
            num_workers=None if self.on_cpu else self.trainer.num_devices * 4,
            prefetch_factor=None if self.on_cpu else 2,
            persistent_workers=None if self.on_cpu else True,
            pin_memory=None if self.on_cpu else True,
        )

    @property
    def on_cpu(self) -> bool:
        return isinstance(self.trainer.accelerator, CPUAccelerator)
