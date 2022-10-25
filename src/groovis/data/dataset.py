import os
from pathlib import Path
from typing import Literal, Union

import albumentations as A
import datasets as D
import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset

from groovis.utils import image_path_to_array

from .augmentation import SIMCLR_AUG

Splits = Literal["train", "validation"]

IMG_EXTENSIONS = [".webp", ".jpg", ".jpeg", ".png"]


class Animals(Dataset):
    def __init__(self, root: str, transforms: A.Compose = SIMCLR_AUG):
        self.paths = [
            path for path in Path(root).iterdir() if path.suffix in IMG_EXTENSIONS
        ]

        self.transforms = transforms

    def __getitem__(self, index) -> list[torch.Tensor]:
        image = image_path_to_array(self.paths[index])

        return [self.transforms(image=image)["image"] / 255.0 for _ in range(2)]

    def __len__(self):
        return len(self.paths)


class BaseImagenet(Dataset):
    dataset: Union[
        D.DatasetDict,
        D.Dataset,
        D.IterableDatasetDict,
        D.IterableDataset,
    ]

    def __init__(
        self,
        transforms: A.Compose = SIMCLR_AUG,
        split: Splits = "train",
    ):
        self.transforms = transforms
        self.set_dataset(split=split)

    def __getitem__(self, index) -> list[torch.Tensor]:
        image: Image.Image = self.dataset[index]["image"]
        image = image.convert("RGB")
        image = np.array(image)

        return [self.transforms(image=image)["image"] / 255.0 for _ in range(2)]

    def __len__(self):
        return self.dataset.num_rows

    def set_dataset(self, split: Splits):
        raise NotImplementedError


class Imagenette(BaseImagenet):
    def __init__(self, transforms: A.Compose = SIMCLR_AUG, split: Splits = "train"):
        super().__init__(transforms, split)

    def set_dataset(self, split: Splits):
        self.dataset = load_dataset(
            path="frgfm/imagenette",
            name="320px",
            split=split,
        )


class Imagenet(BaseImagenet):
    def __init__(self, transforms: A.Compose = SIMCLR_AUG, split: Splits = "train"):
        super().__init__(transforms, split)

    def set_dataset(self, split: Splits):
        if "HF_AUTH_TOKEN" not in os.environ:
            raise KeyError("'HF_AUTH_TOKEN' must be set")

        self.dataset = load_dataset(
            path="imagenet-1k",
            split=split,
            use_auth_token=os.environ["HF_AUTH_TOKEN"],
        )
