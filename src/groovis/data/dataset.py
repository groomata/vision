from pathlib import Path
from typing import Literal

import albumentations as A
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


class Imagenette(Dataset):
    def __init__(
        self,
        transforms: A.Compose = SIMCLR_AUG,
        split: Splits = "train",
    ):
        self.dataset = load_dataset(
            path="frgfm/imagenette",
            name="320px",
            split=split,
        )

        self.transforms = transforms

    def __getitem__(self, index) -> list[torch.Tensor]:
        image: Image.Image = self.dataset[index]["image"]
        image = image.convert("RGB")
        image = np.array(image)

        return [self.transforms(image=image)["image"] / 255.0 for _ in range(2)]

    def __len__(self):
        return self.dataset.num_rows
