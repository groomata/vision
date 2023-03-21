import numpy as np
import torch
from einops import rearrange
from PIL import Image

IMAGE_SIZE = 224


def image_path_to_array(path: str) -> np.ndarray:
    image = Image.open(path)
    image = np.array(image)
    return image


def image_path_to_tensor(path: str) -> np.ndarray:
    image = Image.open(path).resize((IMAGE_SIZE, IMAGE_SIZE))
    image = np.array(image) / 255.0
    image = torch.tensor(image, dtype=torch.float)
    image = rearrange(image, "h w c -> 1 c h w")
    return image
