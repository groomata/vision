import numpy as np
from PIL import Image

IMAGE_SIZE = 224


def image_path_to_array(path: str) -> np.ndarray:
    image = Image.open(path)
    image = np.array(image)
    return image
