import warnings

from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

from pathlib import Path

import hydra
import torch
from hydra_zen import instantiate
from hydra_zen.typing import Partial
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

import wandb
from groovis.configs import Config, register_configs
from groovis.module import Vision
from groovis.utils import image_path_to_tensor


@hydra.main(
    config_name="default",
    version_base="1.2",
)
def main(config: Config):
    loss_fn: nn.Module = instantiate(config.loss)
    architecture: nn.Module = instantiate(config.architecture)
    optimizer: Partial[Optimizer] = instantiate(config.optimizer)
    scheduler: Partial[_LRScheduler] = instantiate(config.scheduler)

    run = wandb.init()  # type: ignore
    artifact = run.use_artifact(
        "groomata-vision/groovis/model-1rzrphsi:v2", type="model"
    )
    artifact_path = artifact.download()
    checkpoint_path = Path(artifact_path) / "model.ckpt"

    model = Vision.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        architecture=architecture,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    image_tiger_1 = image_path_to_tensor("data/train/tiger_1.webp")
    image_tiger_2 = image_path_to_tensor("data/train/tiger_2.webp")
    image_dog = image_path_to_tensor("data/train/dog.webp")

    tiger_1: torch.Tensor = model(image_tiger_1)
    tiger_2: torch.Tensor = model(image_tiger_2)
    dog: torch.Tensor = model(image_dog)

    tiger_1.div_(tiger_1.norm())
    tiger_2.div_(tiger_2.norm())
    dog.div_(dog.norm())

    sim_tiger_tiger = (tiger_2 * tiger_1).sum()
    sim_tiger_dog_1 = (tiger_1 * dog).sum()
    sim_tiger_dog_2 = (tiger_2 * dog).sum()

    quality = -(sim_tiger_dog_1 + sim_tiger_dog_2) / 2 + sim_tiger_tiger

    print(f"{sim_tiger_tiger=}")
    print(f"{sim_tiger_dog_1=}")
    print(f"{sim_tiger_dog_2=}")

    print(f"{quality=}")


if __name__ == "__main__":
    register_configs()
    main()
