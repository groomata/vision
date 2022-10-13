import warnings

from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

from pathlib import Path

import torch
import torch_optimizer as optim
from loguru import logger
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import MeanMetric, MinMetric

import wandb
from groovis import Vision
from groovis.data import SIMCLR_AUG_RELAXED, Imagenette, Splits
from groovis.loss import SimCLRLoss
from groovis.schema import load_config
from groovis.vision import Architecture

config = load_config("config.yaml")

logger.add("logs/train_{time}.log")
logger.info(f"Configuration: {config}")

RUN_NAME = "optimizer-test-1"

Path(f"build/{RUN_NAME}").mkdir(parents=True, exist_ok=True)

wandb.init(
    project="groovis",
    group="first try",
    name=RUN_NAME,
    mode="online",
)

wandb.config.update(OmegaConf.to_container(config))

splits: list[Splits] = ["train", "validation"]

dataloader: dict[Splits, DataLoader] = {
    split: DataLoader(
        dataset=Imagenette(
            transforms=SIMCLR_AUG_RELAXED,
            split=split,
        ),
        batch_size=config.batch_size,
        drop_last=True,
        shuffle=True,
    )
    for split in splits
}

epoch_steps = len(dataloader["train"])
warmup_steps = config.warmup_epochs * epoch_steps
total_steps = config.epochs * epoch_steps

metric: dict[Splits, MeanMetric] = {split: MeanMetric() for split in splits}
best_validation_loss = MinMetric()
best_validation_loss.update(1e9)


loss_fn = SimCLRLoss(temperature=config.temperature)

architecture = Architecture(
    patch_size=config.patch_size,
    channels=config.channels,
    embed_dim=config.embed_dim,
)

vision = Vision(architecture=architecture)

wandb.watch(
    models=vision,
    log="all",
    log_freq=config.log_interval,
)

optimizer = optim.LARS(
    params=vision.parameters(),
    lr=config.base_lr,
)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer=optimizer,
    max_lr=config.base_lr,
    total_steps=total_steps,
    pct_start=config.warmup_epochs / config.epochs,
    anneal_strategy="linear",
    div_factor=config.base_lr / config.warmup_lr,
    final_div_factor=1e6,
)

images: list[torch.Tensor]


global_step = 0

patience = 0

for epoch in range(config.epochs):
    for step, images in enumerate(dataloader["train"]):
        images_1, images_2 = images

        representations_1 = vision(images_1)
        representations_2 = vision(images_2)

        loss = loss_fn(representations_1, representations_2)
        loss.backward()

        nn.utils.clip_grad_norm_(
            parameters=vision.parameters(),
            max_norm=config.clip_grad,
            error_if_nonfinite=True,
        )

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        metric["train"].update(loss)

        lr = optimizer.param_groups[0]["lr"]
        if not (global_step + 1) % config.log_interval:
            wandb.log(
                data={
                    "train": {
                        "lr": lr,
                        "loss": metric["train"].compute(),
                    },
                },
                step=global_step,
                commit=False if step == len(dataloader["train"]) - 1 else True,
            )
            metric["train"].reset()

        logger.info(
            f"Train: "
            f"[{epoch}/{config.epochs}][{step}/{len(dataloader['train'])}]\t"
            f"lr {lr:.4f}\t"
            f"loss {loss:.4f}\t"
            # f"grad_norm {grad_norm:.4f}"
        )

        global_step += 1

    with torch.no_grad():
        for step, images in enumerate(dataloader["validation"]):
            images_1, images_2 = images

            representations_1 = vision(images_1)
            representations_2 = vision(images_2)

            loss = loss_fn(representations_1, representations_2)

            metric["validation"].update(loss)

            logger.info(
                f"Validation: "
                f"[{epoch}/{config.epochs}][{step}/{len(dataloader['validation'])}]\t"
                f"loss {loss:.4f}\t"
            )

        validation_loss = metric["validation"].compute()

        wandb.log(
            data={
                "validation": {
                    "loss": validation_loss,
                },
            },
            step=global_step - 1,
            commit=True,
        )
        metric["validation"].reset()

        torch.save(
            {
                "vision": vision.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            f"build/{RUN_NAME}/{epoch:02d}-{validation_loss:.2f}.pth",
        )

        if validation_loss < best_validation_loss.compute():
            patience = 0
        else:
            patience += 1

        if patience == config.patience:
            break

        best_validation_loss.update(validation_loss)
