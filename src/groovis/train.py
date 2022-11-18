from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from pytorch_lightning.loggers.wandb import WandbLogger
from timm import create_model

from groovis.data import ImagenetModule, Imagenette
from groovis.loss import SimCLRLoss
from groovis.models.module import VAL_LOSS, Vision
from groovis.schema import Config


def train(config: Config):
    RUN_NAME = config.run_name

    datamodule = ImagenetModule(config=config, dataset=Imagenette)

    logger = WandbLogger(
        project="groovis",
        group="first try",
        name=RUN_NAME,
        offline=config.offline,
        log_model=True,
    )

    loss_fn = SimCLRLoss(temperature=config.temperature)

    # architecture = Architecture(
    #     patch_size=config.patch_size,
    #     channels=config.channels,
    #     embed_dim=config.embed_dim,
    # )
    architecture = create_model(
        model_name="vit_small_patch16_224",
        num_classes=0,
    )

    logger.watch(
        model=architecture,
        log="all",
        log_freq=config.log_interval,
    )

    vision = Vision(
        architecture=architecture,
        loss_fn=loss_fn,
        config=config,
    )

    callbacks: list[Callback] = [
        ModelCheckpoint(
            dirpath=f"build/{RUN_NAME}",
            filename="epoch={epoch:02d}-val_loss={" + VAL_LOSS + ":.2f}",
            save_last=True,
            monitor=VAL_LOSS,
            save_top_k=config.save_top_k,
            mode="min",
            save_weights_only=False,
            auto_insert_metric_name=False,
        ),
        LearningRateMonitor(
            logging_interval="step",
        ),
        EarlyStopping(
            monitor=VAL_LOSS,
            patience=config.patience,
            mode="min",
            strict=True,
            check_finite=True,
        ),
        RichModelSummary(max_depth=3),
        RichProgressBar(
            theme=RichProgressBarTheme(
                description="green_yellow",
                progress_bar="green1",
                progress_bar_finished="green1",
                progress_bar_pulse="#6206E0",
                batch_progress="green_yellow",
                time="grey82",
                processing_speed="grey82",
                metrics="grey82",
            )
        ),
    ]

    # profiler = PyTorchProfiler(
    #     dirpath="logs/",
    #     filename=f"profile-{RUN_NAME}",
    #     export_to_chrome=True,
    # )

    trainer = Trainer(
        logger=logger,
        callbacks=callbacks,
        # profiler=profiler,
        max_epochs=config.epochs,
        gradient_clip_algorithm="norm",
        gradient_clip_val=config.clip_grad,
        log_every_n_steps=config.log_interval,
        track_grad_norm=2,
        precision=16,
        accelerator="auto",
        devices="auto",
    )

    trainer.fit(
        model=vision,
        datamodule=datamodule,
    )
