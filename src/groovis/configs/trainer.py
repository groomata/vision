from hydra.core.config_store import ConfigStore
from hydra_zen import builds
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.profilers.pytorch import PyTorchProfiler

from groovis.configs import full_builds
from groovis.module import VAL_LOSS

WandbLoggerConfig = full_builds(
    WandbLogger,
    project="groovis",
    group="Self-Supervised Learning Benchmarks",
    name="default",
    offline=False,
    log_model=True,
)

ModelCheckpointConfig = full_builds(
    ModelCheckpoint,
    dirpath="checkpoints/",
    filename="epoch={epoch:02d}-val_loss={" + VAL_LOSS + ":.2f}",
    save_last=True,
    monitor=VAL_LOSS,
    save_top_k=3,
    mode="min",
    save_weights_only=False,
    auto_insert_metric_name=False,
)

LearningRateMonitorConfig = full_builds(
    LearningRateMonitor,
    logging_interval="step",
)

EarlyStoppingConfig = full_builds(
    EarlyStopping,
    monitor=VAL_LOSS,
    patience=4,
    mode="min",
    strict=True,
    check_finite=True,
)

RichModelSummaryConfig = full_builds(
    RichModelSummary,
    max_depth=3,
)

RichProgressBarThemeConfig = full_builds(
    RichProgressBarTheme,
    description="green_yellow",
    progress_bar="green1",
    progress_bar_finished="green1",
    progress_bar_pulse="#6206E0",
    batch_progress="green_yellow",
    time="grey82",
    processing_speed="grey82",
    metrics="grey82",
)

RichProgressBarConfig = full_builds(
    RichProgressBar,
    theme=RichProgressBarThemeConfig,
)


CallbacksConfig = builds(
    list,
    [
        ModelCheckpointConfig,
        LearningRateMonitorConfig,
        EarlyStoppingConfig,
        RichModelSummaryConfig,
        RichProgressBarConfig,
    ],
)

CallbacksWithoutRichConfig = builds(
    list,
    [
        ModelCheckpointConfig,
        LearningRateMonitorConfig,
        EarlyStoppingConfig,
    ],
)


PyTorchProfilerConfig = full_builds(
    PyTorchProfiler,
    dirpath="profile/",
    filename="result",
    export_to_chrome=True,
)


TrainerConfig = full_builds(
    Trainer,
    logger=WandbLoggerConfig,
    callbacks=CallbacksConfig,
    max_epochs=10,
    gradient_clip_algorithm="norm",
    gradient_clip_val=0.5,
    log_every_n_steps=10,
    track_grad_norm=2,
    precision=16,
    accelerator="auto",
    devices="auto",
)


def _register_configs():
    cs = ConfigStore.instance()

    cs.store(group="trainer", name="auto", node=TrainerConfig)
    cs.store(
        group="trainer",
        name="cpu",
        node=TrainerConfig(
            precision=32,
            accelerator="cpu",
            devices=None,
        ),
    )
    cs.store(
        group="trainer",
        name="gpu",
        node=TrainerConfig(
            precision=16,
            accelerator="gpu",
            devices="auto",
        ),
    )
    cs.store(
        group="trainer",
        name="cpu-profile",
        node=TrainerConfig(
            profiler=PyTorchProfilerConfig,
            precision=32,
            accelerator="cpu",
            devices=None,
        ),
    )
    cs.store(
        group="trainer",
        name="gpu-profile",
        node=TrainerConfig(
            profiler=PyTorchProfilerConfig,
            precision=16,
            accelerator="gpu",
            devices="auto",
        ),
    )

    cs.store(group="trainer/logger", name="wandb", node=WandbLoggerConfig)

    cs.store(group="trainer/callbacks", name="default", node=CallbacksConfig)
    cs.store(group="trainer/callbacks", name="no_rich", node=CallbacksWithoutRichConfig)
