import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from hydra.core.config_store import ConfigStore
from hydra_zen import builds
from torch.utils.data import DataLoader

from groovis.configs import full_builds, partial_builds
from groovis.data.datamodule import ImagenetModule
from groovis.data.dataset import Imagenet, Imagenette

RandomResizedCropConfig = full_builds(
    A.RandomResizedCrop,
    height=224,
    width=224,
    scale=(0.08, 1),
    ratio=(0.75, 1.3333333333333333),
    always_apply=True,
)

HorizontalFlipConfig = full_builds(
    A.HorizontalFlip,
    p=0.5,
)

ColorJitterConfig = full_builds(
    A.ColorJitter,
    brightness=0.8,
    contrast=0.8,
    saturation=0.8,
    hue=0.2,
    p=0.8,
)

ToGrayConfig = full_builds(
    A.ToGray,
    p=0.2,
)

GaussianBlurConfig = full_builds(
    A.GaussianBlur,
    blur_limit=(21, 23),
    sigma_limit=(0.1, 2),
    always_apply=True,
)

ToTensorV2Config = full_builds(
    ToTensorV2,
    always_apply=True,
)


RelaxedAugmentationConfig = builds(
    A.Compose,
    transforms=builds(
        list,
        [
            RandomResizedCropConfig,
            HorizontalFlipConfig,
            GaussianBlurConfig,
            ToTensorV2Config,
        ],
    ),
)

AugmentationConfig = builds(
    A.Compose,
    transforms=builds(
        list,
        [
            RandomResizedCropConfig,
            HorizontalFlipConfig,
            ColorJitterConfig,
            ToGrayConfig,
            GaussianBlurConfig,
            ToTensorV2Config,
        ],
    ),
)


ImagenetConfig = partial_builds(
    Imagenet,
    transforms=None,
)

ImagenetteConfig = partial_builds(
    Imagenette,
    transforms=None,
)


DataLoaderConfig = partial_builds(
    DataLoader,
    shuffle=True,
    drop_last=True,
)


ImagenetModuleConfig = full_builds(
    ImagenetModule,
)


def _register_configs():
    cs = ConfigStore.instance()

    cs.store(group="datamodule", name="imagenet", node=ImagenetModuleConfig)

    cs.store(group="datamodule/dataset", name="imagenet", node=ImagenetConfig)
    cs.store(group="datamodule/dataset", name="imagenette", node=ImagenetteConfig)

    cs.store(
        group="datamodule/dataset/transforms",
        name="default",
        node=AugmentationConfig,
    )
    cs.store(
        group="datamodule/dataset/transforms",
        name="relaxed",
        node=RelaxedAugmentationConfig,
    )

    cs.store(
        group="datamodule/dataloader",
        name="small",
        node=DataLoaderConfig(batch_size=8),
    )
    cs.store(
        group="datamodule/dataloader",
        name="base",
        node=DataLoaderConfig(batch_size=32),
    )
    cs.store(
        group="datamodule/dataloader",
        name="large",
        node=DataLoaderConfig(batch_size=128),
    )
