import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

SIMCLR_AUG_RELAXED = A.Compose(
    [
        A.RandomResizedCrop(
            height=224,
            width=224,
            scale=(0.08, 1),
            ratio=(0.75, 1.3333333333333333),
            always_apply=True,
        ),
        A.HorizontalFlip(p=0.5),
        A.GaussianBlur(blur_limit=(21, 23), sigma_limit=(0.1, 2), always_apply=True),
        ToTensorV2(always_apply=True),
    ]
)


SIMCLR_AUG = A.Compose(
    [
        A.RandomResizedCrop(
            height=224,
            width=224,
            scale=(0.08, 1),
            ratio=(0.75, 1.3333333333333333),
            always_apply=True,
        ),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(
            brightness=0.8,
            contrast=0.8,
            saturation=0.8,
            hue=0.2,
            p=0.8,
        ),
        A.ToGray(p=0.2),
        A.GaussianBlur(blur_limit=(21, 23), sigma_limit=(0.1, 2), always_apply=True),
        ToTensorV2(always_apply=True),
    ]
)
