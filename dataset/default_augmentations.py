
import albumentations as A


# Аугментации для классификатора ракурса:
clrk_train_augs_256 = A.Compose([
    A.Resize(256, 256),
    A.GaussNoise(var_limit=0.0392, p=0.3),
    A.RandomBrightnessContrast(brightness_limit=(-1, 0.2), brightness_by_max=False, p=0.3),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=(0.512), std=(0.272), max_pixel_value=1)
])

clrk_train_augs_128 = A.Compose([
    A.Resize(128, 128),
    A.GaussNoise(var_limit=0.02, p=0.3),
    A.RandomBrightnessContrast(brightness_limit=(-1, 0.5), brightness_by_max=False, p=0.3),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=(0.512,), std=(0.272,), max_pixel_value=1)
])


# Аугментации для классификатора марок и моделей:
clTL_train_augs_128 = A.Compose([
    A.Resize(128, 128),
    A.GaussNoise(var_limit=0.01, p=0.3),
    A.RandomBrightnessContrast(brightness_limit=(-0.8, 0.3), brightness_by_max=False, p=0.3),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=0.512, std=0.272, max_pixel_value=1)
])

clTL_test_augs_128 = A.Compose([
    A.Resize(128, 128),
    A.Normalize(mean=(0.512,), std=(0.272,), max_pixel_value=1)
])


# Аугментации для Triplet Loss:
TL_train_augs_128 = A.Compose([
    A.Resize(128, 128),
    A.GaussNoise(var_limit=0.01, p=0.42),
    A.RandomBrightnessContrast(brightness_limit=(-0.8, 0.3), brightness_by_max=False, p=0.42),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=(0.512,), std=(0.272,), max_pixel_value=1)
])

TL_no_aug_transform_128 = A.Compose([
    A.Resize(128, 128),
    A.Normalize(mean=(0.512,), std=(0.272,), max_pixel_value=1)
])
