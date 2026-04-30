from __future__ import annotations

import numpy as np
import torch
from monai.data.meta_obj import get_track_meta
from monai.transforms import (
    CenterSpatialCropd,
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    MapTransform,
    Orientationd,
    Rand3DElasticd,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandRotated,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSimulateLowResolution,
    RandZoomd,
    RandomizableTransform,
    ScaleIntensityRanged,
    SpatialPadd,
)
from monai.transforms.utils import convert_to_tensor


def normalize_to_unit_range(volume: np.ndarray, a_min: float, a_max: float) -> np.ndarray:
    clipped = np.clip(volume, a_min, a_max)
    scaled = (clipped - a_min) / (a_max - a_min)
    return scaled * 2.0 - 1.0


def pad_or_crop_volume(volume: np.ndarray, spatial_size: tuple[int, int, int]) -> np.ndarray:
    output = np.zeros(spatial_size, dtype=volume.dtype)
    src_slices = []
    dst_slices = []
    for current, target in zip(volume.shape, spatial_size):
        copy_size = min(current, target)
        src_start = max((current - copy_size) // 2, 0)
        dst_start = max((target - copy_size) // 2, 0)
        src_slices.append(slice(src_start, src_start + copy_size))
        dst_slices.append(slice(dst_start, dst_start + copy_size))
    output[tuple(dst_slices)] = volume[tuple(src_slices)]
    return output


class RandGammaCorrectiond(MapTransform, RandomizableTransform):
    def __init__(self, keys, gamma_range=(0.7, 1.5), prob=0.3, allow_missing_keys=False):
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.gamma_range = gamma_range
        self.gamma = 1.0

    def randomize(self, data=None):
        super().randomize(data)
        if self._do_transform:
            self.gamma = self.R.uniform(*self.gamma_range)

    def __call__(self, data):
        d = dict(data)
        self.randomize(None)
        if not self._do_transform:
            for key in self.key_iterator(d):
                d[key] = convert_to_tensor(d[key], track_meta=get_track_meta(), dtype=torch.float32)
            return d
        for key in self.key_iterator(d):
            img = d[key]
            shifted = (img + 1.0) / 2.0
            if torch.is_tensor(img):
                shifted = shifted.clamp(1e-8)
            else:
                shifted = np.clip(shifted, 1e-8, None)
            shifted = shifted ** self.gamma
            d[key] = shifted * 2.0 - 1.0
        return d


class _FixedRandSimulateLowResolutiond(MapTransform, RandomizableTransform):
    """Like MONAI's RandSimulateLowResolutiond but samples zoom once for all keys.

    MONAI 1.5.2's RandSimulateLowResolutiond samples a new zoom_factor per key
    in its per-key loop, which breaks paired data (src/tgt get different factors).
    This wrapper samples the zoom once and reuses it.
    """

    def __init__(self, keys, prob=0.1, zoom_range=(0.5, 1.0), allow_missing_keys=False, **kwargs):
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.sim = RandSimulateLowResolution(prob=1.0, zoom_range=zoom_range, **kwargs)

    def set_random_state(self, seed=None, state=None):
        super().set_random_state(seed, state)
        self.sim.set_random_state(seed, state)
        return self

    def randomize(self, data=None):
        super().randomize(data)
        if self._do_transform:
            self.sim.randomize(None)

    def __call__(self, data):
        d = dict(data)
        self.randomize(None)
        if self._do_transform:
            for key in self.key_iterator(d):
                d[key] = self.sim(d[key], randomize=False)
        else:
            for key in self.key_iterator(d):
                d[key] = convert_to_tensor(d[key], track_meta=get_track_meta(), dtype=torch.float32)
        return d


def build_base_transforms(spatial_size: tuple[int, int, int]):
    return [
        LoadImaged(keys=["src", "tgt"]),
        EnsureChannelFirstd(keys=["src", "tgt"]),
        Orientationd(keys=["src", "tgt"], axcodes="RAS"),
        SpatialPadd(keys=["src", "tgt"], spatial_size=spatial_size),
        CenterSpatialCropd(keys=["src", "tgt"], roi_size=spatial_size),
    ]


def build_nnunet_augmentations():
    return [
        RandFlipd(keys=["src", "tgt"], spatial_axis=[0], prob=0.5),
        RandFlipd(keys=["src", "tgt"], spatial_axis=[1], prob=0.5),
        RandFlipd(keys=["src", "tgt"], spatial_axis=[2], prob=0.5),
        RandRotated(
            keys=["src", "tgt"],
            range_x=0.5236, range_y=0.5236, range_z=0.5236,
            prob=0.2,
            mode="bilinear",
            keep_size=True,
        ),
        RandZoomd(
            keys=["src", "tgt"],
            min_zoom=0.7, max_zoom=1.4,
            prob=0.2,
            mode="trilinear",
            keep_size=True,
        ),
        Rand3DElasticd(
            keys=["src", "tgt"],
            sigma_range=(9, 13),
            magnitude_range=(50, 100),
            prob=0.2,
            mode="bilinear",
        ),
    ]


def build_intensity_augmentations():
    return [
        RandGammaCorrectiond(keys=["src", "tgt"], gamma_range=(0.7, 1.5), prob=0.3),
        RandGaussianNoised(keys=["src", "tgt"], mean=0.0, std=0.1, prob=0.15),
        RandGaussianSmoothd(
            keys=["src", "tgt"],
            sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0), sigma_z=(0.5, 1.0),
            prob=0.2,
        ),
        RandScaleIntensityd(keys=["src", "tgt"], factors=(0.75, 1.25), prob=0.15),
        RandShiftIntensityd(keys=["src", "tgt"], offsets=(-0.1, 0.1), prob=0.15),
        _FixedRandSimulateLowResolutiond(keys=["src", "tgt"], zoom_range=(0.5, 1.0), prob=0.25),
    ]


def build_pair_transforms(spatial_size: tuple[int, int, int], a_min: float, a_max: float):
    return Compose(
        build_base_transforms(spatial_size)
        + [
            ScaleIntensityRanged(
                keys=["src", "tgt"],
                a_min=a_min,
                a_max=a_max,
                b_min=-1.0,
                b_max=1.0,
                clip=True,
            ),
            EnsureTyped(keys=["src", "tgt"]),
        ]
    )


def build_train_transforms(spatial_size: tuple[int, int, int], a_min: float, a_max: float):
    return Compose(
        build_base_transforms(spatial_size)
        + build_nnunet_augmentations()
        + [
            ScaleIntensityRanged(
                keys=["src", "tgt"],
                a_min=a_min,
                a_max=a_max,
                b_min=-1.0,
                b_max=1.0,
                clip=True,
            ),
        ]
        + build_intensity_augmentations()
        + [
            EnsureTyped(keys=["src", "tgt"]),
        ]
    )
