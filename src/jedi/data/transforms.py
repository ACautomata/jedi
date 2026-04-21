from __future__ import annotations

import numpy as np
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    SpatialPadd,
)


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


def build_pair_transforms(spatial_size: tuple[int, int, int], a_min: float, a_max: float):
    return Compose(
        [
            LoadImaged(keys=["src", "tgt"]),
            EnsureChannelFirstd(keys=["src", "tgt"]),
            Orientationd(keys=["src", "tgt"], axcodes="RAS"),
            SpatialPadd(keys=["src", "tgt"], spatial_size=spatial_size),
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
