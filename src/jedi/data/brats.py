from __future__ import annotations

import random
from pathlib import Path

from monai.data import DataLoader, Dataset

from jedi.data.transforms import build_pair_transforms, build_train_transforms


class BraTSContrastDataset(Dataset):
    modalities = ("t1n", "t1c", "t2w", "t2f")
    _modality_to_idx = {m: i for i, m in enumerate(modalities)}

    def __init__(
        self,
        data_dir: str,
        mode: str,
        fixed_mapping: tuple[str, str],
        spatial_size: tuple[int, int, int] = (128, 160, 192),
        a_min: float = 0.0,
        a_max: float = 3000.0,
        transform=None,
    ):
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.fixed_mapping = fixed_mapping
        self.items = sorted(path for path in self.data_dir.iterdir() if path.is_dir())
        if transform is not None:
            self.transform = transform
        elif self.mode == "train":
            self.transform = build_train_transforms(spatial_size=spatial_size, a_min=a_min, a_max=a_max)
        else:
            self.transform = build_pair_transforms(spatial_size=spatial_size, a_min=a_min, a_max=a_max)

    def __len__(self) -> int:
        return len(self.items)

    def _sample_mapping(self) -> tuple[str, str]:
        if self.mode != "train":
            return self.fixed_mapping
        return tuple(random.sample(self.modalities, k=2))

    def __getitem__(self, index: int):
        case_dir = self.items[index]
        case_id = case_dir.name
        src_modality, tgt_modality = self._sample_mapping()
        sample = {
            "src": str(case_dir / f"{case_id}-{src_modality}.nii.gz"),
            "tgt": str(case_dir / f"{case_id}-{tgt_modality}.nii.gz"),
            "case_id": case_id,
            "src_modality": src_modality,
            "tgt_modality": tgt_modality,
            "src_modality_idx": self._modality_to_idx[src_modality],
            "tgt_modality_idx": self._modality_to_idx[tgt_modality],
        }
        return self.transform(sample)



def build_dataloader(
    data_dir: str,
    mode: str,
    fixed_mapping: tuple[str, str],
    batch_size: int,
    num_workers: int,
    spatial_size: tuple[int, int, int],
):
    dataset = BraTSContrastDataset(
        data_dir=data_dir,
        mode=mode,
        fixed_mapping=fixed_mapping,
        spatial_size=spatial_size,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=mode == "train", num_workers=num_workers)
