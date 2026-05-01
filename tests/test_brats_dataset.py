import tempfile
import unittest
from pathlib import Path

import numpy as np

from jedi.data.brats import BraTSContrastDataset, build_dataloader
from jedi.data.transforms import (
    build_pair_random_transforms,
    build_single_volume_transforms,
)
from monai.transforms import Compose


def _create_nifti(path, shape=(4, 4, 4)):
    """Create a minimal NIfTI file for testing."""
    import nibabel as nib
    img = nib.Nifti1Image(np.zeros(shape, dtype=np.float32), np.eye(4))
    nib.save(img, str(path))


class TestBraTSDataset(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        case_dir = Path(self.tmpdir.name) / "BraTS-GLI-00001-000"
        case_dir.mkdir()
        for suffix in ["t1n", "t1c", "t2w", "t2f"]:
            (case_dir / f"BraTS-GLI-00001-000-{suffix}.nii.gz").touch()

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_eval_mode_uses_fixed_mapping(self):
        dataset = BraTSContrastDataset(
            data_dir=self.tmpdir.name,
            mode="val",
            fixed_mapping=("t1n", "t2w"),
            transform=lambda sample: sample,
        )
        sample = dataset[0]
        self.assertEqual(sample["src_modality"], "t1n")
        self.assertEqual(sample["tgt_modality"], "t2w")

    def test_train_mode_never_returns_identical_modalities(self):
        dataset = BraTSContrastDataset(
            data_dir=self.tmpdir.name,
            mode="train",
            fixed_mapping=("t1n", "t2w"),
            transform=lambda sample: sample,
        )
        for _ in range(20):
            sample = dataset[0]
            self.assertNotEqual(sample["src_modality"], sample["tgt_modality"])

    def test_dataset_applies_transform(self):
        dataset = BraTSContrastDataset(
            data_dir=self.tmpdir.name,
            mode="val",
            fixed_mapping=("t1n", "t2w"),
            transform=lambda sample: {**sample, "transformed": True},
        )
        sample = dataset[0]
        self.assertTrue(sample["transformed"])


class TestCachedBraTSDataset(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.cache_dir = tempfile.TemporaryDirectory()
        case_dir = Path(self.tmpdir.name) / "BraTS-GLI-00001-000"
        case_dir.mkdir()
        for suffix in ["t1n", "t1c", "t2w", "t2f"]:
            _create_nifti(case_dir / f"BraTS-GLI-00001-000-{suffix}.nii.gz")

    def tearDown(self):
        self.tmpdir.cleanup()
        self.cache_dir.cleanup()

    def test_cached_val_uses_fixed_mapping(self):
        dataset = BraTSContrastDataset(
            data_dir=self.tmpdir.name,
            mode="val",
            fixed_mapping=("t1n", "t2w"),
            spatial_size=(4, 4, 4),
            cache_dir=self.cache_dir.name,
        )
        sample = dataset[0]
        self.assertEqual(sample["src_modality"], "t1n")
        self.assertEqual(sample["tgt_modality"], "t2w")

    def test_cached_train_different_modalities(self):
        dataset = BraTSContrastDataset(
            data_dir=self.tmpdir.name,
            mode="train",
            fixed_mapping=("t1n", "t2w"),
            spatial_size=(4, 4, 4),
            cache_dir=self.cache_dir.name,
        )
        for _ in range(20):
            sample = dataset[0]
            self.assertNotEqual(sample["src_modality"], sample["tgt_modality"])

    def test_cached_keys_match_uncached(self):
        expected_keys = {
            "src", "tgt", "case_id", "src_modality", "tgt_modality",
            "src_modality_idx", "tgt_modality_idx",
        }
        dataset = BraTSContrastDataset(
            data_dir=self.tmpdir.name,
            mode="val",
            fixed_mapping=("t1n", "t2w"),
            spatial_size=(4, 4, 4),
            cache_dir=self.cache_dir.name,
        )
        sample = dataset[0]
        self.assertEqual(set(sample.keys()), expected_keys)

    def test_cached_val_output_is_tensor(self):
        import torch
        dataset = BraTSContrastDataset(
            data_dir=self.tmpdir.name,
            mode="val",
            fixed_mapping=("t1n", "t2w"),
            spatial_size=(4, 4, 4),
            cache_dir=self.cache_dir.name,
        )
        sample = dataset[0]
        self.assertIsInstance(sample["src"], torch.Tensor)
        self.assertIsInstance(sample["tgt"], torch.Tensor)

    def test_cached_train_output_is_tensor(self):
        import torch
        dataset = BraTSContrastDataset(
            data_dir=self.tmpdir.name,
            mode="train",
            fixed_mapping=("t1n", "t2w"),
            spatial_size=(4, 4, 4),
            cache_dir=self.cache_dir.name,
        )
        sample = dataset[0]
        self.assertIsInstance(sample["src"], torch.Tensor)
        self.assertIsInstance(sample["tgt"], torch.Tensor)

    def test_cache_dir_sets_cache_attribute(self):
        dataset = BraTSContrastDataset(
            data_dir=self.tmpdir.name,
            mode="val",
            fixed_mapping=("t1n", "t2w"),
            spatial_size=(4, 4, 4),
            cache_dir=self.cache_dir.name,
        )
        self.assertIsNotNone(dataset._cache)

    def test_no_cache_dir_leaves_cache_none(self):
        dataset = BraTSContrastDataset(
            data_dir=self.tmpdir.name,
            mode="val",
            fixed_mapping=("t1n", "t2w"),
            transform=lambda sample: sample,
        )
        self.assertIsNone(dataset._cache)


class TestBuildDataloader(unittest.TestCase):
    def test_build_dataloader_passes_cache_dir(self):
        tmpdir = tempfile.TemporaryDirectory()
        cache_dir = tempfile.TemporaryDirectory()
        case_dir = Path(tmpdir.name) / "case-001"
        case_dir.mkdir()
        for suffix in ["t1n", "t1c", "t2w", "t2f"]:
            _create_nifti(case_dir / f"case-001-{suffix}.nii.gz")

        loader = build_dataloader(
            data_dir=tmpdir.name,
            mode="val",
            fixed_mapping=("t1n", "t2w"),
            batch_size=1,
            num_workers=0,
            spatial_size=(4, 4, 4),
            cache_dir=cache_dir.name,
        )
        self.assertIsNotNone(loader.dataset._cache)
        tmpdir.cleanup()
        cache_dir.cleanup()

    def test_build_dataloader_without_cache_dir(self):
        tmpdir = tempfile.TemporaryDirectory()
        case_dir = Path(tmpdir.name) / "case-001"
        case_dir.mkdir()
        for suffix in ["t1n", "t1c", "t2w", "t2f"]:
            (case_dir / f"case-001-{suffix}.nii.gz").touch()

        loader = build_dataloader(
            data_dir=tmpdir.name,
            mode="val",
            fixed_mapping=("t1n", "t2w"),
            batch_size=1,
            num_workers=0,
            spatial_size=(4, 4, 4),
        )
        self.assertIsNone(loader.dataset._cache)
        tmpdir.cleanup()


class TestNewTransformFunctions(unittest.TestCase):
    def test_single_volume_transforms_returns_compose(self):
        transforms = build_single_volume_transforms((128, 160, 192), 0.0, 3000.0)
        self.assertIsInstance(transforms, Compose)

    def test_pair_random_transforms_returns_compose(self):
        transforms = build_pair_random_transforms()
        self.assertIsInstance(transforms, Compose)


if __name__ == "__main__":
    unittest.main()
