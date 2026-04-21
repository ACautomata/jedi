import tempfile
import unittest
from pathlib import Path

from jedi.data.brats import BraTSContrastDataset


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


if __name__ == "__main__":
    unittest.main()
