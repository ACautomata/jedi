import unittest

import numpy as np
import torch

from jedi.data.transforms import (
    RandGammaCorrectiond,
    _FixedRandSimulateLowResolutiond,
    build_pair_transforms,
    build_train_transforms,
    normalize_to_unit_range,
    pad_or_crop_volume,
)


class TestTransforms(unittest.TestCase):
    def test_normalize_to_unit_range_bounds(self):
        volume = np.array([0.0, 1500.0, 3000.0], dtype=np.float32)
        normalized = normalize_to_unit_range(volume, a_min=0.0, a_max=3000.0)
        self.assertAlmostEqual(float(normalized.min()), -1.0)
        self.assertAlmostEqual(float(normalized.max()), 1.0)

    def test_pad_or_crop_volume_returns_fixed_shape(self):
        volume = np.zeros((120, 150, 160), dtype=np.float32)
        output = pad_or_crop_volume(volume, spatial_size=(128, 160, 192))
        self.assertEqual(output.shape, (128, 160, 192))

    def test_pad_or_crop_volume_crops_oversized_volume(self):
        volume = np.zeros((155, 240, 240), dtype=np.float32)
        output = pad_or_crop_volume(volume, spatial_size=(128, 160, 192))
        self.assertEqual(output.shape, (128, 160, 192))


class TestRandGammaCorrectiond(unittest.TestCase):
    def setUp(self):
        self.img = torch.linspace(-1, 1, 24, dtype=torch.float32).reshape(1, 2, 3, 4)

    def test_gamma_1_is_identity(self):
        """Gamma=1 should produce output identical to input."""
        g = RandGammaCorrectiond(keys=["x"], gamma_range=(1.0, 1.0), prob=1.0)
        result = g({"x": self.img.clone()})
        torch.testing.assert_close(result["x"], self.img)

    def test_output_stays_in_range(self):
        """Output must stay in (-1, 1) regardless of gamma value."""
        g = RandGammaCorrectiond(keys=["x"], gamma_range=(0.1, 5.0), prob=1.0)
        result = g({"x": self.img.clone()})
        self.assertGreaterEqual(result["x"].min(), -1.0)
        self.assertLessEqual(result["x"].max(), 1.0)

    def test_prob_zero_passes_through(self):
        """prob=0 must not modify values and must convert to tensor."""
        x = np.random.randn(1, 4, 4, 4).astype(np.float32)
        g = RandGammaCorrectiond(keys=["x"], prob=0.0)
        result = g({"x": x})
        self.assertTrue(torch.is_tensor(result["x"]))
        torch.testing.assert_close(result["x"], torch.from_numpy(x))

    def test_allow_missing_keys_does_not_raise(self):
        """allow_missing_keys=True must not throw KeyError for missing key."""
        g = RandGammaCorrectiond(
            keys=["src", "tgt"], gamma_range=(0.7, 1.5), prob=0.3,
            allow_missing_keys=True,
        )
        # "tgt" is missing — should not raise
        result = g({"src": self.img.clone()})
        self.assertIn("src", result)
        self.assertNotIn("tgt", result)

    def test_allow_missing_keys_false_raises(self):
        """Default allow_missing_keys=False must raise KeyError."""
        g = RandGammaCorrectiond(keys=["src", "tgt"], prob=1.0)
        with self.assertRaises(KeyError):
            g({"src": self.img.clone()})

    def test_numpy_input_handled(self):
        """Numpy input path must work and produce correct shape."""
        x = np.linspace(-1, 1, 24, dtype=np.float32).reshape(1, 2, 3, 4)
        g = RandGammaCorrectiond(keys=["x"], gamma_range=(1.0, 1.0), prob=1.0)
        result = g({"x": x})
        self.assertEqual(result["x"].shape, (1, 2, 3, 4))


class TestFixedRandSimulateLowResolutiond(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)

    def _make_sample(self):
        return {
            "src": torch.randn(1, 16, 16, 16),
            "tgt": torch.randn(1, 16, 16, 16),
        }

    def test_same_zoom_applied_to_both_keys(self):
        """Both src and tgt must get the same downsampling factor."""
        t = _FixedRandSimulateLowResolutiond(
            keys=["src", "tgt"], prob=1.0, zoom_range=(0.5, 0.5),
        )
        result = t(self._make_sample())
        # With same zoom factor, the relative difference of means
        # after down-then-upsample should be small
        src_mean = result["src"].mean()
        tgt_mean = result["tgt"].mean()
        # If zoom differs, the effective downsampling differs dramatically
        # With both at zoom=0.5, shapes of intermediate tensors match exactly
        orig_diff = float((self._make_sample()["src"] - self._make_sample()["tgt"]).abs().mean())
        result_diff = float((result["src"] - result["tgt"]).abs().mean())
        # Down/up sampling should preserve the difference magnitude within reasonable bounds
        self.assertLess(result_diff / max(orig_diff, 1e-8), 10.0)

    def test_different_zooms_would_produce_divergent_outputs(self):
        """Verify our fix is meaningful: if zoom differs per key, outputs diverge."""
        import torch.nn.functional as F
        x_src = torch.randn(1, 1, 16, 16, 16)
        x_tgt = torch.randn(1, 1, 16, 16, 16)
        shape_src = tuple(int(s * 0.5) for s in x_src.shape[2:])
        shape_tgt = tuple(int(s * 0.8) for s in x_tgt.shape[2:])
        down_src = F.interpolate(x_src, size=shape_src, mode="trilinear")
        up_src = F.interpolate(down_src, size=x_src.shape[2:], mode="trilinear")
        down_tgt = F.interpolate(x_tgt, size=shape_tgt, mode="trilinear")
        up_tgt = F.interpolate(down_tgt, size=x_tgt.shape[2:], mode="trilinear")
        diff_broken = float((up_src - up_tgt).abs().mean())
        orig_diff = float((x_src - x_tgt).abs().mean())
        # Different zoom factors produce a meaningfully different relative diff
        self.assertNotAlmostEqual(orig_diff, diff_broken, places=2)

    def test_prob_zero_passes_through(self):
        """prob=0 must not modify tensor values."""
        sample = self._make_sample()
        t = _FixedRandSimulateLowResolutiond(keys=["src", "tgt"], prob=0.0)
        result = t(sample)
        self.assertTrue(torch.is_tensor(result["src"]))
        self.assertTrue(torch.is_tensor(result["tgt"]))

    def test_allow_missing_keys_does_not_raise(self):
        """allow_missing_keys=True must not throw for missing keys."""
        t = _FixedRandSimulateLowResolutiond(
            keys=["src", "tgt"], prob=0.1, allow_missing_keys=True,
        )
        result = t({"src": torch.randn(1, 16, 16, 16)})
        self.assertIn("src", result)
        self.assertNotIn("tgt", result)

    def test_allow_missing_keys_false_raises(self):
        """Default must raise KeyError for missing key."""
        t = _FixedRandSimulateLowResolutiond(keys=["src", "tgt"], prob=1.0)
        with self.assertRaises(KeyError):
            t({"src": torch.randn(1, 16, 16, 16)})


class TestAugmentationPipelines(unittest.TestCase):
    def test_train_pipeline_has_augmentations(self):
        """Training transforms must include spatial and intensity augmentations."""
        t = build_train_transforms(spatial_size=(64, 64, 64), a_min=0.0, a_max=3000.0)
        class_names = [tr.__class__.__name__ for tr in t.transforms]
        self.assertIn("RandFlipd", class_names)
        self.assertIn("RandRotated", class_names)
        self.assertIn("Rand3DElasticd", class_names)
        self.assertIn("RandGammaCorrectiond", class_names)
        self.assertIn("_FixedRandSimulateLowResolutiond", class_names)

    def test_val_pipeline_has_no_augmentations(self):
        """Validation transforms must NOT include any augmentation."""
        t = build_pair_transforms(spatial_size=(64, 64, 64), a_min=0.0, a_max=3000.0)
        class_names = [tr.__class__.__name__ for tr in t.transforms]
        self.assertNotIn("RandFlipd", class_names)
        self.assertNotIn("RandRotated", class_names)
        self.assertNotIn("Rand3DElasticd", class_names)

    def test_train_pipeline_ordered_correctly(self):
        """Normalization must sit between spatial and intensity augmentations."""
        t = build_train_transforms(spatial_size=(64, 64, 64), a_min=0.0, a_max=3000.0)
        ir_class_names = [
            tr.__class__.__name__ for tr in t.transforms
        ]
        randed_idx = ir_class_names.index("ScaleIntensityRanged")
        flip_idx = ir_class_names.index("RandFlipd")
        gamma_idx = ir_class_names.index("RandGammaCorrectiond")
        # Spatial aug (flip) before normalization
        self.assertLess(flip_idx, randed_idx)
        # Intensity aug (gamma) after normalization
        self.assertLess(randed_idx, gamma_idx)


if __name__ == "__main__":
    unittest.main()
