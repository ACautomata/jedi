import unittest

import numpy as np

from jedi.data.transforms import normalize_to_unit_range, pad_or_crop_volume


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


if __name__ == "__main__":
    unittest.main()
