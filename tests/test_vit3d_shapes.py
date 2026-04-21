import unittest

import torch

from jedi.models.vit3d import ViT3DEncoder


class TestViT3DShapes(unittest.TestCase):
    def test_patch_token_shape(self):
        model = ViT3DEncoder(
            image_size=(32, 32, 32),
            patch_size=(8, 8, 8),
            in_channels=1,
            embed_dim=64,
            depth=2,
            heads=4,
            dim_head=16,
            mlp_dim=128,
        )
        x = torch.randn(2, 1, 32, 32, 32)
        output = model(x)
        self.assertEqual(output["patch_embeddings"].shape, (2, 64, 64))
        self.assertEqual(output["cls_embedding"].shape, (2, 64))


if __name__ == "__main__":
    unittest.main()
