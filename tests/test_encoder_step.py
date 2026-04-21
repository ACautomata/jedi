import unittest

import torch

from jedi.models import CrossModalityJEPA, MLP, ViT3DEncoder
from jedi.models.predictor import LatentPredictor
from jedi.models.regularizers import SIGReg
from jedi.training.encoder_module import EncoderTrainingModule


class TestEncoderStep(unittest.TestCase):
    def test_stage1_step_returns_finite_loss(self):
        encoder = ViT3DEncoder(
            image_size=(32, 32, 32),
            patch_size=(8, 8, 8),
            in_channels=1,
            embed_dim=64,
            depth=2,
            heads=4,
            dim_head=16,
            mlp_dim=128,
        )
        projector = MLP(input_dim=64, hidden_dim=128, output_dim=64, norm_fn="LayerNorm")
        predictor = LatentPredictor(input_dim=64, hidden_dim=64)
        pred_proj = MLP(input_dim=64, hidden_dim=128, output_dim=64, norm_fn="LayerNorm")
        model = CrossModalityJEPA(encoder=encoder, projector=projector, predictor=predictor, pred_proj=pred_proj)
        module = EncoderTrainingModule(model=model, regularizer=SIGReg(num_proj=32), lr=1e-4, weight_decay=1e-4, sigreg_weight=0.01)
        batch = {
            "src": torch.randn(2, 1, 32, 32, 32),
            "tgt": torch.randn(2, 1, 32, 32, 32),
        }
        loss = module.training_step(batch, 0)
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
