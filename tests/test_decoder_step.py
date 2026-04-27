import unittest

import torch

from jedi.models import CrossModalityJEPA, MLP, ViT3DEncoder
from jedi.models.decoder3d import VolumeDecoder3D
from jedi.models.predictor import LatentPredictor
from jedi.training.decoder_module import DecoderTrainingModule


class TestDecoderStep(unittest.TestCase):
    def test_stage2_step_freezes_encoder_side_and_returns_finite_loss(self):
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
        decoder = VolumeDecoder3D(embed_dim=64, patch_size=(8, 8, 8), out_channels=1, hidden_channels=32)
        module = DecoderTrainingModule(model=model, decoder=decoder, lr=1e-4, weight_decay=1e-4)
        batch = {
            "src": torch.randn(2, 1, 32, 32, 32),
            "tgt": torch.randn(2, 1, 32, 32, 32),
        }
        loss = module.training_step(batch, 0)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(not param.requires_grad for param in model.parameters()))
        self.assertTrue(any(param.requires_grad for param in decoder.parameters()))

    def test_decoder_validation_step_returns_finite_metrics(self):
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
        decoder = VolumeDecoder3D(embed_dim=64, patch_size=(8, 8, 8), out_channels=1, hidden_channels=32)
        module = DecoderTrainingModule(model=model, decoder=decoder, lr=1e-4, weight_decay=1e-4)
        batch = {
            "src": torch.randn(2, 1, 32, 32, 32),
            "tgt": torch.randn(2, 1, 32, 32, 32),
        }
        loss = module.validation_step(batch, 0)
        self.assertTrue(torch.isfinite(loss))


    def test_wavelet_enabled_returns_combined_loss(self):
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
        decoder = VolumeDecoder3D(embed_dim=64, patch_size=(8, 8, 8), out_channels=1, hidden_channels=32)
        module = DecoderTrainingModule(
            model=model, decoder=decoder, lr=1e-4, weight_decay=1e-4,
            wavelet_weight=1.0,
            wavelet_config={"wave": "haar", "J": 1, "alpha_low": 0.5, "alpha_high": 0.5},
        )
        self.assertIsNotNone(module.wavelet_loss)
        batch = {
            "src": torch.randn(2, 1, 32, 32, 32),
            "tgt": torch.randn(2, 1, 32, 32, 32),
        }
        loss = module.training_step(batch, 0)
        self.assertTrue(torch.isfinite(loss))

    def test_wavelet_weight_zero_no_module_created(self):
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
        decoder = VolumeDecoder3D(embed_dim=64, patch_size=(8, 8, 8), out_channels=1, hidden_channels=32)
        module = DecoderTrainingModule(model=model, decoder=decoder, lr=1e-4, weight_decay=1e-4)
        self.assertIsNone(module.wavelet_loss)


if __name__ == "__main__":
    unittest.main()
