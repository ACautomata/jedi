import unittest

import lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from jedi.models import CrossModalityJEPA, MLP, ViT3DEncoder
from jedi.models.decoder3d import VolumeDecoder3D
from jedi.models.predictor import LatentPredictor
from jedi.training.decoder_module import DecoderTrainingModule


class TinyDecoderDataset(Dataset):
    def __init__(self):
        self.src = torch.randn(1, 1, 32, 32, 32)
        self.tgt = torch.randn(1, 1, 32, 32, 32)

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return {"src": self.src[index], "tgt": self.tgt[index]}


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
        output = module.training_step(batch, 0)
        self.assertTrue(torch.isfinite(output["loss"]))
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
        output = module.validation_step(batch, 0)
        self.assertTrue(torch.isfinite(output["loss"]))
        self.assertIn("prediction", output)
        self.assertIn("target", output)

    def test_pcgrad_sets_decoder_grads(self):
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
        objective_a = sum(param.sum() for param in decoder.parameters())
        objective_b = sum(-param.sum() for param in decoder.parameters())
        module._pc_backward([objective_a, objective_b])
        self.assertTrue(all(param.grad is not None for param in decoder.parameters() if param.requires_grad))
        self.assertTrue(all(torch.allclose(param.grad, torch.zeros_like(param.grad)) for param in decoder.parameters() if param.requires_grad))

    def test_trainer_step_updates_decoder_parameters(self):
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
            model=model,
            decoder=decoder,
            lr=1e-4,
            weight_decay=1e-4,
            gradient_clip_val=1.0,
        )
        before = [param.detach().clone() for param in decoder.parameters() if param.requires_grad]
        loader = DataLoader(TinyDecoderDataset(), batch_size=1)
        trainer = pl.Trainer(
            accelerator="cpu",
            devices=1,
            max_epochs=1,
            limit_train_batches=1,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
            callbacks=[],
        )
        trainer.fit(module, loader)
        after = [param.detach() for param in decoder.parameters() if param.requires_grad]
        self.assertTrue(any(not torch.allclose(old, new) for old, new in zip(before, after)))

    def test_wavelet_loss_is_always_enabled(self):
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
        self.assertIsNotNone(module.wavelet_loss)


if __name__ == "__main__":
    unittest.main()
