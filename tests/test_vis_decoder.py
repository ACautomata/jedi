import unittest

import torch

from jedi.models import CrossModalityJEPA, MLP, ViT3DEncoder
from jedi.models.decoder3d import VolumeDecoder3D
from jedi.models.predictor import LatentPredictor
from jedi.models.vis_decoder import CrossAttention, CrossAttnBlock, VisualizationDecoder
from jedi.training.decoder_module import DecoderTrainingModule


class TestCrossAttention(unittest.TestCase):
    def test_cross_attention_shape(self):
        attn = CrossAttention(dim=64, heads=4, dim_head=16)
        x = torch.randn(2, 16, 64)
        ctx = torch.randn(2, 1, 64)
        out = attn(x, ctx)
        self.assertEqual(out.shape, (2, 16, 64))
        self.assertTrue(torch.isfinite(out).all())

    def test_cross_attention_many_queries_single_context(self):
        attn = CrossAttention(dim=32, heads=2, dim_head=16)
        x = torch.randn(1, 768, 32)
        ctx = torch.randn(1, 1, 32)
        out = attn(x, ctx)
        self.assertEqual(out.shape, (1, 768, 32))
        self.assertTrue(torch.isfinite(out).all())


class TestCrossAttnBlock(unittest.TestCase):
    def test_cross_attn_block_shape(self):
        block = CrossAttnBlock(dim=64, heads=4, dim_head=16, mlp_dim=256)
        x = torch.randn(2, 16, 64)
        ctx = torch.randn(2, 1, 64)
        out = block(x, ctx)
        self.assertEqual(out.shape, (2, 16, 64))
        self.assertTrue(torch.isfinite(out).all())


class TestVisualizationDecoder(unittest.TestCase):
    def test_decoder_forward_shape_and_range(self):
        decoder = VisualizationDecoder(
            cls_dim=64,
            hidden_dim=128,
            image_size=(16, 16, 16),
            patch_size=(4, 4, 4),
            out_channels=1,
            depth=2,
            heads=4,
            dim_head=32,
            mlp_dim=512,
        )
        cls_emb = torch.randn(2, 64)
        out = decoder(cls_emb, (4, 4, 4))
        self.assertEqual(out.shape, (2, 1, 16, 16, 16))
        self.assertTrue(out.min() >= -1.0)
        self.assertTrue(out.max() <= 1.0)
        self.assertTrue(torch.isfinite(out).all())

    def test_decoder_forward_large(self):
        decoder = VisualizationDecoder(
            cls_dim=256,
            hidden_dim=512,
            image_size=(32, 32, 32),
            patch_size=(8, 8, 8),
            out_channels=1,
            depth=2,
            heads=4,
            dim_head=64,
            mlp_dim=1024,
        )
        cls_emb = torch.randn(1, 256)
        out = decoder(cls_emb, (4, 4, 4))
        self.assertEqual(out.shape, (1, 1, 32, 32, 32))
        self.assertTrue(torch.isfinite(out).all())

    def test_decoder_forward_production_scale(self):
        decoder = VisualizationDecoder(
            cls_dim=256,
            hidden_dim=512,
            image_size=(64, 64, 64),
            patch_size=(8, 8, 8),
            out_channels=1,
            depth=2,
            heads=8,
            dim_head=64,
            mlp_dim=2048,
        )
        cls_emb = torch.randn(2, 256)
        out = decoder(cls_emb, (8, 8, 8))
        self.assertEqual(out.shape, (2, 1, 64, 64, 64))
        self.assertTrue(torch.isfinite(out).all())

    def test_mlp_dim_defaults_to_hidden_times_4(self):
        decoder = VisualizationDecoder(
            cls_dim=64,
            hidden_dim=128,
            image_size=(16, 16, 16),
            patch_size=(4, 4, 4),
            out_channels=1,
            depth=1,
            heads=2,
            dim_head=32,
        )
        self.assertEqual(decoder.blocks[0].mlp.net[1].out_features, 512)
        cls_emb = torch.randn(2, 64)
        out = decoder(cls_emb, (4, 4, 4))
        self.assertEqual(out.shape, (2, 1, 16, 16, 16))

    def test_grid_size_mismatch_raises(self):
        decoder = VisualizationDecoder(
            cls_dim=64,
            hidden_dim=128,
            image_size=(16, 16, 16),
            patch_size=(4, 4, 4),
            out_channels=1,
            depth=1,
            heads=2,
            dim_head=32,
        )
        cls_emb = torch.randn(2, 64)
        with self.assertRaises(ValueError):
            decoder(cls_emb, (5, 5, 5))

    def test_grid_size_wrong_dim_assignment_raises(self):
        decoder = VisualizationDecoder(
            cls_dim=64,
            hidden_dim=128,
            image_size=(16, 16, 16),
            patch_size=(4, 4, 4),
            out_channels=1,
            depth=1,
            heads=2,
            dim_head=32,
        )
        cls_emb = torch.randn(2, 64)
        with self.assertRaises(ValueError):
            decoder(cls_emb, (2, 8, 2))

    def test_use_cls_embedding_training_step(self):
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
        decoder = VisualizationDecoder(
            cls_dim=64,
            hidden_dim=128,
            image_size=(32, 32, 32),
            patch_size=(8, 8, 8),
            out_channels=1,
            depth=2,
            heads=4,
            dim_head=32,
            mlp_dim=512,
        )
        module = DecoderTrainingModule(
            model=model,
            decoder=decoder,
            lr=1e-4,
            weight_decay=1e-4,
            use_cls_embedding=True,
        )
        batch = {
            "src": torch.randn(2, 1, 32, 32, 32),
            "tgt": torch.randn(2, 1, 32, 32, 32),
        }
        with torch.no_grad():
            src_output, _ = model.encode_src_tgt(batch["src"], batch["tgt"])
            decoder_input = module._get_decoder_input(src_output, batch)
            self.assertTrue(torch.equal(decoder_input, src_output["cls_embedding"]),
                            "decoder input should be the raw cls_embedding, not the predictor output")
        output = module.training_step(batch, 0)
        self.assertTrue(torch.isfinite(output["loss"]))
        self.assertTrue(all(not param.requires_grad for param in model.parameters()))
        self.assertTrue(any(param.requires_grad for param in decoder.parameters()))

    def test_use_cls_embedding_validation_step(self):
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
        decoder = VisualizationDecoder(
            cls_dim=64,
            hidden_dim=128,
            image_size=(32, 32, 32),
            patch_size=(8, 8, 8),
            out_channels=1,
            depth=2,
            heads=4,
            dim_head=32,
            mlp_dim=512,
        )
        module = DecoderTrainingModule(
            model=model,
            decoder=decoder,
            lr=1e-4,
            weight_decay=1e-4,
            use_cls_embedding=True,
        )
        batch = {
            "src": torch.randn(2, 1, 32, 32, 32),
            "tgt": torch.randn(2, 1, 32, 32, 32),
        }
        output = module.validation_step(batch, 0)
        self.assertTrue(torch.isfinite(output["loss"]))

    def test_visualization_decoder_use_cls_embedding_false_raises(self):
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
        decoder = VisualizationDecoder(
            cls_dim=64,
            hidden_dim=128,
            image_size=(32, 32, 32),
            patch_size=(8, 8, 8),
            out_channels=1,
            depth=1,
            heads=4,
            dim_head=32,
            mlp_dim=512,
        )
        with self.assertRaises(ValueError, msg="use_cls_embedding=False with VisualizationDecoder should raise"):
            DecoderTrainingModule(
                model=model,
                decoder=decoder,
                lr=1e-4,
                weight_decay=1e-4,
                use_cls_embedding=False,
            )

    def test_volume_decoder_use_cls_embedding_true_raises(self):
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
        with self.assertRaises(ValueError, msg="use_cls_embedding=True with VolumeDecoder3D should raise"):
            DecoderTrainingModule(
                model=model,
                decoder=decoder,
                lr=1e-4,
                weight_decay=1e-4,
                use_cls_embedding=True,
            )

    def test_use_cls_embedding_backward_compatible_default(self):
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
        self.assertFalse(module.use_cls_embedding)
        batch = {
            "src": torch.randn(2, 1, 32, 32, 32),
            "tgt": torch.randn(2, 1, 32, 32, 32),
        }
        output = module.training_step(batch, 0)
        self.assertTrue(torch.isfinite(output["loss"]))


if __name__ == "__main__":
    unittest.main()
