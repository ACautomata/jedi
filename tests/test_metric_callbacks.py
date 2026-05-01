import unittest
from unittest.mock import Mock

import torch

from jedi.training.callbacks import LatentEvalMetricsCallback, LossMetricsCallback, ReconstructionEvalMetricsCallback, TrainingDynamicsCallback


class DummyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2))
        self.logged = []

    def log(self, name, value, **kwargs):
        self.logged.append((name, value, kwargs))


class TestMetricCallbacks(unittest.TestCase):
    def test_loss_metrics_callback_logs_available_losses(self):
        module = DummyModule()
        outputs = {
            "loss": torch.tensor(1.0),
            "l1_loss": torch.tensor(0.5),
            "wavelet_loss": torch.tensor(0.25),
        }
        LossMetricsCallback().on_train_batch_end(Mock(), module, outputs, {}, 0)
        names = [name for name, _, _ in module.logged]
        self.assertIn("train/loss", names)
        self.assertIn("train/l1_loss", names)
        self.assertIn("train/wavelet_loss", names)

    def test_latent_eval_metrics_callback_logs_latent_metrics(self):
        module = DummyModule()
        outputs = {
            "pred_emb": torch.randn(2, 4, 8),
            "tgt_emb": torch.randn(2, 4, 8),
        }
        LatentEvalMetricsCallback(log_interval=1).on_train_batch_end(Mock(), module, outputs, {}, 0)
        names = [name for name, _, _ in module.logged]
        self.assertIn("train/latent_mse", names)
        self.assertIn("train/latent_mae", names)
        self.assertIn("train/latent_cosine", names)
        self.assertIn("train/latent_pearson", names)

    def test_reconstruction_eval_metrics_callback_logs_paper_metrics(self):
        module = DummyModule()
        outputs = {
            "prediction": torch.randn(2, 1, 8, 8, 8),
            "target": torch.randn(2, 1, 8, 8, 8),
        }
        ReconstructionEvalMetricsCallback().on_validation_batch_end(Mock(), module, outputs, {}, 0)
        names = [name for name, _, _ in module.logged]
        self.assertIn("val/mse", names)
        self.assertIn("val/mae", names)
        self.assertIn("val/rmse", names)
        self.assertIn("val/nrmse", names)
        self.assertIn("val/psnr", names)
        self.assertIn("val/ssim", names)

    def test_training_dynamics_callback_logs_gradient_metrics(self):
        module = DummyModule()
        module.weight.grad = torch.ones_like(module.weight)
        TrainingDynamicsCallback(log_interval=1).on_train_batch_end(Mock(), module, {}, {}, 0)
        names = [name for name, _, _ in module.logged]
        self.assertIn("dynamics/grad_norm", names)
        self.assertIn("dynamics/param_norm", names)
        self.assertIn("dynamics/grad_to_param_norm", names)


if __name__ == "__main__":
    unittest.main()
