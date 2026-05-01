import os
import unittest
from dataclasses import fields

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from jedi.training.schedule import estimate_total_steps
from jedi.training.trainer_config import TrainerConfig

_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "src", "jedi", "config")


class SizedLoader:
    def __init__(self, length: int):
        self.length = length

    def __len__(self):
        return self.length


class TestTrainerConfig(unittest.TestCase):
    def test_train_configs_include_supported_trainer_fields(self):
        supported_fields = {field.name for field in fields(TrainerConfig)}
        with initialize_config_dir(config_dir=os.path.abspath(_CONFIG_DIR), version_base=None):
            encoder_cfg = compose(config_name="train_encoder")
            decoder_cfg = compose(config_name="train_decoder")

        self.assertLessEqual(set(encoder_cfg.trainer.keys()), supported_fields)
        self.assertLessEqual(set(decoder_cfg.trainer.keys()), supported_fields)
        self.assertEqual(TrainerConfig.from_config(encoder_cfg.trainer).max_epochs, 100)
        self.assertEqual(TrainerConfig.from_config(decoder_cfg.trainer).gradient_clip_val, 1.0)

    def test_estimate_total_steps_prefers_max_steps(self):
        cfg = OmegaConf.create({"max_epochs": None, "max_steps": 25, "accumulate_grad_batches": 4})
        self.assertEqual(estimate_total_steps(SizedLoader(10), cfg), 25)

    def test_estimate_total_steps_accounts_for_batch_limits_and_accumulation(self):
        cfg = OmegaConf.create({"max_epochs": 3, "max_steps": -1, "limit_train_batches": 5, "accumulate_grad_batches": 2})
        self.assertEqual(estimate_total_steps(SizedLoader(10), cfg), 9)
