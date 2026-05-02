import tempfile
import unittest
from pathlib import Path

from omegaconf import OmegaConf

from jedi.training.logging import save_resolved_config


class TestTrainingLogging(unittest.TestCase):
    def test_save_resolved_config_writes_resolved_yaml(self):
        cfg = OmegaConf.create({"trainer": {"default_root_dir": "${root}"}, "root": "/tmp/run"})
        with tempfile.TemporaryDirectory() as tmpdir:
            save_resolved_config(cfg, tmpdir)
            text = Path(tmpdir, "resolved_config.yaml").read_text(encoding="utf-8")
        self.assertIn("default_root_dir: /tmp/run", text)

    def test_save_resolved_config_keeps_missing_placeholders(self):
        cfg = OmegaConf.create({"wandb": {"save_dir": "???"}})
        with tempfile.TemporaryDirectory() as tmpdir:
            save_resolved_config(cfg, tmpdir)
            text = Path(tmpdir, "resolved_config.yaml").read_text(encoding="utf-8")
        self.assertIn("save_dir: ???", text)


if __name__ == "__main__":
    unittest.main()
