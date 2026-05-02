from pathlib import Path

from hydra.core.hydra_config import HydraConfig
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf


def _resolved_container(cfg: DictConfig):
    return OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)


def save_resolved_config(cfg: DictConfig, output_dir: str | None = None):
    if output_dir is None and HydraConfig.initialized():
        output_dir = HydraConfig.get().runtime.output_dir
    if not output_dir:
        return
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    resolved = OmegaConf.create(_resolved_container(cfg))
    Path(output_dir, "resolved_config.yaml").write_text(OmegaConf.to_yaml(resolved), encoding="utf-8")


def update_wandb_config(loggers, cfg: DictConfig):
    resolved = _resolved_container(cfg)
    for logger in loggers:
        if isinstance(logger, WandbLogger):
            logger.experiment.config.update(resolved, allow_val_change=True)
