from math import ceil
from typing import Sized

from omegaconf import DictConfig, OmegaConf


def estimate_total_steps(train_loader: Sized, trainer_cfg: DictConfig) -> int:
    max_steps = OmegaConf.select(trainer_cfg, "max_steps", default=-1)
    if max_steps is not None and max_steps > 0:
        return max_steps

    max_epochs = OmegaConf.select(trainer_cfg, "max_epochs", default=None)
    if max_epochs is None:
        return 0

    batches_per_epoch = len(train_loader)
    limit_train_batches = OmegaConf.select(trainer_cfg, "limit_train_batches", default=None)
    if isinstance(limit_train_batches, int):
        batches_per_epoch = min(batches_per_epoch, limit_train_batches)
    elif isinstance(limit_train_batches, float):
        batches_per_epoch = int(batches_per_epoch * limit_train_batches)

    accumulate_grad_batches = OmegaConf.select(trainer_cfg, "accumulate_grad_batches", default=1)
    return ceil(batches_per_epoch / accumulate_grad_batches) * max_epochs
