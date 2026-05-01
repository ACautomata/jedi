from dataclasses import dataclass
from inspect import signature
from typing import Any

import lightning as pl
from omegaconf import DictConfig, OmegaConf


@dataclass(frozen=True)
class TrainerConfig:
    accelerator: str = "gpu"
    strategy: str = "auto"
    devices: int | str | list[int] = 1
    num_nodes: int = 1
    precision: str | None = "16-mixed"
    logger: Any = None
    callbacks: Any = None
    fast_dev_run: bool | int = False
    max_epochs: int | None = 100
    min_epochs: int | None = None
    max_steps: int = -1
    min_steps: int | None = None
    max_time: str | dict[str, int] | None = None
    limit_train_batches: int | float | None = None
    limit_val_batches: int | float | None = None
    limit_test_batches: int | float | None = None
    limit_predict_batches: int | float | None = None
    overfit_batches: int | float = 0.0
    val_check_interval: int | float | None = None
    check_val_every_n_epoch: int | None = 1
    num_sanity_val_steps: int | None = 2
    log_every_n_steps: int | None = 20
    enable_checkpointing: bool | None = True
    enable_progress_bar: bool | None = True
    enable_model_summary: bool | None = True
    accumulate_grad_batches: int = 1
    gradient_clip_val: int | float | None = 1.0
    gradient_clip_algorithm: str | None = "norm"
    deterministic: bool | str | None = False
    benchmark: bool | None = True
    inference_mode: bool = True
    use_distributed_sampler: bool = True
    profiler: Any = None
    detect_anomaly: bool = False
    barebones: bool = False
    plugins: Any = None
    sync_batchnorm: bool = False
    reload_dataloaders_every_n_epochs: int = 0
    default_root_dir: str | None = None
    enable_autolog_hparams: bool = True
    model_registry: Any = None

    @classmethod
    def from_config(cls, cfg: DictConfig) -> "TrainerConfig":
        return cls(**OmegaConf.to_container(cfg, resolve=True))

    def build(self, **overrides: Any) -> pl.Trainer:
        supported_params = signature(pl.Trainer).parameters
        params = {
            field: value
            for field, value in self.__dict__.items()
            if value is not None and field in supported_params
        }
        params.update({field: value for field, value in overrides.items() if field in supported_params})
        return pl.Trainer(**params)
