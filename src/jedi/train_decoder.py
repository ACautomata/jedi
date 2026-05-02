import hydra
import lightning as pl
from hydra.utils import instantiate
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from omegaconf import DictConfig, OmegaConf

from jedi.data.brats import build_dataloader
from jedi.models import CrossModalityJEPA
from jedi.training.callbacks import LossMetricsCallback
from jedi.training.decoder_module import DecoderTrainingModule
from jedi.training.logging import save_resolved_config, update_wandb_config
from jedi.training.schedule import estimate_total_steps
from jedi.training.trainer_config import TrainerConfig
from jedi.utils import load_encoder_side_checkpoint


@hydra.main(version_base=None, config_path="config", config_name="train_decoder")
def main(cfg: DictConfig):
    pl.seed_everything(cfg.seed)
    encoder = instantiate(cfg.model.encoder)
    projector = instantiate(cfg.model.projector)
    predictor = instantiate(cfg.model.predictor)
    pred_proj = instantiate(cfg.model.pred_proj)
    modality_embedder_cfg = OmegaConf.select(cfg.model, "modality_embedder", default=None)
    modality_embedder = instantiate(modality_embedder_cfg) if modality_embedder_cfg else None
    model = CrossModalityJEPA(
        encoder=encoder,
        projector=projector,
        predictor=predictor,
        pred_proj=pred_proj,
        modality_embedder=modality_embedder,
    )
    model = load_encoder_side_checkpoint(model, cfg.decoder_model.encoder_checkpoint)
    decoder = instantiate(cfg.decoder_model.decoder)
    cache_dir = OmegaConf.select(cfg.data, "cache_dir", default=None)
    train_loader = build_dataloader(cfg.data.data_dir, "train", tuple(cfg.data.fixed_mapping), cfg.data.batch_size, cfg.data.num_workers, tuple(cfg.data.spatial_size), cache_dir=cache_dir)
    val_data_dir = cfg.data.get("val_data_dir") or cfg.data.data_dir
    val_loader = build_dataloader(val_data_dir, "val", tuple(cfg.data.fixed_mapping), cfg.data.batch_size, cfg.data.num_workers, tuple(cfg.data.spatial_size), cache_dir=cache_dir)
    total_steps = estimate_total_steps(train_loader, cfg.trainer)
    warmup_steps = OmegaConf.select(cfg, "scheduler.warmup_steps", default=0)
    use_cls_embedding = OmegaConf.select(cfg.decoder_model, "use_cls_embedding", default=False)
    module = DecoderTrainingModule(
        model=model,
        decoder=decoder,
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        use_cls_embedding=use_cls_embedding,
        gradient_clip_val=OmegaConf.select(cfg.trainer, "gradient_clip_val", default=1.0),
    )
    callbacks = [
        LossMetricsCallback(),
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            dirpath="checkpoints/decoder",
            filename="decoder-{epoch:03d}-{val/loss:.4f}",
            monitor="val/loss",
            save_top_k=3,
            mode="min",
        ),
    ]
    custom_callbacks_cfg = OmegaConf.select(cfg, "callbacks", default=None)
    if custom_callbacks_cfg:
        for cb_cfg in custom_callbacks_cfg.values():
            callbacks.append(instantiate(cb_cfg))
    loggers = [CSVLogger("logs", name="decoder_stage")]
    wandb_cfg = OmegaConf.select(cfg, "wandb", default=None)
    if wandb_cfg and OmegaConf.select(wandb_cfg, "enabled", default=False):
        loggers.append(
            WandbLogger(
                project=wandb_cfg.project,
                name=OmegaConf.select(wandb_cfg, "name", default="decoder_stage"),
                save_dir=OmegaConf.select(wandb_cfg, "save_dir", default="logs"),
            )
        )
    save_resolved_config(cfg, OmegaConf.select(cfg.trainer, "default_root_dir", default=None))
    update_wandb_config(loggers, cfg)
    trainer_config = TrainerConfig.from_config(cfg.trainer)
    trainer = trainer_config.build(
        gradient_clip_val=None,
        gradient_clip_algorithm=None,
        callbacks=callbacks,
        logger=loggers,
    )
    trainer.fit(module, train_loader, val_loader)


if __name__ == "__main__":
    main()
