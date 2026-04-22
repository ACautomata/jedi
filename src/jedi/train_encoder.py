import hydra
import lightning as pl
from hydra.utils import instantiate
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from omegaconf import DictConfig, OmegaConf

from jedi.data.brats import build_dataloader
from jedi.models import CrossModalityJEPA
from jedi.training.encoder_module import EncoderTrainingModule


@hydra.main(version_base=None, config_path="config", config_name="train_encoder")
def main(cfg: DictConfig):
    pl.seed_everything(cfg.seed)
    encoder = instantiate(cfg.model.encoder)
    projector = instantiate(cfg.model.projector)
    predictor = instantiate(cfg.model.predictor)
    pred_proj = instantiate(cfg.model.pred_proj)
    regularizer = instantiate(cfg.model.regularizer)
    modality_embedder_cfg = OmegaConf.select(cfg.model, "modality_embedder", default=None)
    modality_embedder = instantiate(modality_embedder_cfg) if modality_embedder_cfg else None
    model = CrossModalityJEPA(
        encoder=encoder,
        projector=projector,
        predictor=predictor,
        pred_proj=pred_proj,
        modality_embedder=modality_embedder,
    )
    train_loader = build_dataloader(cfg.data.data_dir, "train", tuple(cfg.data.fixed_mapping), cfg.data.batch_size, cfg.data.num_workers, tuple(cfg.data.spatial_size))
    val_loader = build_dataloader(cfg.data.data_dir, "val", tuple(cfg.data.fixed_mapping), cfg.data.batch_size, cfg.data.num_workers, tuple(cfg.data.spatial_size))
    total_steps = len(train_loader) * cfg.trainer.max_epochs
    warmup_steps = OmegaConf.select(cfg, "scheduler.warmup_steps", default=0)
    module = EncoderTrainingModule(
        model=model,
        regularizer=regularizer,
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
        sigreg_weight=cfg.loss.sigreg_weight,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
    )
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            dirpath="checkpoints/encoder",
            filename="encoder-{epoch:03d}-{val/loss:.4f}",
            monitor="val/loss",
            save_top_k=3,
            mode="min",
        ),
    ]
    custom_callbacks_cfg = OmegaConf.select(cfg, "callbacks", default=None)
    if custom_callbacks_cfg:
        for cb_cfg in custom_callbacks_cfg.values():
            callbacks.append(instantiate(cb_cfg))
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        gradient_clip_val=OmegaConf.select(cfg.trainer, "gradient_clip_val", default=1.0),
        gradient_clip_algorithm="norm",
        callbacks=callbacks,
        logger=CSVLogger("logs", name="encoder_stage"),
    )
    trainer.fit(module, train_loader, val_loader)


if __name__ == "__main__":
    main()
