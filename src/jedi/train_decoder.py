import hydra
import lightning as pl
from hydra.utils import instantiate
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from omegaconf import DictConfig, OmegaConf

from jedi.data.brats import build_dataloader
from jedi.models import CrossModalityJEPA
from jedi.training.decoder_module import DecoderTrainingModule
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
    train_loader = build_dataloader(cfg.data.data_dir, "train", tuple(cfg.data.fixed_mapping), cfg.data.batch_size, cfg.data.num_workers, tuple(cfg.data.spatial_size))
    val_data_dir = cfg.data.get("val_data_dir") or cfg.data.data_dir
    val_loader = build_dataloader(val_data_dir, "val", tuple(cfg.data.fixed_mapping), cfg.data.batch_size, cfg.data.num_workers, tuple(cfg.data.spatial_size))
    total_steps = len(train_loader) * cfg.trainer.max_epochs
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
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        callbacks=callbacks,
        logger=CSVLogger("logs", name="decoder_stage"),
    )
    trainer.fit(module, train_loader, val_loader)


if __name__ == "__main__":
    main()
