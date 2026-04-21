import hydra
import lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig

from jedi.data.brats import build_dataloader
from jedi.models import CrossModalityJEPA
from jedi.training.decoder_module import DecoderTrainingModule


@hydra.main(version_base=None, config_path="config", config_name="train_decoder")
def main(cfg: DictConfig):
    pl.seed_everything(cfg.seed)
    encoder = instantiate(cfg.model.encoder)
    projector = instantiate(cfg.model.projector)
    predictor = instantiate(cfg.model.predictor)
    pred_proj = instantiate(cfg.model.pred_proj)
    model = CrossModalityJEPA(encoder=encoder, projector=projector, predictor=predictor, pred_proj=pred_proj)
    decoder = instantiate(cfg.decoder_model.decoder)
    module = DecoderTrainingModule(model=model, decoder=decoder, lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay)
    train_loader = build_dataloader(cfg.data.data_dir, "train", tuple(cfg.data.fixed_mapping), cfg.data.batch_size, cfg.data.num_workers, tuple(cfg.data.spatial_size))
    val_loader = build_dataloader(cfg.data.data_dir, "val", tuple(cfg.data.fixed_mapping), cfg.data.batch_size, cfg.data.num_workers, tuple(cfg.data.spatial_size))
    trainer = pl.Trainer(max_epochs=cfg.trainer.max_epochs, accelerator=cfg.trainer.accelerator, devices=cfg.trainer.devices)
    trainer.fit(module, train_loader, val_loader)


if __name__ == "__main__":
    main()
