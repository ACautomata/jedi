import lightning as pl
import torch
import torch.nn.functional as F

from jedi.training.optim import build_adamw, build_warmup_cosine_scheduler


class EncoderTrainingModule(pl.LightningModule):
    def __init__(self, model, regularizer, lr, weight_decay, sigreg_weight, warmup_steps=0, total_steps=0):
        super().__init__()
        self.model = model
        self.regularizer = regularizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.sigreg_weight = sigreg_weight
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def training_step(self, batch, batch_idx):
        src_output, tgt_output = self.model.encode_src_tgt(batch["src"], batch["tgt"])
        tgt_modality_idx = batch.get("tgt_modality_idx", None)
        pred_tgt_emb = self.model.predict_tgt(src_output["patch_embeddings"], tgt_modality=tgt_modality_idx)
        pred_loss = F.mse_loss(pred_tgt_emb, tgt_output["patch_embeddings"].detach())
        sigreg_loss = self.regularizer(tgt_output["patch_embeddings"].transpose(0, 1))
        loss = pred_loss + self.sigreg_weight * sigreg_loss
        return {
            "loss": loss,
            "pred_loss": pred_loss.detach(),
            "sigreg_loss": sigreg_loss.detach(),
            "patch_embeddings": tgt_output["patch_embeddings"].detach(),
            "pred_emb": pred_tgt_emb.detach(),
            "tgt_emb": tgt_output["patch_embeddings"].detach(),
        }

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            src_output, tgt_output = self.model.encode_src_tgt(batch["src"], batch["tgt"])
            tgt_modality_idx = batch.get("tgt_modality_idx", None)
            pred_tgt_emb = self.model.predict_tgt(src_output["patch_embeddings"], tgt_modality=tgt_modality_idx)
            pred_loss = F.mse_loss(pred_tgt_emb, tgt_output["patch_embeddings"].detach())
            sigreg_loss = self.regularizer(tgt_output["patch_embeddings"].transpose(0, 1))
            loss = pred_loss + self.sigreg_weight * sigreg_loss
        return {
            "loss": loss,
            "pred_loss": pred_loss.detach(),
            "sigreg_loss": sigreg_loss.detach(),
            "pred_emb": pred_tgt_emb.detach(),
            "tgt_emb": tgt_output["patch_embeddings"].detach(),
        }

    def configure_optimizers(self):
        optimizer = build_adamw(self, self.lr, self.weight_decay)
        scheduler = build_warmup_cosine_scheduler(optimizer, self.lr, self.warmup_steps, self.total_steps)
        if scheduler is None:
            return optimizer
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
