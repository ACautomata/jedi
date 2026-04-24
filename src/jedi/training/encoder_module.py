import lightning as pl
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


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
        self.log("train/pred_loss", pred_loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/sigreg_loss", sigreg_loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/loss", loss, on_step=True, on_epoch=True, sync_dist=True)
        return {
            "loss": loss,
            "patch_embeddings": tgt_output["patch_embeddings"].detach(),
            "pred_emb": pred_tgt_emb.detach(),
            "tgt_emb": tgt_output["patch_embeddings"].detach(),
            "sigreg_loss": sigreg_loss.detach(),
        }

    def validation_step(self, batch, batch_idx):
        src_output, tgt_output = self.model.encode_src_tgt(batch["src"], batch["tgt"])
        tgt_modality_idx = batch.get("tgt_modality_idx", None)
        pred_tgt_emb = self.model.predict_tgt(src_output["patch_embeddings"], tgt_modality=tgt_modality_idx)
        pred_loss = F.mse_loss(pred_tgt_emb, tgt_output["patch_embeddings"].detach())
        sigreg_loss = self.regularizer(tgt_output["patch_embeddings"].transpose(0, 1))
        loss = pred_loss + self.sigreg_weight * sigreg_loss
        self.log("val/pred_loss", pred_loss, on_epoch=True, sync_dist=True)
        self.log("val/sigreg_loss", sigreg_loss, on_epoch=True, sync_dist=True)
        self.log("val/loss", loss, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.warmup_steps <= 0 or self.total_steps <= 0:
            return optimizer
        warmup = min(self.warmup_steps, self.total_steps - 1)
        scheduler = SequentialLR(optimizer, [
            LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup),
            CosineAnnealingLR(optimizer, T_max=self.total_steps - warmup, eta_min=self.lr * 0.01),
        ], milestones=[warmup])
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
