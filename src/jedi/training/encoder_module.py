import lightning as pl
import torch
import torch.nn.functional as F


class EncoderTrainingModule(pl.LightningModule):
    def __init__(self, model, regularizer, lr, weight_decay, sigreg_weight):
        super().__init__()
        self.model = model
        self.regularizer = regularizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.sigreg_weight = sigreg_weight

    def training_step(self, batch, batch_idx):
        src_output, tgt_output = self.model.encode_src_tgt(batch["src"], batch["tgt"])
        pred_tgt_emb = self.model.predict_tgt(src_output["patch_embeddings"])
        pred_loss = F.mse_loss(pred_tgt_emb, tgt_output["patch_embeddings"].detach())
        sigreg_loss = self.regularizer(tgt_output["patch_embeddings"].transpose(0, 1))
        loss = pred_loss + self.sigreg_weight * sigreg_loss
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
