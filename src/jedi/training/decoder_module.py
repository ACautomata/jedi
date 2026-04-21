import lightning as pl
import torch
import torch.nn.functional as F


class DecoderTrainingModule(pl.LightningModule):
    def __init__(self, model, decoder, lr, weight_decay):
        super().__init__()
        self.model = model
        self.decoder = decoder
        self.lr = lr
        self.weight_decay = weight_decay
        for param in self.model.parameters():
            param.requires_grad = False

    def training_step(self, batch, batch_idx):
        with torch.no_grad():
            src_output, _ = self.model.encode_src_tgt(batch["src"], batch["tgt"])
            pred_tgt_emb = self.model.predict_tgt(src_output["patch_embeddings"])
            grid_size = src_output["grid_size"]
        prediction = self.decoder(pred_tgt_emb, grid_size)
        return F.l1_loss(prediction, batch["tgt"])

    def configure_optimizers(self):
        return torch.optim.AdamW(self.decoder.parameters(), lr=self.lr, weight_decay=self.weight_decay)
