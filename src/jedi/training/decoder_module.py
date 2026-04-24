import lightning as pl
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


class DecoderTrainingModule(pl.LightningModule):
    def __init__(self, model, decoder, lr, weight_decay, warmup_steps=0, total_steps=0):
        super().__init__()
        self.model = model
        self.decoder = decoder
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        for param in self.model.parameters():
            param.requires_grad = False

    def training_step(self, batch, batch_idx):
        with torch.no_grad():
            src_output, _ = self.model.encode_src_tgt(batch["src"], batch["tgt"])
            tgt_modality_idx = batch.get("tgt_modality_idx", None)
            pred_tgt_emb = self.model.predict_tgt(src_output["patch_embeddings"], tgt_modality=tgt_modality_idx)
            grid_size = src_output["grid_size"]
        prediction = self.decoder(pred_tgt_emb, grid_size)
        l1_loss = F.l1_loss(prediction, batch["tgt"])
        self.log("train/l1_loss", l1_loss, on_step=True, on_epoch=True, sync_dist=True)
        return l1_loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            src_output, _ = self.model.encode_src_tgt(batch["src"], batch["tgt"])
            tgt_modality_idx = batch.get("tgt_modality_idx", None)
            pred_tgt_emb = self.model.predict_tgt(src_output["patch_embeddings"], tgt_modality=tgt_modality_idx)
            grid_size = src_output["grid_size"]
        prediction = self.decoder(pred_tgt_emb, grid_size)
        target = batch["tgt"]
        l1_loss = F.l1_loss(prediction, target)
        self.log("val/l1_loss", l1_loss, on_epoch=True, sync_dist=True)
        mse = F.mse_loss(prediction, target)
        psnr = 10.0 * torch.log10(torch.tensor(4.0) / (mse + 1e-10))
        self.log("val/psnr", psnr, on_epoch=True, sync_dist=True)
        ssim_val = self._compute_ssim(prediction, target)
        self.log("val/ssim", ssim_val, on_epoch=True, sync_dist=True)
        self.log("val/loss", l1_loss, on_epoch=True, sync_dist=True)
        return l1_loss

    @staticmethod
    def _compute_ssim(pred, target, window_size=7):
        mid = pred.shape[2] // 2
        pred_slice = pred[:, 0, mid]
        tgt_slice = target[:, 0, mid]
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        pad = window_size // 2
        mu_p = F.avg_pool2d(pred_slice.unsqueeze(1), window_size, stride=1, padding=pad).squeeze(1)
        mu_t = F.avg_pool2d(tgt_slice.unsqueeze(1), window_size, stride=1, padding=pad).squeeze(1)
        sigma_p = F.avg_pool2d(pred_slice.unsqueeze(1) ** 2, window_size, stride=1, padding=pad).squeeze(1) - mu_p ** 2
        sigma_t = F.avg_pool2d(tgt_slice.unsqueeze(1) ** 2, window_size, stride=1, padding=pad).squeeze(1) - mu_t ** 2
        sigma_pt = F.avg_pool2d(pred_slice.unsqueeze(1) * tgt_slice.unsqueeze(1), window_size, stride=1, padding=pad).squeeze(1) - mu_p * mu_t
        ssim_map = ((2 * mu_p * mu_t + C1) * (2 * sigma_pt + C2)) / ((mu_p ** 2 + mu_t ** 2 + C1) * (sigma_p + sigma_t + C2))
        return ssim_map.mean()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.decoder.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.warmup_steps <= 0 or self.total_steps <= 0:
            return optimizer
        warmup = min(self.warmup_steps, self.total_steps - 1)
        scheduler = SequentialLR(optimizer, [
            LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup),
            CosineAnnealingLR(optimizer, T_max=self.total_steps - warmup, eta_min=self.lr * 0.01),
        ], milestones=[warmup])
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
