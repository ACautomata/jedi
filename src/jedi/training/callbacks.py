import warnings

import lightning as pl
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import Callback


class SystemMonitoringCallback(Callback):
    def __init__(self, log_interval: int = 50):
        super().__init__()
        self.log_interval = log_interval

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % self.log_interval != 0:
            return
        try:
            total_norm = 0.0
            for p in pl_module.parameters(with_callbacks=False):
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5
            pl_module.log("system/grad_norm", total_norm, on_step=True)
            optimizers = trainer.optimizers
            if optimizers:
                pl_module.log("system/learning_rate", optimizers[0].param_groups[0]["lr"], on_step=True)
        except Exception as exc:
            warnings.warn(f"[SystemMonitoringCallback] {exc}", RuntimeWarning)


class EmbeddingStatisticsCallback(Callback):
    def __init__(self, log_interval: int = 100):
        super().__init__()
        self.log_interval = log_interval

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % self.log_interval != 0:
            return
        if not isinstance(outputs, dict) or "patch_embeddings" not in outputs:
            return
        try:
            emb = outputs["patch_embeddings"].detach()
            emb_flat = emb.view(-1, emb.size(-1))
            pl_module.log("embedding/mean", emb_flat.mean().item(), on_step=True)
            pl_module.log("embedding/std", emb_flat.std().item(), on_step=True)
        except Exception as exc:
            warnings.warn(f"[EmbeddingStatisticsCallback] {exc}", RuntimeWarning)


class PredictionQualityCallback(Callback):
    def __init__(self, log_interval: int = 100):
        super().__init__()
        self.log_interval = log_interval

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % self.log_interval != 0:
            return
        if not isinstance(outputs, dict) or "pred_emb" not in outputs:
            return
        try:
            pred_emb = outputs["pred_emb"]
            tgt_emb = outputs["tgt_emb"]
            cos_sim = F.cosine_similarity(pred_emb, tgt_emb, dim=-1)
            pl_module.log("prediction/cosine_sim_mean", cos_sim.mean().item(), on_step=True)
            pl_module.log("prediction/pred_emb_norm", pred_emb.norm(dim=-1).mean().item(), on_step=True)
            pl_module.log("prediction/tgt_emb_norm", tgt_emb.norm(dim=-1).mean().item(), on_step=True)
        except Exception as exc:
            warnings.warn(f"[PredictionQualityCallback] {exc}", RuntimeWarning)


class SIGRegMonitor(Callback):
    def __init__(self, log_interval: int = 50):
        super().__init__()
        self.log_interval = log_interval

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % self.log_interval != 0:
            return
        if not isinstance(outputs, dict) or "sigreg_loss" not in outputs:
            return
        try:
            sigreg_loss = outputs["sigreg_loss"]
            weight = getattr(pl_module, "sigreg_weight", 0)
            pl_module.log("sigreg/weighted", (weight * sigreg_loss).item(), on_step=True)
            pl_module.log("sigreg/raw", sigreg_loss.item(), on_step=True)
        except Exception as exc:
            warnings.warn(f"[SIGRegMonitor] {exc}", RuntimeWarning)
