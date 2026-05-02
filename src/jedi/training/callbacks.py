import warnings

import lightning as pl
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import Callback


class TrainingDynamicsCallback(Callback):
    def __init__(self, log_interval: int = 50):
        super().__init__()
        self.log_interval = log_interval

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % self.log_interval != 0:
            return
        try:
            trainable_params = [p for p in pl_module.parameters() if p.requires_grad]
            grad_norm_sq = 0.0
            param_norm_sq = 0.0
            for param in trainable_params:
                param_norm_sq += param.detach().norm(2).item() ** 2
                if param.grad is not None:
                    grad_norm_sq += param.grad.detach().norm(2).item() ** 2
            grad_norm = grad_norm_sq ** 0.5
            param_norm = param_norm_sq ** 0.5
            pl_module.log("dynamics/grad_norm", grad_norm, on_step=True, sync_dist=True)
            pl_module.log("dynamics/param_norm", param_norm, on_step=True, sync_dist=True)
            pl_module.log("dynamics/grad_to_param_norm", grad_norm / max(param_norm, 1e-12), on_step=True, sync_dist=True)
        except Exception as exc:
            warnings.warn(f"[TrainingDynamicsCallback] {exc}", RuntimeWarning)


class SystemMonitoringCallback(TrainingDynamicsCallback):
    pass


class LossMetricsCallback(Callback):
    def _log_losses(self, pl_module, outputs, prefix, on_step, on_epoch):
        if not isinstance(outputs, dict):
            return
        loss = outputs.get("log_loss", outputs.get("loss"))
        if loss is not None:
            pl_module.log(f"{prefix}/loss", loss, on_step=on_step, on_epoch=on_epoch, sync_dist=True)
        for name in ("pred_loss", "sigreg_loss", "l1_loss", "wavelet_loss"):
            value = outputs.get(name)
            if value is not None:
                pl_module.log(f"{prefix}/{name}", value, on_step=on_step, on_epoch=on_epoch, sync_dist=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._log_losses(pl_module, outputs, "train", on_step=True, on_epoch=True)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self._log_losses(pl_module, outputs, "val", on_step=False, on_epoch=True)


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
            pl_module.log("sigreg/weighted", weight * sigreg_loss, on_step=True, sync_dist=True)
            pl_module.log("sigreg/raw", sigreg_loss, on_step=True, sync_dist=True)
        except Exception as exc:
            warnings.warn(f"[SIGRegMonitor] {exc}", RuntimeWarning)


class LatentEvalMetricsCallback(Callback):
    def __init__(self, log_interval: int = 100):
        super().__init__()
        self.log_interval = log_interval

    def _log_metrics(self, pl_module, outputs, prefix, on_step, on_epoch):
        if not isinstance(outputs, dict) or "pred_emb" not in outputs or "tgt_emb" not in outputs:
            return
        pred_emb = outputs["pred_emb"]
        tgt_emb = outputs["tgt_emb"]
        mse = F.mse_loss(pred_emb, tgt_emb)
        mae = F.l1_loss(pred_emb, tgt_emb)
        cosine = F.cosine_similarity(pred_emb, tgt_emb, dim=-1).mean()
        centered_pred = pred_emb - pred_emb.mean(dim=-1, keepdim=True)
        centered_tgt = tgt_emb - tgt_emb.mean(dim=-1, keepdim=True)
        pearson = F.cosine_similarity(centered_pred, centered_tgt, dim=-1).mean()
        pl_module.log(f"{prefix}/latent_mse", mse, on_step=on_step, on_epoch=on_epoch, sync_dist=True)
        pl_module.log(f"{prefix}/latent_mae", mae, on_step=on_step, on_epoch=on_epoch, sync_dist=True)
        pl_module.log(f"{prefix}/latent_cosine", cosine, on_step=on_step, on_epoch=on_epoch, sync_dist=True)
        pl_module.log(f"{prefix}/latent_pearson", pearson, on_step=on_step, on_epoch=on_epoch, sync_dist=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % self.log_interval != 0:
            return
        try:
            self._log_metrics(pl_module, outputs, "train", on_step=True, on_epoch=False)
        except Exception as exc:
            warnings.warn(f"[LatentEvalMetricsCallback] {exc}", RuntimeWarning)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        try:
            self._log_metrics(pl_module, outputs, "val", on_step=False, on_epoch=True)
        except Exception as exc:
            warnings.warn(f"[LatentEvalMetricsCallback] {exc}", RuntimeWarning)


class ReconstructionEvalMetricsCallback(Callback):
    def __init__(self, data_range: float = 2.0, log_train: bool = False, log_interval: int = 100, ssim_window_size: int = 7):
        super().__init__()
        self.data_range = data_range
        self.log_train = log_train
        self.log_interval = log_interval
        self.ssim_window_size = ssim_window_size

    def _log_metrics(self, pl_module, outputs, prefix, on_step, on_epoch):
        if not isinstance(outputs, dict) or "prediction" not in outputs or "target" not in outputs:
            return
        pred = outputs["prediction"]
        target = outputs["target"]
        mse = F.mse_loss(pred, target)
        mae = F.l1_loss(pred, target)
        rmse = torch.sqrt(mse.clamp_min(1e-12))
        psnr = 20.0 * torch.log10(pred.new_tensor(self.data_range)) - 10.0 * torch.log10(mse.clamp_min(1e-12))
        nrmse = rmse / target.std().clamp_min(1e-12)
        ssim = self._compute_slice_ssim(pred, target)
        pl_module.log(f"{prefix}/mse", mse, on_step=on_step, on_epoch=on_epoch, sync_dist=True)
        pl_module.log(f"{prefix}/mae", mae, on_step=on_step, on_epoch=on_epoch, sync_dist=True)
        pl_module.log(f"{prefix}/rmse", rmse, on_step=on_step, on_epoch=on_epoch, sync_dist=True)
        pl_module.log(f"{prefix}/nrmse", nrmse, on_step=on_step, on_epoch=on_epoch, sync_dist=True)
        pl_module.log(f"{prefix}/psnr", psnr, on_step=on_step, on_epoch=on_epoch, sync_dist=True)
        pl_module.log(f"{prefix}/ssim", ssim, on_step=on_step, on_epoch=on_epoch, sync_dist=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.log_train or batch_idx % self.log_interval != 0:
            return
        try:
            self._log_metrics(pl_module, outputs, "train", on_step=True, on_epoch=False)
        except Exception as exc:
            warnings.warn(f"[ReconstructionEvalMetricsCallback] {exc}", RuntimeWarning)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        try:
            self._log_metrics(pl_module, outputs, "val", on_step=False, on_epoch=True)
        except Exception as exc:
            warnings.warn(f"[ReconstructionEvalMetricsCallback] {exc}", RuntimeWarning)

    def _compute_slice_ssim(self, pred, target):
        mid = pred.shape[2] // 2
        pred_slice = pred[:, 0, mid]
        target_slice = target[:, 0, mid]
        c1 = (0.01 * self.data_range) ** 2
        c2 = (0.03 * self.data_range) ** 2
        pad = self.ssim_window_size // 2
        mu_pred = F.avg_pool2d(pred_slice.unsqueeze(1), self.ssim_window_size, stride=1, padding=pad).squeeze(1)
        mu_target = F.avg_pool2d(target_slice.unsqueeze(1), self.ssim_window_size, stride=1, padding=pad).squeeze(1)
        sigma_pred = F.avg_pool2d(pred_slice.unsqueeze(1) ** 2, self.ssim_window_size, stride=1, padding=pad).squeeze(1) - mu_pred ** 2
        sigma_target = F.avg_pool2d(target_slice.unsqueeze(1) ** 2, self.ssim_window_size, stride=1, padding=pad).squeeze(1) - mu_target ** 2
        sigma_cross = F.avg_pool2d(pred_slice.unsqueeze(1) * target_slice.unsqueeze(1), self.ssim_window_size, stride=1, padding=pad).squeeze(1) - mu_pred * mu_target
        ssim_map = ((2 * mu_pred * mu_target + c1) * (2 * sigma_cross + c2)) / ((mu_pred ** 2 + mu_target ** 2 + c1) * (sigma_pred + sigma_target + c2))
        return ssim_map.mean()
