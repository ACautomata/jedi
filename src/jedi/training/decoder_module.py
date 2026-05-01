import lightning as pl
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from jedi.models.wavelet_loss import WaveletLoss


class DecoderTrainingModule(pl.LightningModule):
    def __init__(
        self,
        model,
        decoder,
        lr,
        weight_decay,
        warmup_steps=0,
        total_steps=0,
        use_cls_embedding=False,
        gradient_clip_val=0.0,
        gradient_clip_algorithm="norm",
    ):
        super().__init__()
        self.automatic_optimization = False
        self.model = model
        self.decoder = decoder
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.use_cls_embedding = use_cls_embedding
        self.gradient_clip_val = gradient_clip_val
        self.gradient_clip_algorithm = gradient_clip_algorithm
        self._validate_decoder_config()
        for param in self.model.parameters():
            param.requires_grad = False
        self.wavelet_loss = WaveletLoss()

    def _validate_decoder_config(self):
        using_vis_decoder = hasattr(self.decoder, "cls_proj")
        if self.use_cls_embedding and not using_vis_decoder:
            raise ValueError(
                "use_cls_embedding=True requires VisualizationDecoder "
                f"(got {type(self.decoder).__name__})"
            )
        if not self.use_cls_embedding and using_vis_decoder:
            raise ValueError(
                "use_cls_embedding=False requires VolumeDecoder3D "
                f"(got {type(self.decoder).__name__})"
            )

    def _get_decoder_input(self, src_output, batch):
        if self.use_cls_embedding:
            return src_output["cls_embedding"]
        tgt_modality_idx = batch.get("tgt_modality_idx", None)
        return self.model.predict_tgt(src_output["patch_embeddings"], tgt_modality=tgt_modality_idx)

    def training_step(self, batch, batch_idx):
        with torch.no_grad():
            src_output, _ = self.model.encode_src_tgt(batch["src"], batch["tgt"])
            decoder_input = self._get_decoder_input(src_output, batch)
            grid_size = src_output["grid_size"]
        prediction = self.decoder(decoder_input, grid_size)
        target = batch["tgt"]
        l1_loss = F.l1_loss(prediction, target)
        wl = self.wavelet_loss(prediction, target)
        loss = (l1_loss + wl) / 2
        step_output = {
            "loss": loss.detach(),
            "l1_loss": l1_loss.detach(),
            "wavelet_loss": wl.detach(),
            "prediction": prediction.detach(),
            "target": target.detach(),
        }

        try:
            optimizer = self.optimizers()
        except RuntimeError as error:
            if "is not attached to a `Trainer`" not in str(error):
                raise
            return step_output
        optimizer.zero_grad()
        self._pc_backward([l1_loss, wl])
        self._clip_gradients(optimizer)
        optimizer.step()
        self._step_scheduler()
        return step_output

    def _pc_backward(self, objectives):
        params = [param for param in self.decoder.parameters() if param.requires_grad]
        if not params:
            return
        grads = []
        masks = []
        for idx, objective in enumerate(objectives):
            for param in params:
                param.grad = None
            retain_graph = idx < len(objectives) - 1
            try:
                self.manual_backward(objective, retain_graph=retain_graph)
            except RuntimeError as error:
                if "is not attached to a `Trainer`" not in str(error):
                    raise
                objective.backward(retain_graph=retain_graph)
            flat_grads = []
            flat_masks = []
            for param in params:
                if param.grad is None:
                    flat_grads.append(torch.zeros_like(param).reshape(-1))
                    flat_masks.append(torch.zeros(param.numel(), dtype=torch.bool, device=param.device))
                else:
                    flat_grads.append(param.grad.detach().clone().reshape(-1))
                    flat_masks.append(torch.ones(param.numel(), dtype=torch.bool, device=param.device))
            grads.append(torch.cat(flat_grads))
            masks.append(torch.cat(flat_masks))

        projected = [grad.clone() for grad in grads]
        for grad in projected:
            for other_grad in grads:
                dot = torch.dot(grad, other_grad)
                if dot < 0:
                    grad -= dot * other_grad / other_grad.pow(2).sum().clamp_min(1e-12)

        stacked_grads = torch.stack(projected)
        stacked_masks = torch.stack(masks)
        shared = stacked_masks.all(dim=0)
        merged = torch.zeros_like(stacked_grads[0])
        if shared.any():
            merged[shared] = stacked_grads[:, shared].mean(dim=0)
        if (~shared).any():
            merged[~shared] = stacked_grads[:, ~shared].sum(dim=0)

        offset = 0
        for param in params:
            numel = param.numel()
            param.grad = merged[offset:offset + numel].view_as(param).clone()
            offset += numel

    def _clip_gradients(self, optimizer):
        if self.gradient_clip_val <= 0:
            return
        self.clip_gradients(
            optimizer,
            gradient_clip_val=self.gradient_clip_val,
            gradient_clip_algorithm=self.gradient_clip_algorithm,
        )

    def _step_scheduler(self):
        scheduler = self.lr_schedulers()
        if scheduler is None:
            return
        if isinstance(scheduler, list):
            for item in scheduler:
                item.step()
            return
        scheduler.step()

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            src_output, _ = self.model.encode_src_tgt(batch["src"], batch["tgt"])
            decoder_input = self._get_decoder_input(src_output, batch)
            grid_size = src_output["grid_size"]
            prediction = self.decoder(decoder_input, grid_size)
            target = batch["tgt"]
            l1_loss = F.l1_loss(prediction, target)
            wl = self.wavelet_loss(prediction, target)
            loss = (l1_loss + wl) / 2
        return {
            "loss": loss,
            "l1_loss": l1_loss.detach(),
            "wavelet_loss": wl.detach(),
            "prediction": prediction.detach(),
            "target": target.detach(),
        }

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
