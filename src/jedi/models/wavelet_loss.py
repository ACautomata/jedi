import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward


class WaveletLoss(nn.Module):
    """Wavelet-domain L1 loss applied per-slice on 3D volumes.

    Decomposes pred and target into lowpass (Yl) and highpass (Yh)
    subbands via 2D DWT, then computes weighted L1 in the wavelet domain.
    """

    def __init__(self, wave: str = "db4", J: int = 2, alpha_low: float = 0.3, alpha_high: float = 0.7):
        super().__init__()
        self.dwt = DWTForward(J=J, wave=wave, mode="symmetric")
        self.J = J
        self.alpha_low = alpha_low
        self.alpha_high = alpha_high

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert pred.ndim == 5, f"Expected 5D tensor (B,C,D,H,W), got shape {pred.shape}"
        B, C, D = pred.shape[:3]
        H, W = pred.shape[3], pred.shape[4]
        pred_2d = pred.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)
        tgt_2d = target.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)

        # Align DWT filter dtype to input for AMP compatibility
        dwt_dtype = self.dwt.h0_col.dtype
        if pred_2d.dtype != dwt_dtype:
            pred_2d = pred_2d.to(dwt_dtype)
            tgt_2d = tgt_2d.to(dwt_dtype)

        Yl_p, Yh_p = self.dwt(pred_2d)
        Yl_t, Yh_t = self.dwt(tgt_2d)

        loss_ll = F.l1_loss(Yl_p, Yl_t)
        loss_hf = sum(F.l1_loss(hp, ht) for hp, ht in zip(Yh_p, Yh_t))
        # /J normalizes the highpass term to mean across decomposition levels
        return self.alpha_low * loss_ll + (self.alpha_high / self.J) * loss_hf
