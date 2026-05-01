import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


def build_adamw_parameter_groups(module: nn.Module, weight_decay: float):
    decay_params = []
    no_decay_params = []
    no_decay_terms = ("bias", "norm", "embedding", "cls_token", "pos_embedding")
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or any(term in name.lower() for term in no_decay_terms):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


def build_adamw(module: nn.Module, lr: float, weight_decay: float):
    return torch.optim.AdamW(
        build_adamw_parameter_groups(module, weight_decay),
        lr=lr,
        betas=(0.9, 0.95),
        eps=1e-8,
    )


def build_warmup_cosine_scheduler(optimizer, lr: float, warmup_steps: int, total_steps: int):
    if warmup_steps <= 0 or total_steps <= 0:
        return None
    warmup = min(warmup_steps, total_steps - 1)
    if warmup <= 0:
        return CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=lr * 0.01)
    return SequentialLR(
        optimizer,
        [
            LinearLR(optimizer, start_factor=1e-4, end_factor=1.0, total_iters=warmup),
            CosineAnnealingLR(optimizer, T_max=total_steps - warmup, eta_min=lr * 0.01),
        ],
        milestones=[warmup],
    )
