import math

import torch
from torch import nn

from jedi.models.transformer import Attention, FeedForward


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class ConditionalBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), causal=True, apply_norm=False)
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class LatentPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, depth=2, heads=8, dim_head=32, mlp_dim=None, dropout=0.0, emb_dropout=0.0, max_patches=8192):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        self.cond_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        self.output_proj = nn.Linear(hidden_dim, input_dim) if hidden_dim != input_dim else nn.Identity()
        self.blocks = nn.ModuleList([
            ConditionalBlock(hidden_dim, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim or hidden_dim * 4, dropout=dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(emb_dropout)
        self.register_buffer("pos_encoding", self._sin_cos_pe(max_patches, hidden_dim))
        self.null_cond = nn.Parameter(torch.randn(1, 1, input_dim) * 0.02)

    @staticmethod
    def _sin_cos_pe(length, dim):
        pe = torch.zeros(length, dim)
        position = torch.arange(length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x, c=None):
        x = self.input_proj(x)
        T = x.size(1)
        if c is None:
            c = self.null_cond.expand(x.size(0), T, -1)
        c = self.cond_proj(c)
        x = x + self.pos_encoding[:, :T]
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x, c)
        return self.output_proj(self.norm(x))
