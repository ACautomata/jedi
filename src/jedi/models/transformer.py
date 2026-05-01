import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

try:
    from flash_attn_interface import flash_attn_func as _fa3_func

    _HAS_FA3 = True
except ImportError:
    _HAS_FA3 = False


def _flash_attn(q, k, v, causal=False):
    """FlashAttention-3 backend; returns None if not applicable."""
    if not (_HAS_FA3 and q.is_cuda and q.dtype in (torch.float16, torch.bfloat16)):
        return None
    out, _ = _fa3_func(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        causal=causal,
    )
    return out.transpose(1, 2)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x, causal=False, apply_norm=True):
        if apply_norm:
            x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv)
        dropout_p = self.dropout if self.training else 0.0
        out = _flash_attn(q, k, v, causal=causal)
        if out is None:
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=causal)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


class Block(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
