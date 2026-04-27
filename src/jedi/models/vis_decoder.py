from math import prod

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from jedi.models.transformer import FeedForward


class CrossAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x, context, apply_norm=True):
        if apply_norm:
            x = self.norm(x)
        q = rearrange(self.to_q(x), "b n (h d) -> b h n d", h=self.heads)
        k = rearrange(self.to_k(context), "b n (h d) -> b h n d", h=self.heads)
        v = rearrange(self.to_v(context), "b n (h d) -> b h n d", h=self.heads)
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class CrossAttnBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.attn = CrossAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, context):
        x = x + self.attn(self.norm1(x), self.norm1(context), apply_norm=False)
        x = x + self.mlp(self.norm2(x))
        return x


class VisualizationDecoder(nn.Module):
    """Diagnostic decoder that reconstructs a 3D volume from the [CLS] token.

    Designed as a diagnostic tool to visualize what spatial information is
    retained in the [CLS] representation. Uses learnable query tokens (one per
    patch position) that cross-attend to a CLS-derived K/V token through
    multiple cross-attention layers with residual MLP blocks.

    Unlike VolumeDecoder3D, this decoder has no spatial refinement CNN — the
    raw reconstruction from [CLS] is the diagnostic signal, not a post-processed
    version.
    """
    def __init__(
        self,
        cls_dim=256,
        hidden_dim=512,
        image_size=(128, 160, 192),
        patch_size=(8, 8, 8),
        out_channels=1,
        depth=4,
        heads=8,
        dim_head=64,
        mlp_dim=None,
        dropout=0.0,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.out_channels = out_channels
        if mlp_dim is None:
            mlp_dim = hidden_dim * 4

        grid_size = tuple(s // p for s, p in zip(image_size, patch_size))
        num_queries = prod(grid_size)
        patch_volume = prod(patch_size)

        self.cls_proj = nn.Linear(cls_dim, hidden_dim)
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, hidden_dim) * 0.02)
        self.blocks = nn.ModuleList([
            CrossAttnBlock(hidden_dim, heads, dim_head, mlp_dim, dropout=dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.to_voxels = nn.Linear(hidden_dim, out_channels * patch_volume)

    def forward(self, cls_embedding, grid_size):
        B = cls_embedding.shape[0]
        num_queries = self.query_tokens.size(1)
        if prod(grid_size) != num_queries:
            raise ValueError(
                f"grid_size {grid_size} (product={prod(grid_size)}) does not match "
                f"query_tokens count ({num_queries}). "
                f"Expected grid_size matching image_size // patch_size = {self.image_size} // {self.patch_size}."
            )
        cls_hidden = self.cls_proj(cls_embedding).unsqueeze(1)
        queries = self.query_tokens.expand(B, -1, -1)
        x = queries
        for block in self.blocks:
            x = block(x, cls_hidden)
        x = self.norm(x)
        patches = self.to_voxels(x)
        volume = rearrange(
            patches,
            "b (gd gh gw) (c pd ph pw) -> b c (gd pd) (gh ph) (gw pw)",
            gd=grid_size[0],
            gh=grid_size[1],
            gw=grid_size[2],
            pd=self.patch_size[0],
            ph=self.patch_size[1],
            pw=self.patch_size[2],
        )
        return torch.tanh(volume)
