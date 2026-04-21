import math

import torch
from einops import rearrange
from torch import nn

from jedi.models.transformer import Block


class ViT3DEncoder(nn.Module):
    def __init__(
        self,
        image_size,
        patch_size,
        in_channels,
        embed_dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = tuple(i // p for i, p in zip(image_size, patch_size))
        self.num_patches = math.prod(self.grid_size)
        self.patch_embed = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embedding, std=0.02)
        self.blocks = nn.ModuleList(
            [Block(embed_dim, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim, dropout=dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.hidden_size = embed_dim

    def forward(self, x):
        x = self.patch_embed(x)
        grid_size = x.shape[-3:]
        x = rearrange(x, "b c d h w -> b (d h w) c")
        cls_token = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embedding[:, : x.size(1)]
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return {
            "last_hidden_state": x,
            "cls_embedding": x[:, 0],
            "patch_embeddings": x[:, 1:],
            "grid_size": grid_size,
        }
