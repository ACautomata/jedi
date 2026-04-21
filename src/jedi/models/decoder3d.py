from einops import rearrange
from torch import nn


class VolumeDecoder3D(nn.Module):
    def __init__(self, embed_dim, patch_size, out_channels, hidden_channels):
        super().__init__()
        self.patch_size = patch_size
        patch_volume = patch_size[0] * patch_size[1] * patch_size[2]
        self.to_voxels = nn.Linear(embed_dim, out_channels * patch_volume)
        self.refine = nn.Sequential(
            nn.Conv3d(out_channels, hidden_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(hidden_channels),
            nn.GELU(),
            nn.Conv3d(hidden_channels, out_channels, kernel_size=1),
            nn.Tanh(),
        )

    def forward(self, patch_embeddings, grid_size):
        patches = self.to_voxels(patch_embeddings)
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
        return self.refine(volume)
