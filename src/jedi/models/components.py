from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=None, norm_fn=nn.LayerNorm, act_fn=nn.GELU):
        super().__init__()
        if isinstance(norm_fn, str):
            norm_fn = getattr(nn, norm_fn)
        if isinstance(act_fn, str):
            act_fn = getattr(nn, act_fn)
        norm_layer = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm_layer,
            act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x):
        return self.net(x)


class ModalityEmbedder(nn.Module):
    def __init__(self, num_modalities=4, emb_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(num_modalities, emb_dim)
        self.proj = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, modality_indices):
        x = self.embedding(modality_indices)
        x = self.proj(x)
        return x.unsqueeze(1)
