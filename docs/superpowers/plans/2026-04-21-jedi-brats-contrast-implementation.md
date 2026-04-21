# Jedi BraTS2023 Contrast Conversion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a `src`-layout Python package named `jedi` that implements BraTS2023 cross-modality contrast conversion with a `le-wm`-aligned two-stage pipeline: cross-modality JEPA latent prediction, then frozen-latent volume decoding.

**Architecture:** The implementation mirrors `le-wm` responsibilities while swapping temporal prediction for modality prediction. Stage 1 trains a shared 3D ViT encoder, projector, predictor, and pred-proj so that `predictor(projector(encoder(src)))` aligns with `projector(encoder(tgt))`; Stage 2 freezes that stack and trains only a 3D decoder from predicted target latents to target volumes.

**Tech Stack:** Python 3.10+, PyTorch, Lightning, MONAI, Hydra, einops, unittest/pytest-style shape tests

---

## File map

### Create
- `pyproject.toml` — package metadata and runtime dependencies for `jedi`
- `README.md` — setup, BraTS folder layout, stage-1/stage-2 commands, checkpoint usage
- `src/jedi/__init__.py` — package version export
- `src/jedi/config/train_encoder.yaml` — Hydra entry config for stage 1
- `src/jedi/config/train_decoder.yaml` — Hydra entry config for stage 2
- `src/jedi/config/data/brats2023.yaml` — dataset root, target size, train random mapping flag, fixed eval mapping
- `src/jedi/config/model/encoder.yaml` — 3D ViT + projector + predictor + pred-proj settings
- `src/jedi/config/model/decoder.yaml` — decoder settings and stage-2 checkpoint loading settings
- `src/jedi/data/brats.py` — BraTS file discovery, random training mapping, fixed eval mapping, dataloader factory
- `src/jedi/data/transforms.py` — MONAI transforms for orientation, spacing, padding, cropping, normalization
- `src/jedi/models/transformer.py` — reusable transformer blocks copied from `le-wm` and kept dimension-agnostic
- `src/jedi/models/components.py` — MLP projector/pred-proj blocks
- `src/jedi/models/regularizers.py` — SIGReg copied from `le-wm`
- `src/jedi/models/vit3d.py` — 3D ViT encoder with patch embedding and positional embeddings
- `src/jedi/models/predictor.py` — latent predictor used in stage 1 and stage 2 frozen path
- `src/jedi/models/jepa.py` — cross-modality JEPA model wrapper exposing `encode_src_tgt()` and `predict_tgt()`
- `src/jedi/models/decoder3d.py` — patch-token-to-volume decoder with `Tanh` output
- `src/jedi/models/__init__.py` — exported model classes
- `src/jedi/training/encoder_module.py` — LightningModule for stage 1 latent alignment training
- `src/jedi/training/decoder_module.py` — LightningModule for stage 2 frozen-stack decoding
- `src/jedi/training/__init__.py` — training exports
- `src/jedi/train_encoder.py` — Hydra/Lightning stage-1 entrypoint
- `src/jedi/train_decoder.py` — Hydra/Lightning stage-2 entrypoint
- `src/jedi/infer.py` — inference entrypoint loading encoder-side checkpoint and decoder checkpoint
- `tests/test_brats_dataset.py` — dataset and mapping behavior tests
- `tests/test_transforms.py` — padding and normalization tests
- `tests/test_vit3d_shapes.py` — encoder token shape tests
- `tests/test_encoder_step.py` — stage-1 forward/loss smoke test
- `tests/test_decoder_step.py` — stage-2 frozen-path smoke test

### Delete or replace
- Delete generated prototype directory `brats-contrast-model/` after the new `src/jedi` implementation is in place and verified

## Dependency order
1. Package skeleton and dependency metadata
2. Data transforms and dataset mapping behavior
3. Shared model primitives (`transformer`, `components`, `regularizers`)
4. 3D ViT, predictor, JEPA wrapper, decoder
5. Stage-1 Lightning module and config
6. Stage-2 Lightning module and config
7. Entrypoints and inference
8. README and cleanup

## Key risks
- Random training modality mapping can accidentally leak into validation/test if dataset mode boundaries are weak
- Full-volume 3D tensors can create shape mismatches if padding/cropping order is inconsistent across `src` and `tgt`
- Stage 2 can silently update encoder-side parameters if freezing is incomplete
- Decoder input must be `pred_tgt_emb`, not `src_emb`; naming mistakes here would break the intended `le-wm` alignment

### Task 1: Create the package skeleton and dependency metadata

**Files:**
- Create: `pyproject.toml`
- Create: `src/jedi/__init__.py`
- Test: `python -c "import jedi; print(jedi.__version__)"`

- [ ] **Step 1: Write the failing import check**

Run:
```bash
PYTHONPATH=/Users/junran/Documents/jedi/src python -c "import jedi; print(jedi.__version__)"
```

Expected: `ModuleNotFoundError: No module named 'jedi'`

- [ ] **Step 2: Write minimal package metadata**

Create `pyproject.toml`:
```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "jedi"
version = "0.1.0"
description = "BraTS2023 cross-modality contrast conversion with a le-wm-aligned pipeline"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
  "einops",
  "hydra-core",
  "lightning",
  "monai[nibabel]",
  "numpy",
  "omegaconf",
  "torch",
  "torchvision",
]

[project.scripts]
jedi-train-encoder = "jedi.train_encoder:main"
jedi-train-decoder = "jedi.train_decoder:main"
jedi-infer = "jedi.infer:main"

[tool.hatch.build.targets.wheel]
packages = ["src/jedi"]
```

Create `src/jedi/__init__.py`:
```python
__version__ = "0.1.0"
```

- [ ] **Step 3: Run the import check to verify it passes**

Run:
```bash
PYTHONPATH=/Users/junran/Documents/jedi/src python -c "import jedi; print(jedi.__version__)"
```

Expected: output `0.1.0`

- [ ] **Step 4: Commit**

Run:
```bash
git add pyproject.toml src/jedi/__init__.py
git commit -m "feat: initialize jedi package"
```

### Task 2: Build MONAI transforms for fixed-size padded volumes

**Files:**
- Create: `src/jedi/data/transforms.py`
- Create: `tests/test_transforms.py`
- Test: `tests/test_transforms.py`

- [ ] **Step 1: Write the failing transform tests**

Create `tests/test_transforms.py`:
```python
import unittest

import numpy as np

from jedi.data.transforms import normalize_to_unit_range, pad_or_crop_volume


class TestTransforms(unittest.TestCase):
    def test_normalize_to_unit_range_bounds(self):
        volume = np.array([0.0, 1500.0, 3000.0], dtype=np.float32)
        normalized = normalize_to_unit_range(volume, a_min=0.0, a_max=3000.0)
        self.assertAlmostEqual(float(normalized.min()), -1.0)
        self.assertAlmostEqual(float(normalized.max()), 1.0)

    def test_pad_or_crop_volume_returns_fixed_shape(self):
        volume = np.zeros((120, 150, 160), dtype=np.float32)
        output = pad_or_crop_volume(volume, spatial_size=(128, 160, 192))
        self.assertEqual(output.shape, (128, 160, 192))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
PYTHONPATH=/Users/junran/Documents/jedi/src python -m pytest tests/test_transforms.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'jedi.data'`

- [ ] **Step 3: Write minimal transform implementation**

Create `src/jedi/data/transforms.py`:
```python
from __future__ import annotations

import numpy as np
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    SpatialPadd,
)


def normalize_to_unit_range(volume: np.ndarray, a_min: float, a_max: float) -> np.ndarray:
    clipped = np.clip(volume, a_min, a_max)
    scaled = (clipped - a_min) / (a_max - a_min)
    return scaled * 2.0 - 1.0


def pad_or_crop_volume(volume: np.ndarray, spatial_size: tuple[int, int, int]) -> np.ndarray:
    output = np.zeros(spatial_size, dtype=volume.dtype)
    src_slices = []
    dst_slices = []
    for current, target in zip(volume.shape, spatial_size):
        copy_size = min(current, target)
        src_start = max((current - copy_size) // 2, 0)
        dst_start = max((target - copy_size) // 2, 0)
        src_slices.append(slice(src_start, src_start + copy_size))
        dst_slices.append(slice(dst_start, dst_start + copy_size))
    output[tuple(dst_slices)] = volume[tuple(src_slices)]
    return output


def build_pair_transforms(spatial_size: tuple[int, int, int], a_min: float, a_max: float):
    return Compose(
        [
            LoadImaged(keys=["src", "tgt"]),
            EnsureChannelFirstd(keys=["src", "tgt"]),
            Orientationd(keys=["src", "tgt"], axcodes="RAS"),
            SpatialPadd(keys=["src", "tgt"], spatial_size=spatial_size),
            ScaleIntensityRanged(
                keys=["src", "tgt"],
                a_min=a_min,
                a_max=a_max,
                b_min=-1.0,
                b_max=1.0,
                clip=True,
            ),
            EnsureTyped(keys=["src", "tgt"]),
        ]
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
PYTHONPATH=/Users/junran/Documents/jedi/src python -m pytest tests/test_transforms.py -v
```

Expected: `2 passed`

- [ ] **Step 5: Commit**

Run:
```bash
git add src/jedi/data/transforms.py tests/test_transforms.py
git commit -m "feat: add fixed-size volume transforms"
```

### Task 3: Build BraTS dataset pairing with random train mapping and fixed eval mapping

**Files:**
- Create: `src/jedi/data/brats.py`
- Create: `tests/test_brats_dataset.py`
- Test: `tests/test_brats_dataset.py`

- [ ] **Step 1: Write the failing dataset tests**

Create `tests/test_brats_dataset.py`:
```python
import tempfile
import unittest
from pathlib import Path

from jedi.data.brats import BraTSContrastDataset


class TestBraTSDataset(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        case_dir = Path(self.tmpdir.name) / "BraTS-GLI-00001-000"
        case_dir.mkdir()
        for suffix in ["t1n", "t1c", "t2w", "t2f"]:
            (case_dir / f"BraTS-GLI-00001-000-{suffix}.nii.gz").touch()

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_eval_mode_uses_fixed_mapping(self):
        dataset = BraTSContrastDataset(
            data_dir=self.tmpdir.name,
            mode="val",
            fixed_mapping=("t1n", "t2w"),
        )
        sample = dataset[0]
        self.assertEqual(sample["src_modality"], "t1n")
        self.assertEqual(sample["tgt_modality"], "t2w")

    def test_train_mode_never_returns_identical_modalities(self):
        dataset = BraTSContrastDataset(
            data_dir=self.tmpdir.name,
            mode="train",
            fixed_mapping=("t1n", "t2w"),
        )
        for _ in range(20):
            sample = dataset[0]
            self.assertNotEqual(sample["src_modality"], sample["tgt_modality"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
PYTHONPATH=/Users/junran/Documents/jedi/src python -m pytest tests/test_brats_dataset.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'jedi.data.brats'`

- [ ] **Step 3: Write minimal dataset implementation**

Create `src/jedi/data/brats.py`:
```python
from __future__ import annotations

import random
from pathlib import Path

from monai.data import DataLoader, Dataset

from jedi.data.transforms import build_pair_transforms


class BraTSContrastDataset(Dataset):
    modalities = ("t1n", "t1c", "t2w", "t2f")

    def __init__(
        self,
        data_dir: str,
        mode: str,
        fixed_mapping: tuple[str, str],
        spatial_size: tuple[int, int, int] = (128, 160, 192),
        a_min: float = 0.0,
        a_max: float = 3000.0,
    ):
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.fixed_mapping = fixed_mapping
        self.items = sorted([path for path in self.data_dir.iterdir() if path.is_dir()])
        self.transform = build_pair_transforms(spatial_size=spatial_size, a_min=a_min, a_max=a_max)

    def __len__(self) -> int:
        return len(self.items)

    def _sample_mapping(self) -> tuple[str, str]:
        if self.mode != "train":
            return self.fixed_mapping
        src, tgt = random.sample(self.modalities, k=2)
        return src, tgt

    def __getitem__(self, index: int):
        case_dir = self.items[index]
        src_modality, tgt_modality = self._sample_mapping()
        case_id = case_dir.name
        sample = {
            "src": str(case_dir / f"{case_id}-{src_modality}.nii.gz"),
            "tgt": str(case_dir / f"{case_id}-{tgt_modality}.nii.gz"),
            "case_id": case_id,
            "src_modality": src_modality,
            "tgt_modality": tgt_modality,
        }
        return sample


def build_dataloader(
    data_dir: str,
    mode: str,
    fixed_mapping: tuple[str, str],
    batch_size: int,
    num_workers: int,
    spatial_size: tuple[int, int, int],
):
    dataset = BraTSContrastDataset(
        data_dir=data_dir,
        mode=mode,
        fixed_mapping=fixed_mapping,
        spatial_size=spatial_size,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=mode == "train", num_workers=num_workers)
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
PYTHONPATH=/Users/junran/Documents/jedi/src python -m pytest tests/test_brats_dataset.py -v
```

Expected: `2 passed`

- [ ] **Step 5: Commit**

Run:
```bash
git add src/jedi/data/brats.py tests/test_brats_dataset.py
git commit -m "feat: add BraTS modality pairing dataset"
```

### Task 4: Copy shared model primitives from le-wm

**Files:**
- Create: `src/jedi/models/transformer.py`
- Create: `src/jedi/models/components.py`
- Create: `src/jedi/models/regularizers.py`
- Test: `PYTHONPATH=/Users/junran/Documents/jedi/src python -c "from jedi.models.regularizers import SIGReg; print(SIGReg.__name__)"`

- [ ] **Step 1: Write the failing import check**

Run:
```bash
PYTHONPATH=/Users/junran/Documents/jedi/src python -c "from jedi.models.regularizers import SIGReg; print(SIGReg.__name__)"
```

Expected: FAIL with `ModuleNotFoundError: No module named 'jedi.models'`

- [ ] **Step 2: Copy minimal shared primitives**

Create `src/jedi/models/transformer.py`:
```python
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


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

    def forward(self, x, causal=False):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=causal,
        )
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
```

Create `src/jedi/models/components.py`:
```python
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
```

Create `src/jedi/models/regularizers.py`:
```python
import torch


class SIGReg(torch.nn.Module):
    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        a = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        a = a.div_(a.norm(p=2, dim=0))
        x_t = (proj @ a).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()
```

- [ ] **Step 3: Run the import check to verify it passes**

Run:
```bash
PYTHONPATH=/Users/junran/Documents/jedi/src python -c "from jedi.models.regularizers import SIGReg; print(SIGReg.__name__)"
```

Expected: output `SIGReg`

- [ ] **Step 4: Commit**

Run:
```bash
git add src/jedi/models/transformer.py src/jedi/models/components.py src/jedi/models/regularizers.py
git commit -m "feat: add shared latent model primitives"
```

### Task 5: Build the 3D ViT encoder and verify token shapes

**Files:**
- Create: `src/jedi/models/vit3d.py`
- Create: `tests/test_vit3d_shapes.py`
- Test: `tests/test_vit3d_shapes.py`

- [ ] **Step 1: Write the failing shape test**

Create `tests/test_vit3d_shapes.py`:
```python
import unittest

import torch

from jedi.models.vit3d import ViT3DEncoder


class TestViT3DShapes(unittest.TestCase):
    def test_patch_token_shape(self):
        model = ViT3DEncoder(
            image_size=(32, 32, 32),
            patch_size=(8, 8, 8),
            in_channels=1,
            embed_dim=64,
            depth=2,
            heads=4,
            dim_head=16,
            mlp_dim=128,
        )
        x = torch.randn(2, 1, 32, 32, 32)
        output = model(x)
        self.assertEqual(output["patch_embeddings"].shape, (2, 64, 64))
        self.assertEqual(output["cls_embedding"].shape, (2, 64))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
PYTHONPATH=/Users/junran/Documents/jedi/src python -m pytest tests/test_vit3d_shapes.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'jedi.models.vit3d'`

- [ ] **Step 3: Write minimal 3D ViT implementation**

Create `src/jedi/models/vit3d.py`:
```python
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
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))
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
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
PYTHONPATH=/Users/junran/Documents/jedi/src python -m pytest tests/test_vit3d_shapes.py -v
```

Expected: `1 passed`

- [ ] **Step 5: Commit**

Run:
```bash
git add src/jedi/models/vit3d.py tests/test_vit3d_shapes.py
git commit -m "feat: add 3d vit encoder"
```

### Task 6: Build predictor and cross-modality JEPA wrapper

**Files:**
- Create: `src/jedi/models/predictor.py`
- Create: `src/jedi/models/jepa.py`
- Create: `src/jedi/models/__init__.py`
- Test: `tests/test_encoder_step.py`

- [ ] **Step 1: Write the failing stage-1 smoke test**

Create `tests/test_encoder_step.py`:
```python
import unittest

import torch

from jedi.models import CrossModalityJEPA, MLP, ViT3DEncoder
from jedi.models.predictor import LatentPredictor
from jedi.models.regularizers import SIGReg
from jedi.training.encoder_module import EncoderTrainingModule


class TestEncoderStep(unittest.TestCase):
    def test_stage1_step_returns_finite_loss(self):
        encoder = ViT3DEncoder(
            image_size=(32, 32, 32),
            patch_size=(8, 8, 8),
            in_channels=1,
            embed_dim=64,
            depth=2,
            heads=4,
            dim_head=16,
            mlp_dim=128,
        )
        projector = MLP(input_dim=64, hidden_dim=128, output_dim=64, norm_fn="LayerNorm")
        predictor = LatentPredictor(input_dim=64, hidden_dim=64)
        pred_proj = MLP(input_dim=64, hidden_dim=128, output_dim=64, norm_fn="LayerNorm")
        model = CrossModalityJEPA(encoder=encoder, projector=projector, predictor=predictor, pred_proj=pred_proj)
        module = EncoderTrainingModule(model=model, regularizer=SIGReg(num_proj=32), lr=1e-4, weight_decay=1e-4, sigreg_weight=0.01)
        batch = {
            "src": torch.randn(2, 1, 32, 32, 32),
            "tgt": torch.randn(2, 1, 32, 32, 32),
        }
        loss = module.training_step(batch, 0)
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
PYTHONPATH=/Users/junran/Documents/jedi/src python -m pytest tests/test_encoder_step.py -v
```

Expected: FAIL with import errors for missing model/training modules

- [ ] **Step 3: Write minimal predictor and JEPA wrapper**

Create `src/jedi/models/predictor.py`:
```python
from torch import nn


class LatentPredictor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        return self.net(x)
```

Create `src/jedi/models/jepa.py`:
```python
from torch import nn


class CrossModalityJEPA(nn.Module):
    def __init__(self, encoder, projector, predictor, pred_proj):
        super().__init__()
        self.encoder = encoder
        self.projector = projector
        self.predictor = predictor
        self.pred_proj = pred_proj

    def encode_volume(self, volume):
        encoded = self.encoder(volume)
        patch_embeddings = self.projector(encoded["patch_embeddings"])
        return {
            "patch_embeddings": patch_embeddings,
            "cls_embedding": encoded["cls_embedding"],
            "grid_size": encoded["grid_size"],
        }

    def encode_src_tgt(self, src, tgt):
        src_output = self.encode_volume(src)
        tgt_output = self.encode_volume(tgt)
        return src_output, tgt_output

    def predict_tgt(self, src_embeddings):
        predicted = self.predictor(src_embeddings)
        return self.pred_proj(predicted)
```

Create `src/jedi/models/__init__.py`:
```python
from jedi.models.components import MLP
from jedi.models.jepa import CrossModalityJEPA
from jedi.models.predictor import LatentPredictor
from jedi.models.regularizers import SIGReg
from jedi.models.vit3d import ViT3DEncoder

__all__ = ["MLP", "CrossModalityJEPA", "LatentPredictor", "SIGReg", "ViT3DEncoder"]
```

- [ ] **Step 4: Create the stage-1 Lightning module**

Create `src/jedi/training/encoder_module.py`:
```python
import lightning as pl
import torch
import torch.nn.functional as F


class EncoderTrainingModule(pl.LightningModule):
    def __init__(self, model, regularizer, lr, weight_decay, sigreg_weight):
        super().__init__()
        self.model = model
        self.regularizer = regularizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.sigreg_weight = sigreg_weight

    def training_step(self, batch, batch_idx):
        src_output, tgt_output = self.model.encode_src_tgt(batch["src"], batch["tgt"])
        pred_tgt_emb = self.model.predict_tgt(src_output["patch_embeddings"])
        pred_loss = F.mse_loss(pred_tgt_emb, tgt_output["patch_embeddings"].detach())
        sigreg_loss = self.regularizer(tgt_output["patch_embeddings"].transpose(0, 1))
        loss = pred_loss + self.sigreg_weight * sigreg_loss
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
```

- [ ] **Step 5: Run test to verify it passes**

Run:
```bash
PYTHONPATH=/Users/junran/Documents/jedi/src python -m pytest tests/test_encoder_step.py -v
```

Expected: `1 passed`

- [ ] **Step 6: Commit**

Run:
```bash
git add src/jedi/models/predictor.py src/jedi/models/jepa.py src/jedi/models/__init__.py src/jedi/training/encoder_module.py tests/test_encoder_step.py
git commit -m "feat: add cross-modality jepa stage"
```

### Task 7: Build decoder and frozen stage-2 training module

**Files:**
- Create: `src/jedi/models/decoder3d.py`
- Create: `src/jedi/training/decoder_module.py`
- Create: `src/jedi/training/__init__.py`
- Test: `tests/test_decoder_step.py`

- [ ] **Step 1: Write the failing stage-2 smoke test**

Create `tests/test_decoder_step.py`:
```python
import unittest

import torch

from jedi.models import CrossModalityJEPA, MLP, ViT3DEncoder
from jedi.models.decoder3d import VolumeDecoder3D
from jedi.models.predictor import LatentPredictor
from jedi.training.decoder_module import DecoderTrainingModule


class TestDecoderStep(unittest.TestCase):
    def test_stage2_step_freezes_encoder_side_and_returns_finite_loss(self):
        encoder = ViT3DEncoder(
            image_size=(32, 32, 32),
            patch_size=(8, 8, 8),
            in_channels=1,
            embed_dim=64,
            depth=2,
            heads=4,
            dim_head=16,
            mlp_dim=128,
        )
        projector = MLP(input_dim=64, hidden_dim=128, output_dim=64, norm_fn="LayerNorm")
        predictor = LatentPredictor(input_dim=64, hidden_dim=64)
        pred_proj = MLP(input_dim=64, hidden_dim=128, output_dim=64, norm_fn="LayerNorm")
        model = CrossModalityJEPA(encoder=encoder, projector=projector, predictor=predictor, pred_proj=pred_proj)
        decoder = VolumeDecoder3D(embed_dim=64, patch_size=(8, 8, 8), out_channels=1, hidden_channels=32)
        module = DecoderTrainingModule(model=model, decoder=decoder, lr=1e-4, weight_decay=1e-4)
        batch = {
            "src": torch.randn(2, 1, 32, 32, 32),
            "tgt": torch.randn(2, 1, 32, 32, 32),
        }
        loss = module.training_step(batch, 0)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(not param.requires_grad for param in model.parameters()))
        self.assertTrue(any(param.requires_grad for param in decoder.parameters()))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
PYTHONPATH=/Users/junran/Documents/jedi/src python -m pytest tests/test_decoder_step.py -v
```

Expected: FAIL with import errors for missing decoder modules

- [ ] **Step 3: Write the minimal decoder implementation**

Create `src/jedi/models/decoder3d.py`:
```python
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
```

Create `src/jedi/training/decoder_module.py`:
```python
import lightning as pl
import torch
import torch.nn.functional as F


class DecoderTrainingModule(pl.LightningModule):
    def __init__(self, model, decoder, lr, weight_decay):
        super().__init__()
        self.model = model
        self.decoder = decoder
        self.lr = lr
        self.weight_decay = weight_decay
        for param in self.model.parameters():
            param.requires_grad = False

    def training_step(self, batch, batch_idx):
        with torch.no_grad():
            src_output, _ = self.model.encode_src_tgt(batch["src"], batch["tgt"])
            pred_tgt_emb = self.model.predict_tgt(src_output["patch_embeddings"])
            grid_size = self.model.encoder(batch["src"])["grid_size"]
        prediction = self.decoder(pred_tgt_emb, grid_size)
        return F.l1_loss(prediction, batch["tgt"])

    def configure_optimizers(self):
        return torch.optim.AdamW(self.decoder.parameters(), lr=self.lr, weight_decay=self.weight_decay)
```

Create `src/jedi/training/__init__.py`:
```python
from jedi.training.decoder_module import DecoderTrainingModule
from jedi.training.encoder_module import EncoderTrainingModule

__all__ = ["DecoderTrainingModule", "EncoderTrainingModule"]
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
PYTHONPATH=/Users/junran/Documents/jedi/src python -m pytest tests/test_decoder_step.py -v
```

Expected: `1 passed`

- [ ] **Step 5: Commit**

Run:
```bash
git add src/jedi/models/decoder3d.py src/jedi/training/decoder_module.py src/jedi/training/__init__.py tests/test_decoder_step.py
git commit -m "feat: add frozen decoder training stage"
```

### Task 8: Add Hydra configs and stage entrypoints

**Files:**
- Create: `src/jedi/config/train_encoder.yaml`
- Create: `src/jedi/config/train_decoder.yaml`
- Create: `src/jedi/config/data/brats2023.yaml`
- Create: `src/jedi/config/model/encoder.yaml`
- Create: `src/jedi/config/model/decoder.yaml`
- Create: `src/jedi/train_encoder.py`
- Create: `src/jedi/train_decoder.py`
- Test: import commands for both entrypoints

- [ ] **Step 1: Write the failing entrypoint import check**

Run:
```bash
PYTHONPATH=/Users/junran/Documents/jedi/src python -c "from jedi.train_encoder import main; from jedi.train_decoder import main; print(callable(main))"
```

Expected: FAIL with `ModuleNotFoundError` for missing files

- [ ] **Step 2: Write minimal configs and entrypoints**

Create `src/jedi/config/data/brats2023.yaml`:
```yaml
data_dir: /path/to/BraTS2023
spatial_size: [128, 160, 192]
batch_size: 1
num_workers: 4
fixed_mapping: [t1n, t2w]
a_min: 0.0
a_max: 3000.0
```

Create `src/jedi/config/model/encoder.yaml`:
```yaml
encoder:
  _target_: jedi.models.ViT3DEncoder
  image_size: [128, 160, 192]
  patch_size: [8, 8, 8]
  in_channels: 1
  embed_dim: 256
  depth: 4
  heads: 8
  dim_head: 32
  mlp_dim: 512
projector:
  _target_: jedi.models.MLP
  input_dim: 256
  hidden_dim: 512
  output_dim: 256
  norm_fn: LayerNorm
predictor:
  _target_: jedi.models.LatentPredictor
  input_dim: 256
  hidden_dim: 256
pred_proj:
  _target_: jedi.models.MLP
  input_dim: 256
  hidden_dim: 512
  output_dim: 256
  norm_fn: LayerNorm
regularizer:
  _target_: jedi.models.SIGReg
  knots: 17
  num_proj: 1024
```

Create `src/jedi/config/model/decoder.yaml`:
```yaml
decoder:
  _target_: jedi.models.decoder3d.VolumeDecoder3D
  embed_dim: 256
  patch_size: [8, 8, 8]
  out_channels: 1
  hidden_channels: 64
encoder_checkpoint: /path/to/stage1.ckpt
```

Create `src/jedi/config/train_encoder.yaml`:
```yaml
defaults:
  - data: brats2023
  - model: encoder
  - _self_

seed: 2024
optimizer:
  lr: 1e-4
  weight_decay: 1e-4
loss:
  sigreg_weight: 0.01
trainer:
  max_epochs: 100
  accelerator: gpu
  devices: 1
```

Create `src/jedi/config/train_decoder.yaml`:
```yaml
defaults:
  - data: brats2023
  - model: encoder
  - model@decoder_model: decoder
  - _self_

seed: 2024
optimizer:
  lr: 1e-4
  weight_decay: 1e-4
trainer:
  max_epochs: 100
  accelerator: gpu
  devices: 1
```

Create `src/jedi/train_encoder.py`:
```python
import hydra
import lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig

from jedi.data.brats import build_dataloader
from jedi.models import CrossModalityJEPA
from jedi.training.encoder_module import EncoderTrainingModule


@hydra.main(version_base=None, config_path="config", config_name="train_encoder")
def main(cfg: DictConfig):
    pl.seed_everything(cfg.seed)
    encoder = instantiate(cfg.model.encoder)
    projector = instantiate(cfg.model.projector)
    predictor = instantiate(cfg.model.predictor)
    pred_proj = instantiate(cfg.model.pred_proj)
    regularizer = instantiate(cfg.model.regularizer)
    model = CrossModalityJEPA(encoder=encoder, projector=projector, predictor=predictor, pred_proj=pred_proj)
    module = EncoderTrainingModule(
        model=model,
        regularizer=regularizer,
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
        sigreg_weight=cfg.loss.sigreg_weight,
    )
    train_loader = build_dataloader(cfg.data.data_dir, "train", tuple(cfg.data.fixed_mapping), cfg.data.batch_size, cfg.data.num_workers, tuple(cfg.data.spatial_size))
    val_loader = build_dataloader(cfg.data.data_dir, "val", tuple(cfg.data.fixed_mapping), cfg.data.batch_size, cfg.data.num_workers, tuple(cfg.data.spatial_size))
    trainer = pl.Trainer(max_epochs=cfg.trainer.max_epochs, accelerator=cfg.trainer.accelerator, devices=cfg.trainer.devices)
    trainer.fit(module, train_loader, val_loader)
```

Create `src/jedi/train_decoder.py`:
```python
import hydra
import lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig

from jedi.data.brats import build_dataloader
from jedi.models import CrossModalityJEPA
from jedi.training.decoder_module import DecoderTrainingModule


@hydra.main(version_base=None, config_path="config", config_name="train_decoder")
def main(cfg: DictConfig):
    pl.seed_everything(cfg.seed)
    encoder = instantiate(cfg.model.encoder)
    projector = instantiate(cfg.model.projector)
    predictor = instantiate(cfg.model.predictor)
    pred_proj = instantiate(cfg.model.pred_proj)
    model = CrossModalityJEPA(encoder=encoder, projector=projector, predictor=predictor, pred_proj=pred_proj)
    decoder = instantiate(cfg.decoder_model.decoder)
    module = DecoderTrainingModule(model=model, decoder=decoder, lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay)
    train_loader = build_dataloader(cfg.data.data_dir, "train", tuple(cfg.data.fixed_mapping), cfg.data.batch_size, cfg.data.num_workers, tuple(cfg.data.spatial_size))
    val_loader = build_dataloader(cfg.data.data_dir, "val", tuple(cfg.data.fixed_mapping), cfg.data.batch_size, cfg.data.num_workers, tuple(cfg.data.spatial_size))
    trainer = pl.Trainer(max_epochs=cfg.trainer.max_epochs, accelerator=cfg.trainer.accelerator, devices=cfg.trainer.devices)
    trainer.fit(module, train_loader, val_loader)
```

- [ ] **Step 3: Run the import check to verify it passes**

Run:
```bash
PYTHONPATH=/Users/junran/Documents/jedi/src python -c "from jedi.train_encoder import main as encoder_main; from jedi.train_decoder import main as decoder_main; print(callable(encoder_main) and callable(decoder_main))"
```

Expected: output `True`

- [ ] **Step 4: Commit**

Run:
```bash
git add src/jedi/config/train_encoder.yaml src/jedi/config/train_decoder.yaml src/jedi/config/data/brats2023.yaml src/jedi/config/model/encoder.yaml src/jedi/config/model/decoder.yaml src/jedi/train_encoder.py src/jedi/train_decoder.py
git commit -m "feat: add hydra training entrypoints"
```

### Task 9: Add inference entrypoint that uses predicted target latents

**Files:**
- Create: `src/jedi/infer.py`
- Test: `python -c "from jedi.infer import run_inference; print(callable(run_inference))"`

- [ ] **Step 1: Write the failing inference import check**

Run:
```bash
PYTHONPATH=/Users/junran/Documents/jedi/src python -c "from jedi.infer import run_inference; print(callable(run_inference))"
```

Expected: FAIL with `ModuleNotFoundError: No module named 'jedi.infer'`

- [ ] **Step 2: Write minimal inference entrypoint**

Create `src/jedi/infer.py`:
```python
from __future__ import annotations

import torch


def run_inference(model, decoder, src_volume: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        src_output = model.encode_volume(src_volume)
        pred_tgt_emb = model.predict_tgt(src_output["patch_embeddings"])
        grid_size = model.encoder(src_volume)["grid_size"]
        return decoder(pred_tgt_emb, grid_size)


def main():
    raise NotImplementedError("Wire checkpoints and CLI arguments before production inference use.")
```

- [ ] **Step 3: Run the inference import check to verify it passes**

Run:
```bash
PYTHONPATH=/Users/junran/Documents/jedi/src python -c "from jedi.infer import run_inference; print(callable(run_inference))"
```

Expected: output `True`

- [ ] **Step 4: Commit**

Run:
```bash
git add src/jedi/infer.py
git commit -m "feat: add inference entrypoint"
```

### Task 10: Write README and remove the obsolete prototype directory

**Files:**
- Create: `README.md`
- Delete: `brats-contrast-model/`
- Test: `ls /Users/junran/Documents/jedi`

- [ ] **Step 1: Write the README**

Create `README.md`:
```markdown
# Jedi

BraTS2023 cross-modality contrast conversion with a `le-wm`-aligned two-stage pipeline.

## Stages

### Stage 1
Train a shared 3D ViT encoder, projector, predictor, and pred-proj so that predicted target latents align with encoded target latents.

### Stage 2
Freeze the stage-1 stack and train only the decoder from predicted target latents to target MRI volumes.

## Data layout

```text
BraTS-GLI-00001-000/
  BraTS-GLI-00001-000-t1n.nii.gz
  BraTS-GLI-00001-000-t1c.nii.gz
  BraTS-GLI-00001-000-t2w.nii.gz
  BraTS-GLI-00001-000-t2f.nii.gz
```

## Commands

```bash
PYTHONPATH=src python -m jedi.train_encoder
PYTHONPATH=src python -m jedi.train_decoder
```
```

- [ ] **Step 2: Remove the obsolete prototype directory**

Run:
```bash
rm -rf /Users/junran/Documents/jedi/brats-contrast-model
```

- [ ] **Step 3: Verify the workspace layout**

Run:
```bash
ls /Users/junran/Documents/jedi
```

Expected: includes `README.md`, `pyproject.toml`, `src`, `tests`, `docs`

- [ ] **Step 4: Commit**

Run:
```bash
git add README.md
git add -A brats-contrast-model
git commit -m "docs: add project README and remove prototype"
```

## Spec coverage check
- `src` layout and package name `jedi`: covered by Task 1
- MONAI data loading, fixed-size padding/cropping, normalization to `(-1, 1)`: covered by Tasks 2 and 3
- random training mapping and fixed eval mapping: covered by Task 3
- 3D ViT encoder: covered by Task 5
- `le-wm`-aligned cross-modality JEPA stage: covered by Task 6
- frozen stage-2 decoder from `pred_tgt_emb`: covered by Task 7
- Hydra entrypoints and config split: covered by Task 8
- inference path using predicted target latents: covered by Task 9
- README and cleanup of obsolete generated prototype: covered by Task 10

## Placeholder scan
- No `TODO`, `TBD`, or “implement later” placeholders remain in tasks
- All code-producing steps include concrete code blocks
- All verification steps include explicit commands and expected outcomes

## Type consistency check
- Stage 1 always uses `CrossModalityJEPA.encode_src_tgt()` and `predict_tgt()`
- Stage 2 always decodes `pred_tgt_emb`
- Dataset sample keys remain `src`, `tgt`, `src_modality`, `tgt_modality`, `case_id` across all tasks
