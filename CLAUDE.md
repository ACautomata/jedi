# jedi

## Commands
```bash
PYTHONPATH=src .venv/bin/python -m pytest tests -v           # run tests
PYTHONPATH=src .venv/bin/python -m jedi.train_encoder       # stage 1 training
PYTHONPATH=src .venv/bin/python -m jedi.train_decoder       # stage 2 training
```

## Environment
- `python` not available; use `python3` or `.venv/bin/python`
- `uv` is installed; use `uv venv` + `uv pip install -e .` to set up

## Architecture
- Two-stage training based on le-wm JEPA flow
- Stage 1: CrossModalityJEPA — shared 3D ViT encodes src/tgt, predictor aligns pred_tgt_emb to tgt latent
- Stage 2: freezes encoder-side stack (encoder/projector/predictor/pred_proj), trains only VolumeDecoder3D from pred_tgt_emb
- `jedi/models/jepa.py` is the core model wrapper; `jedi/data/brats.py` is the dataset

## Key patterns
- `BraTSContrastDataset.__getitem__` must call `self.transform(sample)` — not returning raw dicts
- MONAI transform chain uses `SpatialPadd` + `CenterSpatialCropd` to guarantee fixed spatial_size
- Stage 2 loads stage 1 checkpoint via `load_encoder_side_checkpoint()` before freezing
- Decoder output uses `Tanh` to match `(-1, 1)` normalization
- Lightning `validation_step` should wrap the entire computation in `torch.no_grad()`, not just the encoder

## Dependencies
- `pytorch-wavelets` requires `uv pip install PyWavelets` as an undeclared runtime dependency
