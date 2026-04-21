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
PYTHONPATH=src .venv/bin/python -m jedi.train_encoder
PYTHONPATH=src .venv/bin/python -m jedi.train_decoder
```
