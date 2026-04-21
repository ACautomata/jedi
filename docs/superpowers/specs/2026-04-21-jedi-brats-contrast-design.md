# Jedi BraTS2023 Contrast Conversion Design

## Goal

Build a BraTS2023 contrast conversion project in the current repository using `src` layout with package name `jedi`.
The implementation should follow the `le-wm` training flow as closely as possible, while replacing temporal prediction with cross-modality prediction.

Core requirements:
- 3D input, 3D output
- MONAI-based dataset loading
- MRI size handling by padding to fixed size
- intensity normalization to `(-1, 1)`
- 3D ViT encoder
- stage 1 learns cross-modality latent prediction
- stage 2 freezes the encoder-side stack and trains only the decoder

## Confirmed decisions

### Repository structure
Use the current directory as the project root and adopt `src` layout.
The Python package name is `jedi`.

Recommended structure:

```text
jedi/
  pyproject.toml
  README.md
  src/
    jedi/
      __init__.py
      train_encoder.py
      train_decoder.py
      infer.py
      config/
        train_encoder.yaml
        train_decoder.yaml
        data/
          brats2023.yaml
        model/
          encoder.yaml
          decoder.yaml
      data/
        brats.py
        transforms.py
      models/
        vit3d.py
        jepa.py
        predictor.py
        decoder3d.py
        losses.py
      training/
        encoder_module.py
        decoder_module.py
        callbacks.py
      utils/
        io.py
        checkpoint.py
        modality.py
  tests/
    test_vit3d_shapes.py
    test_brats_dataset.py
    test_encoder_step.py
    test_decoder_step.py
```

### Stage 1 training flow
Stage 1 follows the `le-wm` flow, but the target is cross-modality rather than future-in-time.

Training path:

```text
src_volume -> encoder -> projector -> src_emb
tgt_volume -> encoder -> projector -> tgt_emb
src_emb -> predictor -> pred_proj -> pred_tgt_emb
loss(pred_tgt_emb, tgt_emb) + regularizer
```

Important constraints:
- the same encoder is applied to both `src` and `tgt`
- the same projector is applied to both `src` and `tgt`
- predictor output is aligned to the encoded latent of `tgt`
- this is not source-only self-supervised pretraining

Saved artifacts:
- `checkpoints/stage1/<experiment>/encoder.ckpt`
- optional full stage-1 checkpoint for reproducibility

### Stage 2 training flow
Stage 2 keeps the encoder-side prediction stack fixed and trains only the decoder.

Training path:

```text
src_volume -> encoder -> projector -> predictor -> pred_proj -> pred_tgt_emb
pred_tgt_emb -> decoder -> reconstructed_tgt_volume
reconstruction_loss(reconstructed_tgt_volume, tgt_volume)
```

Frozen modules:
- encoder
- projector
- predictor
- pred_proj

Trainable modules:
- decoder only

Saved artifacts:
- `checkpoints/stage2/<src>_to_<tgt>/<experiment>/decoder.ckpt`

Inference takes two explicit checkpoint paths:
- encoder-side checkpoint path
- decoder checkpoint path

## Model design

### Encoder
Use a 3D ViT as the encoder backbone.
Input shape is `(B, 1, D, H, W)`.
The encoder performs 3D patch embedding and adds 3D positional embeddings.
The output exposes token-level latent features used by projector and decoder.

The first version uses single-channel, single-modality input only.
One forward pass consumes either `src` or `tgt`, never a channel-stacked multi-modality tensor.

### Projector / Predictor / PredProj
Keep the role split close to `le-wm`:
- `projector`: map encoder output into the training latent space
- `predictor`: predict target latent from source latent
- `pred_proj`: map predictor output into the final alignment space

### Decoder
Decoder input is `pred_tgt_emb`, not `src_emb`.
This preserves the same logic as `le-wm`: the decoder consumes the representation that is trained to match the target-side latent.

The first version uses a patch-based 3D decoder:
- token latent to voxel-patch projection
- patch-grid reassembly into a coarse 3D volume
- a small stack of `Conv3d` refinement blocks
- final `Tanh` to match the normalized range `(-1, 1)`

### Losses
Stage 1:
- latent prediction loss between `pred_tgt_emb` and `tgt_emb`
- `SIGReg` or the equivalent latent regularizer used in `le-wm`

Stage 2:
- reconstruction loss between predicted target volume and real target volume
- first version uses `L1` only
- no extra SSIM or perceptual loss in the first version

## Data pipeline

### BraTS pairing
Each case directory contains modality files such as:
- `t1n`
- `t1c`
- `t2w`
- `t2f`

A sample returns:
- `src`
- `tgt`
- `case_id`
- `src_modality`
- `tgt_modality`

### Mapping strategy
Training:
- each batch randomly samples one `src -> tgt` mapping
- one sample remains single-pair only
- `src == tgt` is forbidden

Validation and test:
- use one fixed mapping pair from config
- example: always `t1n -> t2w`
- validation and test must not use random mapping logic

### MONAI transforms
Recommended transform order:
1. `LoadImaged`
2. `EnsureChannelFirstd`
3. `Orientationd`
4. `Spacingd`
5. `SpatialPadd`
6. fixed-size handling after pad
7. `ScaleIntensityRanged` to `(-1, 1)`
8. `EnsureTyped`

### Size handling
The model trains on full 3D volumes with fixed input size.
Padding is the primary size-handling mechanism.
If a case is smaller than the target size, pad it.
If a case is larger than the target size, center crop it to the target size.
This keeps the full-volume training interface valid while honoring the padding-first requirement.

### Intensity normalization
Both `src` and `tgt` are mapped to `(-1, 1)`.
The first version uses a fixed intensity range rather than per-case z-score normalization.
This keeps the target distribution aligned with a `Tanh` decoder output.

### Augmentation
Keep augmentation minimal in the first version:
- random flips
- optionally small affine perturbations

Avoid heavy intensity augmentation in the first version because the task itself is cross-contrast translation.

## Testing boundary

### Data tests
- case discovery and modality pairing work
- training mode samples random single-pair mappings
- validation/test mode always returns the configured fixed mapping
- padding and normalization produce fixed-size tensors within `[-1, 1]`

### Model tests
- 3D ViT produces expected token shapes
- stage 1 forward pass produces finite latent losses
- stage 2 forward pass reconstructs a volume with the expected target shape
- stage 2 freezes encoder-side modules and keeps decoder trainable

### Smoke tests
- one synthetic stage-1 training step
- one synthetic stage-2 training step
- no shape or pipeline failures

### Explicitly out of scope for v1
- long-horizon convergence tests
- benchmark-grade medical metrics
- multi-GPU support
- visual regression fixtures

## Implementation guidance

The implementation should start with the minimum code necessary to prove the architecture and training flow.
Avoid speculative abstractions.
Every changed file should trace directly to one of the confirmed requirements above.
