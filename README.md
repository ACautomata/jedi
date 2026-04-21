# Jedi

BraTS2023 跨模态对比度转换模型，基于 [le-wm](https://github.com/ACautomata/le-wm) 的 JEPA 流程。

核心思路：把 le-wm 中的"时序预测"替换为"跨模态预测"——同一个 3D ViT encoder 分别编码 source 和 target modality，训练 predictor 让 `predictor(encoder(src))` 的 latent 对齐 `encoder(tgt)` 的 latent；第二阶段冻结 encoder-side stack，只训练 decoder 从预测出的 target latent 重建 target volume。

## 架构

### Stage 1：跨模态 JEPA 潜在空间对齐

```text
src_volume ──┐                          ┌── pred_loss ──┐
             encoder (3D ViT, 共享参数)   │               │
tgt_volume ──┘                          │            SIGReg
             ↓                           │               │
          projector (共享)               │               │
             ↓                           ↓               ↓
     src_emb ──→ predictor ──→ pred_proj ──→ pred_tgt_emb
                                         ←── 对齐 ──→ tgt_emb
```

- `encoder`: 3D ViT，将 3D MRI volume 切成 3D patch 后用 transformer 编码
- `projector`: MLP，把 encoder 输出映射到训练用 latent 空间
- `predictor` + `pred_proj`: 从 src latent 预测 tgt latent
- 损失：prediction loss（MSE）+ SIGReg（latent 分布正则化）

### Stage 2：冻结 encoder，训练 decoder

```text
src_volume → encoder(frozen) → projector(frozen) → predictor(frozen) → pred_proj(frozen)
                                                                              ↓
                                                                    pred_tgt_emb → decoder → recon_volume
                                                                                              ↑
                                                                                    L1 loss vs tgt_volume
```

- `decoder`: patch-based 3D volume decoder，先线性投影回 voxel patch，再 reshape 成 volume，最后用 Conv3d 精修
- 输出层 `Tanh` 保证预测值在 `(-1, 1)` 范围内

### 数据策略

- **训练**：每个 batch 随机采样一个 `src -> tgt` 单对单模态映射（如 t1n→t2w），保证 `src != tgt`
- **验证/测试**：固定 config 中指定的映射对（如始终 t1n→t2w）

## 安装

```bash
git clone <repo-url> && cd jedi
uv venv
uv pip install -e .
```

依赖：PyTorch, Lightning, MONAI, Hydra, einops。需要 GPU。

## 数据准备

将 BraTS2023 解压后放到一个目录下，目录结构需符合以下格式：

```text
data_root/
  BraTS-GLI-00001-000/
    BraTS-GLI-00001-000-t1n.nii.gz
    BraTS-GLI-00001-000-t1c.nii.gz
    BraTS-GLI-00001-000-t2w.nii.gz
    BraTS-GLI-00001-000-t2f.nii.gz
  BraTS-GLI-00002-000/
    ...
```

然后在配置文件 `src/jedi/config/data/brats2023.yaml` 中修改 `data_dir` 指向该目录。

数据预处理流程：
1. `LoadImaged` 加载 NIfTI
2. `EnsureChannelFirstd` 添加通道维
3. `Orientationd` 统一到 RAS 方向
4. `SpatialPadd` + `CenterSpatialCropd` 保证固定尺寸（默认 128×160×192）
5. `ScaleIntensityRanged` 归一化到 `(-1, 1)`

## 训练

### Stage 1：encoder 预训练

```bash
PYTHONPATH=src .venv/bin/python -m jedi.train_encoder
```

关键参数在 `src/jedi/config/` 下通过 Hydra 覆盖：

```bash
PYTHONPATH=src .venv/bin/python -m jedi.train_encoder \
  data.data_dir=/path/to/BraTS2023 \
  data.fixed_mapping=[t1n,t2w] \
  data.spatial_size=[128,160,192] \
  trainer.max_epochs=100 \
  loss.sigreg_weight=0.01
```

训练完成后 checkpoint 保存在 Hydra 输出目录。

### Stage 2：decoder 训练

```bash
PYTHONPATH=src .venv/bin/python -m jedi.train_decoder \
  decoder_model.encoder_checkpoint=/path/to/stage1/encoder.ckpt
```

Stage 2 会自动加载 stage 1 的 encoder-side 权重并冻结，只训练 decoder。

## 推理

```bash
PYTHONPATH=src .venv/bin/python -m jedi.infer \
  --model-config src/jedi/config/model/encoder.yaml \
  --decoder-config src/jedi/config/model/decoder.yaml \
  --encoder-checkpoint /path/to/stage1.ckpt \
  --decoder-checkpoint /path/to/stage2.ckpt \
  --input /path/to/src_volume.pt \
  --output /path/to/predicted_tgt.pt
```

## 项目结构

```text
src/jedi/
  __init__.py
  train_encoder.py          # Stage 1 Hydra/Lightning 入口
  train_decoder.py          # Stage 2 Hydra/Lightning 入口
  infer.py                  # 推理 CLI
  config/
    train_encoder.yaml      # Stage 1 主配置
    train_decoder.yaml      # Stage 2 主配置
    data/brats2023.yaml     # 数据集配置
    model/encoder.yaml      # encoder-side 模型配置
    model/decoder.yaml      # decoder 模型配置
  data/
    brats.py                # BraTS 数据集（随机/固定映射）
    transforms.py           # MONAI transforms（pad/crop/normalize）
  models/
    vit3d.py                # 3D ViT encoder
    jepa.py                 # CrossModalityJEPA 模型封装
    predictor.py            # latent predictor
    decoder3d.py            # 3D volume decoder
    transformer.py          # transformer 基础块
    components.py           # MLP 等通用组件
    regularizers.py         # SIGReg 正则化器
  training/
    encoder_module.py       # Stage 1 LightningModule
    decoder_module.py       # Stage 2 LightningModule
tests/                      # 形状、数据集、前向步骤测试
```
