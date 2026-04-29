# Research Idea Report

**方向**: BraTS2023 跨模态 MRI 对比度转换 — 解码器架构 + 频域损失优化
**生成日期**: 2026-04-26
**Pipeline**: research-lit → idea-creator (Codex GPT-5.4 xhigh)
**Idea 评估**: 12 个生成 → 8 个通过过滤 → 3 个推荐优先执行

## Landscape Summary

当前 JEDI 项目是唯一已知的将 JEPA 框架用于 BraTS 跨模态 MRI 对比度转换的方法。文献调研发现：

1. **JEPA/SIGReg 理论**: LeJEPA (Nov 2025) 理论上证明了各向同性高斯是最优嵌入分布；LeWorldModel (Mar 2026) 展示端到端 JEPA 在单 GPU 上的可行性和 48× 加速。这些最新理论进展尚未在医学影像领域被应用。

2. **BraTS 模态翻译 SOTA**: 主流方法为 GANs (cGAN, CycleGAN) 和扩散模型 (Latent Diffusion, FADM)。没有 JEPA 类方法。

3. **解码器趋势**: EffiDec3D (CVPR 2025) 展示通道压缩可减少 96% 参数；LKDA-Net 用大核 depthwise 卷积模拟自注意力；Cross-attention transformer decoder 在 ViT 可视化中被证明有效。

4. **频域损失**: FADM (2025) 用小波变换在 BraTS 上验证频域损失有效性；频率分解 + 差异化约束在各频带上都带来收益。

## 推荐 Idea（排序）

### Idea 1: Cross-Attention Patch Decoder Baseline ⭐
- **假设**: 用可学习 query tokens 的 transformer decoder 替换当前线性卷积解码器，能显著提升重建质量。因为每个 query 可以通过 cross-attention 关注所有预测 patch embedding 获取全局上下文，而不是仅做局部线性投影。
- **最小实验**: 固定 Stage 1，只在 Stage 2 训练。decoder depth=2, heads=8, query=7680 (每 patch 一个), dim=256。Loss 仅用 L1。训练 20 epochs。对比当前 VolumeDecoder3D。
- **贡献类型**: method
- **风险**: LOW
- **预估工作量**: 数小时
- **差异化**: 首个 JEPA latent-to-volume 交叉注意力解码器基线，与 GAN/U-Net 类解码器正交

### Idea 2: Fourier 振幅损失 (Amplitude Loss) ⭐
- **假设**: 振幅损失应该能改善整体强度纹理和模态特定的频率统计，而不会过度约束空间对齐。因为预测的 latent 可能保留了语义结构但丢失了高频对比度细节。
- **最小实验**: 在 Idea 1 的 cross-attention decoder 基础上叠加振幅损失。L = L1 + λ_amp * L_amp, λ_amp ∈ {0.05, 0.1}。训练 20 epochs。
- **贡献类型**: empirical
- **风险**: LOW
- **预估工作量**: 数小时
- **差异化**: 测试 JEPA latent 解码是否特别受益于频谱分布匹配，这不同于直接的 image-to-image 频域损失

### Idea 3: 相位感知 Fourier 损失 (Phase-Aware Loss)
- **假设**: 相位损失可能改善肿瘤边界位置和解剖结构，但若预测图像初始质量差可能不稳定。弱相位项可以揭示 JEPA embedding 是否保留了足够的位置结构。
- **最小实验**: L = L1 + 0.05 * L_amp + λ_phase * L_phase, λ_phase ∈ {0.005, 0.01}。前 5 epoch 不使用相位损失（预热）。
- **贡献类型**: empirical / diagnostic
- **风险**: MEDIUM
- **预估工作量**: 数小时
- **差异化**: 在 JEPA 重建场景中分离振幅 vs 相位的效用分析

### Idea 4: 频带消融实验 (Low/Mid/High Band Ablation)
- **假设**: 并非所有频带都同等重要。低频振幅影响模态强度风格，高频影响边缘、肿瘤边界和噪声。频带特定损失可以揭示 JEPA 解码在哪些频率上失败。
- **最小实验**: 三组 15 epoch 训练：低(0-20%)、中(20-50%)、高(50-100%) 频振幅损失。用径向 mask 在 rFFT 空间实现。
- **贡献类型**: diagnostic
- **风险**: LOW
- **预估工作量**: 数小时
- **差异化**: 产生关于 JEPA latent 缺失哪些频谱信息的可解释证据

### Idea 5: 渐进式频率课程学习 (Frequency Curriculum)
- **假设**: 早期训练应优先粗粒度结构，后期引入高频约束。这能避免频域损失在解剖结构重建完成前拉偏 decoder。
- **最小实验**: 25 epoch。阶段 1(1-5): L1 only；阶段 2(6-15): L1 + 低中频振幅；阶段 3(16-25): L1 + 全频段振幅 + 弱相位。
- **贡献类型**: method
- **风险**: MEDIUM
- **预估工作量**: 数小时
- **差异化**: 将课程学习适配到频谱监督的 JEPA latent-to-MRI 解码

### Idea 6: Query 初始化策略比较
- **假设**: 全随机学习的 query 可能浪费容量重新发现空间布局。用 3D 正弦位置编码或可学习因子化位置编码初始化可改善收敛。
- **最小实验**: 15 epoch 比较：A-随机学习；B-固定 3D 正弦编码+可学习投影；C-可学习因子化位置编码 q[x]+q[y]+q[z]。Loss = L1 + 0.05 振幅。
- **贡献类型**: diagnostic / method
- **风险**: LOW
- **预估工作量**: 数小时
- **差异化**: 测试 JEPA 解码的性能瓶颈在 latent 信息不足还是 decoder 空间 query 设计

### Idea 7: Cross-Attention Depth Scaling
- **假设**: 浅层 cross-attention decoder 可能已足够从 pred_tgt_emb 提取信息；深层 decoder 可能过拟合或 OOM。
- **最小实验**: depth={1,2,4}, heads=8, dim=256, Loss = L1+0.05 振幅。训练 12 epoch。使用 gradient checkpointing。
- **贡献类型**: empirical / diagnostic
- **风险**: LOW
- **预估工作量**: 数小时
- **差异化**: 提供 transformer decoder 在 latent MRI 对比度转换中的扩展性证据

### Idea 8: 混合 Transformer-Conv 精修 (Hybrid Refinement)
- **假设**: Cross-attention 可以恢复全局 patch 内容，但体素级平滑和局部边缘可能需要小卷积头精修。
- **最小实验**: Cross-attention decoder 输出 reshape 为 volume 后，接 Conv3d(1,32,3)+IN+GELU+Conv3d(32,32,3)+IN+GELU+Conv3d(32,1,1)。Loss = L1+0.05 振幅。对比无精修版本。
- **贡献类型**: method
- **风险**: LOW
- **预估工作量**: 数小时
- **差异化**: 最小混合 decoder：transformer 做 latent 条件化，conv 只做体素清理

## 推荐执行顺序

基于风险和信号清晰度的推荐执行路径：

```
Phase A（基线建立）:
  Idea 1 → Cross-Attention Decoder + L1 基线
  ↓ 与当前 VolumeDecoder3D 对比
Phase B（频域损失探索）:
  Idea 2 → 叠加振幅损失（找最佳 λ）
  ↓
  Idea 4 → 频带消融（看哪个频段问题最大）
  ↓ 如振幅有帮助：
  Idea 3 → 加入弱相位损失（预热后）
Phase C（消融和优化）:
  Idea 6 → Query 初始化策略比较
  Idea 7 → Depth 消融
  Idea 8 → 混合精修
Phase D（高级）:
  Idea 5 → 课程学习
```

## 核心技术设计

### Cross-Attention Decoder 架构

```
pred_tgt_emb (B, 7680, 256) 
    │
    ├── Linear proj(dim_proj) → K, V  (与 embed_dim 相同，避免信息丢失)
    │
    └── 可学习 queries (7680, d_model=256)
            │
            └── cross-attn(Q, K, V) × N_layers
            │      每个 query 关注所有 K/V tokens
            │      FlashAttention 处理 O(7680²) 注意力矩阵
            │
            └── MLP (残差) × N_layers
            │
            └── output proj: Linear(256, 512) → patch voxels (8×8×8=512)
            │
            └── reshape: (B, 7680, 512) → (B, 1, 128, 160, 192)
            │
            └── [可选] Conv3d 精修 head → Tanh
```

### Fourier 频域损失

```
pred (B, 1, D, H, W), target (B, 1, D, H, W)
    │
    ├── rFFT3D (dim=-3,-2,-1)
    │
    ├── F_pred = torch.fft.rfftn(pred, dim=(-3,-2,-1))
    │   F_tgt  = torch.fft.rfftn(target, dim=(-3,-2,-1))
    │
    ├── amplitude loss: L1(log(1+|F_pred|), log(1+|F_tgt|))
    │     全局频率分布匹配
    │
    ├── phase loss (可选): L1(cos(angle_pred - angle_tgt), 1)
    │     结构/位置对齐，建议加预热和低权重
    │
    └── 频带消融: 径向 mask 分解低/中/高频
```

## 下一步
- [ ] Phase A: 实现 Cross-Attention Decoder + 与当前 decoder 对比
- [ ] Phase B: 频域损失实验
- [ ] 进入 /auto-review-loop 迭代优化
