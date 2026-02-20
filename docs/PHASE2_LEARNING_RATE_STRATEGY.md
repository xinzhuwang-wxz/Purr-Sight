# Phase 2 分组学习率策略

## 概述

Phase 2训练使用**分组学习率**（Differential Learning Rates）策略，对不同组件使用不同的学习率，以实现最优的训练效果。

## 参数分组

### 1. 冻结参数（不训练）
- **Encoders**: image_encoder, audio_encoder, text_encoder
- **Aligner**: 从Phase 1加载的对齐器
- **学习率**: N/A（参数冻结）

### 2. Projector（投影头）
- **状态**: 从头训练
- **作用**: 将对齐后的特征投影到LLM输入空间
- **参数量**: ~8.4M
- **学习率**: **5x base_lr** (默认: 5e-4)
- **原因**: 
  - 投影头是随机初始化的，需要较大学习率快速收敛
  - 投影头参数较少，较大学习率不会导致不稳定

### 3. LoRA Adapters（LoRA适配器）
- **状态**: 微调预训练模型
- **作用**: 对LLM进行参数高效微调
- **参数量**: ~2.2M
- **学习率**: **0.5x base_lr** (默认: 5e-5)
- **原因**:
  - LoRA微调预训练模型，需要较小学习率保持稳定
  - 避免破坏预训练权重
  - 防止梯度爆炸和NaN loss

## 学习率配置

### 自动配置（推荐）
```yaml
phase2:
  learning_rate: 0.0001  # 基础学习率 1e-4
  # projector_lr 自动设置为 5e-4 (5x base)
  # lora_lr 自动设置为 5e-5 (0.5x base)
```

### 手动配置（高级）
```yaml
phase2:
  learning_rate: 0.0001    # 基础学习率
  projector_lr: 0.0005     # 投影头学习率 (手动指定)
  lora_lr: 0.00005         # LoRA学习率 (手动指定)
```

## 学习率调度

所有参数组使用相同的学习率调度策略：

1. **Warmup阶段**: 线性增长到目标学习率
   - 步数: `warmup_steps` (默认: 500步)
   - 目的: 避免训练初期的不稳定

2. **Cosine Decay**: 余弦衰减到0
   - 总步数: `total_training_steps`
   - 目的: 平滑收敛

## 参数量对比

| 组件 | 参数量 | 可训练 | 学习率 | 占比 |
|------|--------|--------|--------|------|
| Encoders | 104M | ❌ | - | - |
| Aligner | 1.7M | ❌ | - | - |
| **Projector** | **8.4M** | ✅ | **5e-4** | **79.5%** |
| **LoRA** | **2.2M** | ✅ | **5e-5** | **20.5%** |
| **Total Trainable** | **10.6M** | ✅ | - | **1.79%** |

## 为什么需要分组学习率？

### 问题：统一学习率的缺陷

如果对所有参数使用相同学习率（如2e-4）：

1. **Projector收敛慢**
   - 投影头从头训练，需要更多步数才能收敛
   - 导致训练初期loss下降缓慢

2. **LoRA不稳定**
   - LoRA学习率过大，容易破坏预训练权重
   - 可能导致梯度爆炸和NaN loss

### 解决方案：分组学习率

通过为不同组件设置不同学习率：

1. **Projector快速收敛**
   - 较大学习率（5e-4）加速投影头训练
   - 快速建立模态特征到LLM空间的映射

2. **LoRA稳定微调**
   - 较小学习率（5e-5）保护预训练权重
   - 避免梯度爆炸，防止NaN loss

## 实验建议

### 初始训练
```yaml
learning_rate: 0.0001  # 1e-4
# projector_lr: 5e-4 (自动)
# lora_lr: 5e-5 (自动)
epochs: 3
batch_size: 2
```

### 如果遇到NaN loss
```yaml
learning_rate: 0.00005  # 5e-5 (降低基础学习率)
# projector_lr: 2.5e-4 (自动)
# lora_lr: 2.5e-5 (自动)
gradient_clip_val: 0.5  # 添加梯度裁剪
```

### 如果收敛太慢
```yaml
learning_rate: 0.0002  # 2e-4 (提高基础学习率)
# projector_lr: 1e-3 (自动)
# lora_lr: 1e-4 (自动)
warmup_steps: 100  # 减少warmup步数
```

## 监控指标

训练时关注以下指标：

1. **Loss曲线**
   - 应该平滑下降
   - 如果出现NaN，降低学习率

2. **梯度范数**
   - 正常范围: 0.1-10
   - 如果>100，可能需要梯度裁剪

3. **学习率曲线**
   - Warmup阶段线性增长
   - 之后余弦衰减

## 代码实现

参见 `train/train_llm/multimodal_llm_module.py` 中的 `configure_optimizers()` 方法：

```python
def configure_optimizers(self):
    # 分组参数
    projector_params = []  # 投影头参数
    lora_params = []       # LoRA参数
    
    # 创建参数组
    param_groups = [
        {'params': projector_params, 'lr': self.projector_lr},
        {'params': lora_params, 'lr': self.lora_lr}
    ]
    
    # 创建优化器
    optimizer = torch.optim.AdamW(param_groups)
    
    # 创建调度器
    scheduler = get_cosine_schedule_with_warmup(...)
    
    return {"optimizer": optimizer, "lr_scheduler": scheduler}
```

## 参考文献

1. **Differential Learning Rates**: Howard & Ruder, "Universal Language Model Fine-tuning for Text Classification", 2018
2. **LoRA**: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", 2021
3. **Cosine Annealing**: Loshchilov & Hutter, "SGDR: Stochastic Gradient Descent with Warm Restarts", 2017

## 总结

✅ **使用分组学习率的优势**：
- Projector快速收敛（5x base_lr）
- LoRA稳定微调（0.5x base_lr）
- 避免NaN loss
- 更好的训练效果

⚠️ **注意事项**：
- 根据数据量调整学习率
- 监控loss和梯度范数
- 必要时使用梯度裁剪
