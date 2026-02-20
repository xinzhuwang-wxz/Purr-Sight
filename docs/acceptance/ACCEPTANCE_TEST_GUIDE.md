# Purr-Sight 验收测试指南

本文档描述了Purr-Sight项目的完整验收测试流程，包括Phase 1（对齐训练）、Phase 2（LLM微调）和推理模块的测试。

## 📋 目录

1. [环境准备](#环境准备)
2. [Phase 1 验收测试](#phase-1-验收测试)
3. [Phase 2 验收测试](#phase-2-验收测试)
4. [推理模块测试](#推理模块测试)
5. [验收标准](#验收标准)
6. [故障排除](#故障排除)

---

## 环境准备

### 1. 激活conda环境

```bash
conda activate purrsight
```

### 2. 验证环境

```bash
# 检查Python版本
python --version  # 应该是 3.10.x

# 检查关键依赖
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import lightning; print(f'Lightning: {lightning.__version__}')"
python -c "import hypothesis; print(f'Hypothesis: {hypothesis.__version__}')"
```

### 3. 准备数据

确保以下数据目录存在：
- `data/test_alignment/` - Phase 1 离线测试数据
- `data/instruction/` - Phase 1 在线训练数据
- `data/preprocessed/` - Phase 2 预处理数据

---

## Phase 1 验收测试

### 验收目标

1. ✅ 离线模式训练3个epoch
2. ✅ 在线模式训练3个epoch
3. ✅ 验证checkpoint文件正常生成
4. ✅ 验证MLflow日志正常记录

### 运行测试

#### 测试离线模式

```bash
python acceptance_test_phase1.py --mode offline --epochs 3
```

**预期输出：**
```
================================================================================
Phase 1 Acceptance Test - Mode: offline, Epochs: 3
================================================================================

Test 1: Training Execution
--------------------------------------------------------------------------------
Running command: python train/train_alignment/train.py --config config/train_config.yaml --max_epochs 3 ...
✅ Training completed successfully

Test 2: Checkpoint Verification
--------------------------------------------------------------------------------
✅ Found 3 checkpoint file(s)
  ✓ checkpoint_epoch001_step100_train_loss0.5000.pt: epoch=1, size=45.2MB
  ✓ checkpoint_epoch002_step200_train_loss0.4500.pt: epoch=2, size=45.2MB
  ✓ checkpoint_epoch003_step300_train_loss0.4000.pt: epoch=3, size=45.2MB
✅ 3/3 checkpoints are valid

Test 3: MLflow Logging Verification
--------------------------------------------------------------------------------
✅ Found 1 MLflow experiment(s)
✅ Found 1 recent MLflow run(s)
✅ Latest run has 5 metric(s):
  ✓ train_loss: 0.4000
  ✓ learning_rate: 0.0001
  ✓ epoch: 3.0000
  ...

Test 4: Checkpoint Loading Test
--------------------------------------------------------------------------------
Loading checkpoint: checkpoint_epoch003_step300_train_loss0.4000.pt
✅ Checkpoint structure is valid
  ✓ Epoch: 3
  ✓ Model parameters: 150 keys

================================================================================
✅ All acceptance tests PASSED
================================================================================

📄 Report saved to: acceptance_report_phase1_offline_20260201_143022.json
```

#### 测试在线模式

```bash
python acceptance_test_phase1.py --mode online --epochs 3
```

#### 测试两种模式

```bash
python acceptance_test_phase1.py --mode both --epochs 3
```

### 验证结果

#### 1. 检查checkpoint目录

```bash
ls -lh checkpoints/alignment/
```

**预期内容：**
- 至少3个checkpoint文件（每个epoch一个）
- 文件大小约40-50MB
- 文件名包含epoch、step和metrics信息

#### 2. 检查MLflow日志

```bash
# 启动MLflow UI
mlflow ui --backend-store-uri file://./mlruns

# 在浏览器打开 http://localhost:5000
```

**验证内容：**
- ✅ 实验名称正确（如"alignment_training"）
- ✅ 运行记录存在
- ✅ 指标被正确记录（train_loss, learning_rate等）
- ✅ 参数被正确记录（batch_size, learning_rate等）
- ✅ Artifacts包含checkpoint文件

#### 3. 检查日志文件

```bash
tail -100 logs/info.log
```

---

## Phase 2 验收测试

### 验收目标

1. ✅ 加载Phase 1 checkpoint
2. ✅ 应用LoRA微调
3. ✅ 训练3个epoch
4. ✅ 验证JSON输出格式
5. ✅ 验证checkpoint和MLflow日志

### 运行测试

#### 1. 找到Phase 1最佳checkpoint

```bash
# 列出所有Phase 1 checkpoints
ls -lt checkpoints/alignment/*.pt | head -5

# 或使用Python脚本查找
python -c "
from pathlib import Path
checkpoints = list(Path('checkpoints/alignment').glob('*.pt'))
if checkpoints:
    latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
    print(f'Latest checkpoint: {latest}')
"
```

#### 2. 运行Phase 2验收测试

```bash
# 使用找到的checkpoint路径
python acceptance_test_phase2.py \
    --phase1_checkpoint checkpoints/alignment/best_checkpoint_epoch003.pt \
    --epochs 3
```

**预期输出：**
```
================================================================================
Phase 2 Acceptance Test - Epochs: 3
Phase 1 Checkpoint: checkpoints/alignment/best_checkpoint_epoch003.pt
================================================================================

Test 1: Phase 1 Checkpoint Verification
--------------------------------------------------------------------------------
✅ Phase 1 checkpoint is valid
  ✓ Epoch: 3
  ✓ Aligner parameters: 45 keys
  ✓ Size: 45.2MB

Test 2: Phase 2 Training Execution (with LoRA)
--------------------------------------------------------------------------------
Running command: python train_phase2.py --config config/phase2_example.yaml ...
✅ Phase 2 training completed successfully

Test 3: JSON Output Verification
--------------------------------------------------------------------------------
Testing JSON output format...
✅ Model outputs valid JSON
  Sample output: {
    "behavior": "sitting",
    "posture": "relaxed",
    "activity_level": "low",
    ...
  }

Test 4: Phase 2 Checkpoint Verification
--------------------------------------------------------------------------------
✅ Found 3 checkpoint file(s)
✅ Found 25 LoRA parameter keys
  ✓ Latest checkpoint: checkpoint_epoch003_step300.pt
  ✓ Epoch: 3
  ✓ Size: 48.5MB

Test 5: MLflow Logging Verification
--------------------------------------------------------------------------------
✅ Found 1 MLflow run(s) in experiment 'phase2_training'
✅ Latest run has 6 metric(s)
  ✓ train_loss: 0.3500
  ✓ learning_rate: 0.0001
  ...

Test 6: LoRA Parameters Verification
--------------------------------------------------------------------------------
✅ Found 25 LoRA parameter keys
✅ Trainable parameters: 2,359,296 (5.23% of total)

================================================================================
✅ All acceptance tests PASSED
================================================================================

📄 Report saved to: acceptance_report_phase2_20260201_150045.json
```

### 验证结果

#### 1. 检查Phase 2 checkpoint

```bash
ls -lh checkpoints/phase2/
```

**验证内容：**
- ✅ Checkpoint文件存在
- ✅ 文件大小略大于Phase 1（包含LoRA参数）
- ✅ 包含LoRA权重

#### 2. 验证LoRA参数

```python
import torch

checkpoint = torch.load('checkpoints/phase2/latest.pt', map_location='cpu')
model_state = checkpoint['model_state_dict']

# 统计LoRA参数
lora_keys = [k for k in model_state.keys() if 'lora' in k.lower()]
print(f"LoRA parameters: {len(lora_keys)}")

# 检查可训练参数比例
metadata = checkpoint.get('metadata', {})
trainable = metadata.get('trainable_params', 0)
total = metadata.get('total_params', 0)
print(f"Trainable: {trainable:,} ({trainable/total*100:.2f}%)")
```

---

## 推理模块测试

### 验收目标

1. ✅ 视频输入 → JSON输出
2. ✅ 图片输入 → JSON输出
3. ✅ 文字输入 → JSON输出
4. ✅ JSON格式合理性验证

### 运行测试

#### 1. 准备测试数据

```bash
# 确保测试文件存在
ls data/cat.png
ls data/test1.mov
```

#### 2. 测试图片推理

```bash
python inference_module.py \
    --checkpoint checkpoints/phase2/best_checkpoint.pt \
    --image data/cat.png \
    --output results/inference_image.json
```

**预期输出：**
```
Initializing Purr-Sight Inference on device: cpu
Loading model from: checkpoints/phase2/best_checkpoint.pt
✅ Model loaded successfully (epoch 3)
✅ Inference pipeline initialized successfully
Processing image: cat.png

================================================================================
INFERENCE RESULT
================================================================================
{
  "timestamp": "2026-02-01T15:30:45.123456",
  "input_type": "image",
  "input_file": "data/cat.png",
  "model_checkpoint": "checkpoints/phase2/best_checkpoint.pt",
  "analysis": {
    "behavior": "sitting",
    "posture": "relaxed",
    "activity_level": "low",
    "emotional_state": "calm",
    "confidence": 0.85,
    "spatial_features": {
      "location": "indoor",
      "objects_detected": ["cat", "furniture", "window"],
      "scene_context": "home environment"
    }
  },
  "metadata": {
    "model_version": "1.0",
    "processing_time_ms": 150
  }
}
================================================================================

✅ Result saved to: results/inference_image.json

✅ Inference completed successfully
```

#### 3. 测试视频推理

```bash
python inference_module.py \
    --checkpoint checkpoints/phase2/best_checkpoint.pt \
    --video data/test1.mov \
    --output results/inference_video.json
```

#### 4. 测试文字推理

```bash
python inference_module.py \
    --checkpoint checkpoints/phase2/best_checkpoint.pt \
    --text "A cat is sitting on a windowsill, looking outside. The cat appears calm and relaxed." \
    --output results/inference_text.json
```

### 验证JSON输出

#### 1. 检查JSON格式

```python
import json

# 读取推理结果
with open('results/inference_image.json', 'r') as f:
    result = json.load(f)

# 验证必需字段
required_fields = ['timestamp', 'input_type', 'analysis', 'metadata']
for field in required_fields:
    assert field in result, f"Missing required field: {field}"

# 验证analysis结构
analysis = result['analysis']
assert 'behavior' in analysis
assert 'posture' in analysis
assert 'activity_level' in analysis
assert 'emotional_state' in analysis
assert 'confidence' in analysis

print("✅ JSON structure is valid")
```

#### 2. 验证输出合理性

检查以下内容：
- ✅ `behavior` 字段有意义（如"sitting", "walking", "playing"）
- ✅ `confidence` 在0-1之间
- ✅ `timestamp` 格式正确
- ✅ 根据输入类型有相应的特征字段：
  - 视频：`temporal_features`
  - 图片：`spatial_features`
  - 文字：`interpretation`

---

## 验收标准

### Phase 1 验收标准

| 测试项 | 标准 | 状态 |
|--------|------|------|
| 离线模式训练 | 3 epochs无错误完成 | ⬜ |
| 在线模式训练 | 3 epochs无错误完成 | ⬜ |
| Checkpoint生成 | 每个epoch生成有效checkpoint | ⬜ |
| Checkpoint大小 | 40-60MB范围内 | ⬜ |
| MLflow日志 | 实验和运行记录存在 | ⬜ |
| 指标记录 | train_loss, learning_rate等被记录 | ⬜ |
| Checkpoint加载 | 可以成功加载和验证 | ⬜ |

### Phase 2 验收标准

| 测试项 | 标准 | 状态 |
|--------|------|------|
| Phase 1 checkpoint加载 | 成功加载aligner权重 | ⬜ |
| LoRA应用 | LoRA参数正确添加 | ⬜ |
| 训练执行 | 3 epochs无错误完成 | ⬜ |
| 可训练参数比例 | 3-10%范围内 | ⬜ |
| JSON输出 | 模型输出有效JSON | ⬜ |
| Checkpoint生成 | 包含LoRA权重 | ⬜ |
| MLflow日志 | Phase 2实验记录存在 | ⬜ |

### 推理模块验收标准

| 测试项 | 标准 | 状态 |
|--------|------|------|
| 图片推理 | 成功处理并输出JSON | ⬜ |
| 视频推理 | 成功处理并输出JSON | ⬜ |
| 文字推理 | 成功处理并输出JSON | ⬜ |
| JSON格式 | 包含所有必需字段 | ⬜ |
| 输出合理性 | 行为分析结果有意义 | ⬜ |
| 置信度 | 0-1范围内 | ⬜ |
| 处理时间 | <5秒（CPU）或<1秒（GPU） | ⬜ |

---

## 故障排除

### 常见问题

#### 1. 训练失败：找不到数据

**错误：** `FileNotFoundError: data/test_alignment not found`

**解决：**
```bash
# 检查数据目录
ls -la data/

# 如果缺少测试数据，创建符号链接或复制数据
ln -s data/instruction data/test_alignment
```

#### 2. MLflow连接错误

**错误：** `mlflow.exceptions.MlflowException: Could not connect to tracking server`

**解决：**
```bash
# 设置本地tracking URI
export MLFLOW_TRACKING_URI=file://./mlruns

# 或在代码中设置
python -c "import mlflow; mlflow.set_tracking_uri('file://./mlruns')"
```

#### 3. Checkpoint加载失败

**错误：** `RuntimeError: Error loading checkpoint`

**解决：**
```python
# 检查checkpoint内容
import torch
checkpoint = torch.load('path/to/checkpoint.pt', map_location='cpu')
print("Keys:", checkpoint.keys())
print("Epoch:", checkpoint.get('epoch'))
```

#### 4. CUDA内存不足

**错误：** `RuntimeError: CUDA out of memory`

**解决：**
```bash
# 使用CPU模式
python inference_module.py --device cpu ...

# 或减小batch size
# 在config文件中设置 batch_size: 1
```

#### 5. HuggingFace连接错误

**错误：** `HTTPSConnectionPool: Failed to resolve 'huggingface.co'`

**解决：**
- 所有脚本已配置 `local_files_only=True`
- 确保模型文件已下载到 `models/` 目录
- 不需要网络连接

### 日志查看

```bash
# 查看最近的训练日志
tail -100 logs/info.log

# 查看错误日志
tail -100 logs/error.log

# 实时监控日志
tail -f logs/info.log
```

### 性能监控

```bash
# 监控GPU使用
watch -n 1 nvidia-smi

# 监控CPU和内存
htop

# 检查磁盘空间
df -h
```

---

## 总结

完成所有验收测试后，您应该有：

1. ✅ **Phase 1 训练产物**
   - `checkpoints/alignment/` 中的checkpoint文件
   - `mlruns/` 中的实验记录
   - 验收测试报告JSON

2. ✅ **Phase 2 训练产物**
   - `checkpoints/phase2/` 中的checkpoint文件（含LoRA）
   - MLflow中的Phase 2实验记录
   - 验收测试报告JSON

3. ✅ **推理结果**
   - `results/` 中的推理输出JSON文件
   - 验证过的多模态推理能力

4. ✅ **文档**
   - 验收测试报告
   - 性能指标记录
   - 问题和解决方案记录

---

## 下一步

验收测试通过后，可以进行：

1. **生产部署准备**
   - 优化模型大小
   - 配置推理服务
   - 设置监控和告警

2. **性能优化**
   - 模型量化
   - 推理加速
   - 批处理优化

3. **功能扩展**
   - 添加更多动物种类
   - 支持更多行为类别
   - 改进JSON输出格式

---

**文档版本：** 1.0  
**最后更新：** 2026-02-01  
**维护者：** Purr-Sight Team
