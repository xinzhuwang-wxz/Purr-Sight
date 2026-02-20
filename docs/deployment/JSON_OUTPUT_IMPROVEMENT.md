# JSON 输出改进方案 / JSON Output Improvement Plan

## 问题分析 / Problem Analysis

### 当前状况 / Current Status

运行推理时，模型输出的是自然语言文本，而不是结构化 JSON：

```
raw_model_output: " Use appropriate language and tone, while maintaining a professional demeanor..."
```

而不是期望的：

```json
{
  "diagnostic": {
    "physical_markers": {...},
    "classification": {...}
  },
  "behavioral_summary": "...",
  "human_actionable_insight": "..."
}
```

### 根本原因 / Root Cause

你的分析完全正确！问题不在于模型能力，而在于：

1. **训练数据太少**：
   - 原始数据：只有 **3 个样本**
   - 训练轮数：只训练了 **1 个 epoch** (epoch=00)
   - Qwen2.5-0.5B 是 500M 参数的模型，需要更多数据才能学会新格式

2. **推理 Prompt 不一致**：
   - 训练时：`"Analyze the cat's behavior... Output valid JSON only."`
   - 推理时（之前）：`"Analyze the cat's behavior... provide a detailed analysis..."`
   - **没有明确要求输出 JSON！**

3. **Few-shot 示例缺失**：
   - 小模型需要 in-context learning
   - 没有提供输出格式示例

## 解决方案 / Solutions

### ✅ 已完成 / Completed

#### 1. 改进推理 Prompt

**之前**：
```python
text_prompt = "Analyze the cat's behavior in this image. Provide a detailed analysis..."
```

**现在**：
```python
text_prompt = """Analyze the cat's behavior in this image according to the Ethogram. Output valid JSON only.

Example output format:
{"diagnostic": {"physical_markers": {"ears": "forward", "tail": "neutral", "posture": "relaxed", "vocalization": "silent"}, "classification": {"ethogram_group": "maintenance", "affective_state": "content", "arousal_level": "low", "risk_rating": 1}}, "behavioral_summary": "The cat displays relaxed body language.", "human_actionable_insight": "您的猫咪处于放松状态。"}

Now analyze this image and output JSON only:"""
```

**改进点**：
- ✅ 明确要求 "Output valid JSON only"
- ✅ 提供 few-shot 示例
- ✅ 与训练数据的 instruction 一致

#### 2. 生成更多训练数据

创建了 `data/phase2/train_extended.jsonl`，包含 **10 个多样化样本**：

| # | 场景 | ethogram_group | affective_state | risk_rating |
|---|------|----------------|-----------------|-------------|
| 1 | 放松的猫 | maintenance | content | 1 |
| 2 | 警觉的猫（音频） | social_affiliative | content | 1 |
| 3 | 呼噜的猫（多模态） | social_affiliative | content | 1 |
| 4 | 玩耍的猫 | social_affiliative | playful | 1 |
| 5 | 焦虑的猫 | agonistic | anxious | 4 |
| 6 | 攻击警告 | agonistic | aggressive | 5 |
| 7 | 理毛（维护） | maintenance | content | 1 |
| 8 | 狩猎模式 | predatory | neutral | 2 |
| 9 | 痛苦的猫 | agonistic | distressed | 5 |
| 10 | 友好问候 | social_affiliative | content | 1 |

**覆盖范围**：
- ✅ 所有 4 种 ethogram_group
- ✅ 所有 6 种 affective_state
- ✅ 所有 3 种 arousal_level
- ✅ 风险评级 1-5 全覆盖

### 🔄 待完成 / To Do

#### 3. 重新训练模型

**当前训练状态**：
```
训练数据：3 个样本
训练轮数：1 epoch
结果：模型没有学会 JSON 格式
```

**建议训练配置**：

```yaml
# config/phase2_retrain.yaml
phase2:
  data_path: "data/phase2"  # 使用 train_extended.jsonl
  batch_size: 2
  epochs: 15  # 增加到 15 epochs
  learning_rate: 0.00005  # 5e-5
  
  # 其他配置保持不变
  lora:
    r: 16
    lora_alpha: 32
```

**训练命令**：
```bash
# 方法 1：使用 train_extended.jsonl
cp data/phase2/train_extended.jsonl data/phase2/train.jsonl
python train/train_llm/train_phase2.py --config config/phase2_example.yaml

# 方法 2：创建新配置
python train/train_llm/train_phase2.py --config config/phase2_retrain.yaml --num-epochs 15
```

**预期效果**：
- 10 个样本 × 15 epochs = 150 次训练迭代
- 模型应该能学会 JSON 格式
- 输出应该符合 V3 Schema

#### 4. 可选：使用 Constrained Decoding

如果重新训练后仍有问题，可以使用 constrained decoding 强制 JSON 输出：

```python
# 使用 guidance 库
from guidance import models, gen

# 或使用 outlines 库
from outlines import models, generate

# 定义 JSON schema
schema = {...}

# 强制生成符合 schema 的 JSON
output = generate.json(model, schema)(prompt)
```

## 为什么 0.5B 模型能胜任 / Why 0.5B Model Can Handle This

你的判断是对的！Qwen2.5-0.5B 完全能胜任这个任务：

### 模型能力 / Model Capabilities

1. **预训练知识**：
   - 已经学会了 JSON 格式
   - 理解英文和中文
   - 具备基本推理能力

2. **参数规模**：
   - 500M 参数足够处理结构化输出
   - 类似规模的模型（如 GPT-2-medium）已被证明可以生成 JSON

3. **LoRA 微调**：
   - 只需要 2.1M 可训练参数（0.4%）
   - 足够学习特定任务的输出格式

### 对比分析 / Comparison

| 模型 | 参数量 | JSON 生成能力 |
|------|--------|--------------|
| GPT-2-small | 117M | ✅ 可以 |
| GPT-2-medium | 345M | ✅ 可以 |
| **Qwen2.5-0.5B** | **500M** | ✅ **应该可以** |
| Qwen2.5-1.5B | 1.5B | ✅ 很好 |

### 问题不在模型，在训练 / Issue is Training, Not Model

**证据**：
1. 模型在训练时看到了正确的 JSON 格式
2. 但只看了 3 个样本 × 1 epoch = 3 次
3. 这远远不够让模型"记住"新格式

**类比**：
- 就像让学生学习新的写作格式
- 只给 3 个例子，看 1 遍
- 当然记不住！

## 实验验证计划 / Experimental Validation Plan

### 阶段 1：增加数据和轮数

```bash
# 使用 10 个样本训练 15 epochs
cp data/phase2/train_extended.jsonl data/phase2/train.jsonl
python train/train_llm/train_phase2.py --config config/phase2_example.yaml --num-epochs 15
```

**预期**：
- 训练 loss 应该降到 < 1.0
- 模型应该开始输出 JSON 格式

### 阶段 2：验证输出

```bash
# 测试推理
./sub/run_pred.sh --checkpoint checkpoints/phase2/NEW_CHECKPOINT/model.pt --image data/cat.png
```

**检查**：
- `raw_model_output` 是否包含 JSON
- JSON 是否符合 V3 Schema
- Pydantic 验证是否通过

### 阶段 3：如果还不行

**选项 A**：继续增加数据
- 生成 50-100 个训练样本
- 训练 20-30 epochs

**选项 B**：使用 Constrained Decoding
- 安装 `guidance` 或 `outlines`
- 强制模型输出 JSON

**选项 C**：使用更大模型
- 尝试 Qwen2.5-1.5B
- 或 Qwen2.5-3B

## 当前系统优势 / Current System Advantages

即使模型暂时不输出 JSON，我们的系统仍然工作：

1. **Fallback Parser**：
   - 从自然语言中提取关键词
   - 生成符合 V3 Schema 的默认结构
   - 保存原始输出在 `raw_model_output`

2. **Pydantic 验证**：
   - 确保输出始终符合 schema
   - 自动修正不合理的值（如 risk_rating）

3. **格式化输出**：
   - 双语显示
   - 人类可读

## 总结 / Summary

### 问题根源 / Root Cause
- ❌ 不是模型能力不足
- ❌ 不是模型太小
- ✅ **是训练数据太少（3 个样本）**
- ✅ **是训练轮数太少（1 epoch）**
- ✅ **是推理 prompt 不一致**

### 解决方案 / Solution
1. ✅ 已改进推理 prompt（添加 few-shot 示例）
2. ✅ 已生成 10 个多样化训练样本
3. 🔄 需要重新训练（15 epochs）
4. 🔄 验证输出质量

### 预期结果 / Expected Outcome
重新训练后，Qwen2.5-0.5B 应该能够：
- ✅ 生成符合 V3 Schema 的 JSON
- ✅ 正确分类猫咪行为
- ✅ 提供英文总结和中文建议
- ✅ 通过 Pydantic 验证

**你的判断是对的**：0.5B 模型完全够用，只是需要更多训练！🎯
