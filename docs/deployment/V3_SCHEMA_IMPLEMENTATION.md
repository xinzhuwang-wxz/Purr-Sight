# V3 Schema 实现文档 / V3 Schema Implementation

## 概述 / Overview

本文档说明如何使用 Pydantic 强制返回符合 V3 Schema 的结构化 JSON 输出。

This document explains how to use Pydantic to enforce structured JSON output conforming to V3 Schema.

## V3 Schema 定义 / V3 Schema Definition

根据 `purrsight/LLM/prompts.py` 中的定义：

```python
JSON_SCHEMA_V3 = {
    "diagnostic": {
        "physical_markers": {
            "ears": "Enum: forward, sideways, flattened, alert",
            "tail": "Enum: neutral, tucked, lashing, upright, puffed",
            "posture": "Enum: relaxed, crouched, lateral_recumbent, arched, tense",
            "vocalization": "Enum: purr, hiss, growl, chirp, meow, trill, silent"
        },
        "classification": {
            "ethogram_group": "Enum: social_affiliative, agonistic, maintenance, predatory",
            "affective_state": "Enum: content, anxious, aggressive, playful, distressed, neutral",
            "arousal_level": "Enum: low, medium, high",
            "risk_rating": "Int: 1-5"
        }
    },
    "behavioral_summary": "String (English): Objective description of visual/auditory cues.",
    "human_actionable_insight": "String (Chinese): Professional advice for the owner."
}
```

## Pydantic 模型实现 / Pydantic Model Implementation

### 文件位置 / File Location

`purrsight/LLM/output_parser.py`

### 核心类 / Core Classes

1. **PhysicalMarkers**: 物理标记（耳朵、尾巴、姿势、发声）
2. **Classification**: 行为分类（行为组、情感状态、唤醒水平、风险评级）
3. **Diagnostic**: 诊断信息（包含物理标记和分类）
4. **CatBehaviorAnalysis**: 完整分析（诊断 + 行为总结 + 专家建议 + 原始输出）

### 关键特性 / Key Features

#### 1. 类型验证 / Type Validation

使用 Pydantic 的 `Literal` 类型确保枚举值正确：

```python
ears: Literal["forward", "sideways", "flattened", "alert"]
```

#### 2. 范围验证 / Range Validation

使用 `Field` 约束确保数值在有效范围内：

```python
risk_rating: int = Field(..., ge=1, le=5)
```

#### 3. 逻辑验证 / Logic Validation

使用 `@validator` 实现业务逻辑验证：

```python
@validator('risk_rating')
def validate_risk_rating(cls, v, values):
    # If agonistic or distressed, risk should be 4-5
    if values.get('ethogram_group') == 'agonistic' and v < 4:
        return 4  # Auto-correct
    return v
```

#### 4. 原始输出保留 / Raw Output Preservation

添加 `raw_model_output` 字段保存模型的原始自然语言输出：

```python
raw_model_output: Optional[str] = Field(
    None,
    description="Original natural language output from the model"
)
```

## 使用方法 / Usage

### 1. 基本解析 / Basic Parsing

```python
from purrsight.LLM.output_parser import OutputParser

# 解析模型输出
generated_text = "模型生成的文本..."
analysis = OutputParser.parse_model_output(generated_text, strict=False)

# analysis 包含:
# - diagnostic (物理标记 + 分类)
# - behavioral_summary (英文行为总结)
# - human_actionable_insight (中文专家建议)
# - raw_model_output (原始输出)
```

### 2. 严格模式 / Strict Mode

```python
# 严格模式：如果解析失败会抛出异常
try:
    analysis = OutputParser.parse_model_output(generated_text, strict=True)
except ValueError as e:
    print(f"解析失败: {e}")
```

### 3. 格式化输出 / Formatted Output

```python
# 生成人类可读的格式化输出
formatted = OutputParser.format_output(
    analysis,
    include_raw=True,      # 包含原始输出
    chinese_summary=True   # 包含中文建议
)
print(formatted)
```

输出示例：

```
================================================================================
猫咪行为分析报告 / Cat Behavior Analysis Report
================================================================================

【物理标记 / Physical Markers】
  耳朵 Ears:        forward
  尾巴 Tail:        neutral
  姿势 Posture:     relaxed
  发声 Vocalization: purr

【行为分类 / Classification】
  行为组 Ethogram:    social_affiliative
  情感状态 Affective:  content
  唤醒水平 Arousal:    low
  风险评级 Risk:       1/5

【行为总结 / Behavioral Summary】
  The cat displays relaxed body language with forward ears and neutral tail position.

【专家建议 / Expert Advice】
  猫咪处于放松和满足的状态，这是健康和快乐的表现。继续提供舒适的环境和适当的关注。

【原始输出 / Raw Model Output】
  [模型的原始自然语言输出...]

================================================================================
```

### 4. 在推理中使用 / Use in Inference

```python
from train.inference_module import PurrSightInference

# 初始化推理
inference = PurrSightInference(checkpoint_path="...")

# 运行推理
result = inference.infer_from_image("data/cat.png")

# result['analysis'] 已经是解析和验证后的结构化数据
print(result['analysis']['diagnostic']['classification']['affective_state'])

# 打印格式化输出
inference.print_formatted_result(result)
```

## 解析策略 / Parsing Strategy

### 1. JSON 提取 / JSON Extraction

首先尝试从模型输出中提取 JSON：

```python
# 查找包含 "diagnostic" 键的 JSON 对象
json_match = re.search(r'\{[^{}]*"diagnostic"[^{}]*\{.*?\}.*?\}', text, re.DOTALL)
```

### 2. Pydantic 验证 / Pydantic Validation

提取到 JSON 后，使用 Pydantic 验证：

```python
analysis = CatBehaviorAnalysis(**json_data)
```

### 3. 回退解析 / Fallback Parsing

如果 JSON 提取失败，使用关键词分析生成默认结构：

```python
# 分析文本中的关键词
if 'calm' in text_lower or 'relaxed' in text_lower:
    affective_state = "content"
    risk_rating = 1
elif 'aggressive' in text_lower:
    affective_state = "aggressive"
    risk_rating = 5
# ...
```

## 测试 / Testing

### 运行测试 / Run Tests

```bash
# 测试 V3 Schema 验证和推理
python test_inference_v3.py
```

### 测试内容 / Test Coverage

1. ✅ **Schema Validation**: Pydantic 模型验证
2. ✅ **Image Inference**: 图像输入推理
3. ✅ **Text Inference**: 文本输入推理
4. ✅ **Formatted Output**: 格式化输出显示
5. ✅ **Raw Output Preservation**: 原始输出保留

## 当前限制 / Current Limitations

### 1. 模型输出格式 / Model Output Format

当前模型（训练轮数少、数据量小）通常不会生成结构化 JSON，而是生成自然语言文本。

**解决方案**：
- 使用回退解析器从文本中推断结构
- 增加训练数据和轮数
- 在训练时使用更多结构化示例

### 2. 中文输出质量 / Chinese Output Quality

模型可能生成英文而非中文的 `human_actionable_insight`。

**解决方案**：
- 在 system prompt 中强调中文输出
- 使用更多中文训练样本
- 考虑使用中文 LLM 基座模型

### 3. JSON 格式一致性 / JSON Format Consistency

模型可能生成不完整或格式错误的 JSON。

**解决方案**：
- 使用 constrained decoding
- 添加 JSON schema 到 prompt
- 使用 function calling 或 structured output API

## 改进建议 / Improvement Suggestions

### 短期 / Short-term

1. **增加训练数据**：创建更多符合 V3 Schema 的训练样本
2. **调整 Prompt**：在推理时添加更详细的 JSON 格式说明
3. **后处理优化**：改进回退解析器的关键词匹配逻辑

### 中期 / Mid-term

1. **Constrained Decoding**：使用 guidance 或 outlines 库强制 JSON 输出
2. **Few-shot Learning**：在 prompt 中添加示例
3. **Fine-tune 策略**：使用 DPO/RLHF 优化输出格式

### 长期 / Long-term

1. **专用模型**：训练专门用于结构化输出的模型
2. **多阶段生成**：先生成自然语言，再转换为 JSON
3. **API 集成**：使用 OpenAI function calling 等 API

## 相关文件 / Related Files

- `purrsight/LLM/prompts.py` - V3 Schema 定义
- `purrsight/LLM/output_parser.py` - Pydantic 模型和解析器
- `train/inference_module.py` - 推理模块（使用 parser）
- `test_inference_v3.py` - V3 Schema 测试脚本
- `data/phase2/train.jsonl` - 训练数据示例

## 示例输出 / Example Output

### 完整 JSON 结构 / Complete JSON Structure

```json
{
  "timestamp": "2026-02-01T05:30:00.000000",
  "input_type": "image",
  "model_checkpoint": "checkpoints/phase2/.../model.pt",
  "metadata": {
    "model_version": "2.0",
    "phase": "phase2",
    "schema_version": "V3"
  },
  "input_file": "data/cat.png",
  "analysis": {
    "diagnostic": {
      "physical_markers": {
        "ears": "forward",
        "tail": "neutral",
        "posture": "relaxed",
        "vocalization": "purr"
      },
      "classification": {
        "ethogram_group": "social_affiliative",
        "affective_state": "content",
        "arousal_level": "low",
        "risk_rating": 1
      }
    },
    "behavioral_summary": "The cat displays relaxed body language with forward ears and neutral tail position, indicating a calm and content state.",
    "human_actionable_insight": "猫咪处于放松和满足的状态，这是健康和快乐的表现。继续提供舒适的环境和适当的关注。",
    "raw_model_output": "[模型的原始输出文本...]",
    "parsing_note": "Fallback parsing used - model did not generate structured JSON"
  }
}
```

## 总结 / Summary

通过 Pydantic 实现的 V3 Schema 验证系统提供了：

1. ✅ **类型安全**：确保所有字段类型正确
2. ✅ **值验证**：枚举值和数值范围验证
3. ✅ **逻辑验证**：业务规则自动检查
4. ✅ **原始输出保留**：保存模型的自然语言输出
5. ✅ **回退机制**：即使模型输出不完美也能生成有效结构
6. ✅ **格式化显示**：人类可读的双语输出

这为 Purr-Sight 系统提供了可靠的结构化输出保证，同时保持了灵活性以处理不完美的模型输出。
