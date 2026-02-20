# Phase 2 训练数据说明

## 数据格式

Phase 2 数据遵循 `purrsight/LLM/prompts.py` 中定义的 **SYSTEM_PROMPT_V3** 和 **JSON_SCHEMA_V3** 规范。

### JSONL 格式

每行是一个JSON对象，包含以下字段：

```json
{
  "instruction": "用户指令（英文）",
  "response": "模型响应（JSON字符串，符合V3 Schema）",
  "image": "图像文件路径（可选）",
  "audio": "音频文件路径（可选）",
  "video": "视频文件路径（可选）"
}
```

### Response JSON Schema V3

Response字段是一个JSON字符串，解析后应符合以下结构：

```json
{
  "diagnostic": {
    "physical_markers": {
      "ears": "forward | sideways | flattened | alert",
      "tail": "neutral | tucked | lashing | upright | puffed",
      "posture": "relaxed | crouched | lateral_recumbent | arched | tense",
      "vocalization": "purr | hiss | growl | chirp | meow | trill | silent"
    },
    "classification": {
      "ethogram_group": "social_affiliative | agonistic | maintenance | predatory",
      "affective_state": "content | anxious | aggressive | playful | distressed | neutral",
      "arousal_level": "low | medium | high",
      "risk_rating": 1-5
    }
  },
  "behavioral_summary": "英文描述：客观描述视觉/听觉线索",
  "human_actionable_insight": "中文建议：给主人的专业建议"
}
```

## 数据生成方式

### 方式1：测试/演示数据（当前使用）

使用 `scripts/create_phase2_sample_data.py` 生成符合规范的示例数据：

```bash
# 生成3个示例（默认）
python scripts/create_phase2_sample_data.py --output data/phase2/train.jsonl

# 生成所有5个示例
python scripts/create_phase2_sample_data.py --output data/phase2/train.jsonl --num_samples 5
```

**优点**：
- 快速生成测试数据
- 格式完全符合规范
- 无需API调用

**缺点**：
- 数据量少，仅用于测试
- 响应内容是预定义的，不是真实标注

### 方式2：生产数据（推荐）

使用 `scripts/auto_label_phase2_data.py` 调用大模型API自动标注：

```bash
# 设置API密钥
export ARK_API_KEY="your_volcengine_api_key"

# 准备原始数据文件（JSONL格式，包含image/video/audio路径）
# 例如: data/raw/phase2_raw.jsonl

# 调用API自动标注
python scripts/auto_label_phase2_data.py \
  --input_file data/raw/phase2_raw.jsonl \
  --output_dir data/phase2
```

**支持的输入格式**：

```json
{"image": "path/to/image.jpg"}
{"video": "path/to/video.mp4"}
{"image": "path/to/image.jpg", "audio": "path/to/audio.wav"}
```

**API标注流程**：
1. 读取原始数据（图像/视频/音频路径）
2. 对于视频：自动提取关键帧（4帧）和音频轨道
3. 将多模态输入编码为base64
4. 调用豆包（Doubao）多模态大模型API
5. 使用 SYSTEM_PROMPT_V3 引导模型生成符合规范的JSON响应
6. 验证和清理JSON格式
7. 保存到输出文件

**优点**：
- 大规模自动标注
- 响应质量高，符合专业行为学标准
- 支持图像、视频、音频多模态

**配置**：
- 模型：`doubao-seed-1-8-251228`（或其他豆包模型）
- API Base URL：`https://ark.cn-beijing.volces.com/api/v3`
- 需要火山引擎（Volcengine）API密钥

## 当前数据集

### train.jsonl

- **样本数量**：3
- **生成方式**：`create_phase2_sample_data.py`
- **用途**：测试和演示Phase 2训练流程
- **模态分布**：
  - 图像only：1个
  - 音频only：1个
  - 图像+音频：1个

### 数据质量

所有响应均符合以下标准：
- ✅ 遵循 JSON_SCHEMA_V3 结构
- ✅ 使用规范的行为学术语（Ethogram）
- ✅ `behavioral_summary` 为英文
- ✅ `human_actionable_insight` 为中文
- ✅ `risk_rating` 逻辑正确（agonistic/distressed → 4-5分）

## 扩展数据集

### 准备原始数据

1. 收集猫咪的图像/视频/音频文件
2. 创建索引文件（JSONL格式）：

```bash
# 示例：为data/raw/目录下的所有图像创建索引
cat > data/raw/phase2_raw.jsonl << EOF
{"image": "data/raw/cat_001.jpg"}
{"image": "data/raw/cat_002.jpg"}
{"video": "data/raw/cat_video_001.mp4"}
EOF
```

### 调用API标注

```bash
export ARK_API_KEY="your_api_key"

python scripts/auto_label_phase2_data.py \
  --input_file data/raw/phase2_raw.jsonl \
  --output_dir data/phase2 \
  --api_key $ARK_API_KEY  # 可选，也可以用环境变量
```

### 合并数据集

```bash
# 合并多个标注文件
cat data/phase2/labeled_*.jsonl > data/phase2/train_full.jsonl

# 分割训练集和验证集（90/10）
head -n 900 data/phase2/train_full.jsonl > data/phase2/train.jsonl
tail -n 100 data/phase2/train_full.jsonl > data/phase2/val.jsonl
```

## 训练使用

```bash
# 使用Phase 2数据训练
python train/train_llm/train_phase2.py --config config/phase2_example.yaml
```

配置文件中指定数据路径：
```yaml
phase2:
  data_path: "data/phase2"  # 目录，包含train.jsonl和val.jsonl
```

## 数据验证

验证数据格式是否正确：

```python
import json

with open('data/phase2/train.jsonl', 'r') as f:
    for i, line in enumerate(f, 1):
        sample = json.loads(line)
        
        # 检查必需字段
        assert 'instruction' in sample
        assert 'response' in sample
        
        # 检查至少有一个模态
        assert any(k in sample for k in ['image', 'audio', 'video'])
        
        # 验证response是有效的JSON
        response = json.loads(sample['response'])
        
        # 验证Schema
        assert 'diagnostic' in response
        assert 'behavioral_summary' in response
        assert 'human_actionable_insight' in response
        
        print(f"✓ Sample {i} valid")
```

## 参考文档

- **System Prompt**: `purrsight/LLM/prompts.py` - SYSTEM_PROMPT_V3
- **JSON Schema**: `purrsight/LLM/prompts.py` - JSON_SCHEMA_V3
- **示例生成脚本**: `scripts/create_phase2_sample_data.py`
- **API标注脚本**: `scripts/auto_label_phase2_data.py`
- **训练脚本**: `train/train_llm/train_phase2.py`
