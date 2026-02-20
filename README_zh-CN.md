# Purr-Sight: 端侧轻量级多模态 AI

<div align="center">

![Purr-Sight Logo](docs/assets/logo.png)

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Code Style: Google](https://img.shields.io/badge/code%20style-google-blueviolet.svg)](https://github.com/google/styleguide/blob/gh-pages/pyguide.md)

**端侧实时猫咪情绪与行为理解系统**

[English](README.md) • [功能特性](#-功能特性) • [架构设计](#-架构设计) • [快速开始](#-快速开始) • [训练指南](#-训练指南) • [路线图](#-路线图)

</div>

---

## 📖 项目介绍

**Purr-Sight** 是一个端侧优先的轻量级多模态 AI 系统，专为实时解码猫咪的情绪与行为而设计。不同于只能“看”的传统监控系统，Purr-Sight 通过融合视觉信号（如耳朵位置）和音频信号（如哈气声或呼噜声），实现了真正的“理解”。

该系统专为 **Raspberry Pi 5** 和 **NVIDIA Jetson** 等边缘设备打造，采用紧凑的 **0.6B 参数** 架构，实现了毫秒级推理，让高级宠物监控既能保护隐私，又触手可及。

## 🚀 功能特性

*   **多模态理解**：融合 **图像** (MobileNetV4)、**音频** (PANNs) 和 **文本** (MiniLM) 特征，精准捕捉“飞机耳”或“低吼”等细微线索。
*   **端侧优化**：专为 ARM 架构设计，推理延迟 <200ms。模型总体积 <1GB。
*   **故障安全设计 (Fail-Safe)**：对缺失模态（如摄像头损坏或麦克风静音）具有鲁棒性。系统永不崩溃，只会自动适应。
*   **两阶段训练流**：
    1.  **对齐阶段 (Phase 1)**：利用对比学习 (InfoNCE) 将视觉和音频特征对齐到统一的语义空间。
    2.  **指令微调 (Phase 2)**：将对齐后的特征投影到轻量级 LLM (MatFormer-OLMo-0.5B)，生成结构化的 JSON 报告。

## 🏗️ 架构设计

Purr-Sight 采用非对称的三塔架构，通过对比对齐机制进行统一。

```mermaid
graph TD
    %% Data Ingress
    subgraph Data [数据入口 & 预处理]
        Raw[原始输入] -->|FFmpeg| Pre[预处理器]
        Pre -->|Tokenize| T_Dat[文本]
        Pre -->|Resize| I_Dat[图像 (224px)]
        Pre -->|Mel Spec| A_Dat[音频 (Log-Mel)]
    end

    %% Encoders
    subgraph Encoders [冻结编码器]
        T_Dat -->|MiniLM| T_Enc[文本编码器 (384d)]
        I_Dat -->|MobileNetV4| I_Enc[图像编码器 (960d)]
        A_Dat -->|PANNs| A_Enc[音频编码器 (2048d)]
    end

    %% Phase 1
    subgraph Phase1 [Phase 1: 语义对齐]
        T_Enc & I_Enc & A_Enc -->|投影头| Shared[共享空间 (512d)]
        Shared -->|InfoNCE Loss| Aligned[对齐特征]
    end

    %% Phase 2
    subgraph Phase2 [Phase 2: 指令微调]
        Aligned -->|Linear-GELU| Adapter[多模态投影器]
        Adapter -->|Soft Prompts| LLM[LLM (OLMo-0.5B)]
        LLM -->|生成| Output[JSON 报告]
    end
```

### 核心组件

1.  **编码器 (Encoders)**：
    *   **图像**：MobileNetV4-ConvLarge (高精度/延迟比)。
    *   **音频**：PANNs (CNN14) 用于鲁棒的声音事件检测。
    *   **文本**：MiniLM-L6-v2 用于高效语义嵌入。
2.  **对齐器 (Aligner - Phase 1)**：通过可学习的投影头和温度缩放 InfoNCE 损失，将异构特征 (384d/960d/2048d) 映射到统一的 512d 超球面。
3.  **投影器 (Projector - Phase 2)**：一个简单的 MLP 适配器，将对齐后的特征转换为 LLM 可理解的 Soft Prompt Token。

## 🏁 快速开始

### 前置要求

*   Python 3.8+
*   FFmpeg (用于视频/音频处理)
*   兼容 CUDA 的 GPU (推荐用于训练) 或 Apple Silicon (MPS)

### 安装步骤

1.  **克隆仓库：**
    ```bash
    git clone https://github.com/physicsboy/Purr-Sight.git
    cd Purr-Sight
    ```

2.  **安装依赖：**
    ```bash
    pip install -r requirements.txt
    ```

3.  **下载预训练权重：**
    将模型权重放置在 `models/` 目录下：
    *   `models/mobilenetv4/`
    *   `models/panns/`
    *   `models/mini-lm-l6-h384-uncased/`
    *   `models/Qwen2.5-0.5B-Instruct/` (或您选择的其他 LLM)

## 🏋️‍♂️ 训练指南

Purr-Sight 使用统一入口脚本 `run_train.sh` 管理两个阶段的训练。配置由 `config/train_config.yaml` 统一管理。

### 1. 数据准备 (离线预处理)

**关键步骤**：为了最大化训练速度，请将原始视频/音频数据预处理为 Tensor。

```bash
# 处理原始 JSONL 数据
python -m purrsight.preprocess.prepre \
  --input_file data_formal_alin/align_v0.jsonl \
  --output_dir data_formal_alin/preprocessed \
  --num_workers 8
```

### 2. Phase 1: 对齐训练

将编码器对齐到共享语义空间。

*   **目标**：最小化匹配的 图像-文本 和 音频-文本 对之间的 InfoNCE 损失。
*   **配置**：查看 `config/train_config.yaml` 中的 `phase1` 部分。

```bash
./run_train.sh 1
```

**输出**：
- 日志：`mlflow ui` (http://localhost:5000)
- 检查点：`outputs/alignment_phase1_{timestamp}/checkpoints/`
  - `aligner.pt`：**重要！** Phase 2 需要此文件。

### 3. Phase 2: 指令微调

连接对齐后的编码器与 LLM，生成文本描述。

**⚠️ 过渡步骤**：
在运行 Phase 2 之前，您必须更新 `config/train_config.yaml` 以指向您训练好的 Phase 1 权重。

1.  找到您的 Phase 1 输出文件：`outputs/alignment_phase1_.../checkpoints/aligner.pt`
2.  编辑 `config/train_config.yaml`：
    ```yaml
    phase2:
      # ...
      adapter_path: "outputs/alignment_phase1_20260126_XXXXXX/checkpoints/aligner.pt"
    ```

*   **目标**：微调 Projector (及可选微调 LLM/LoRA) 以适应指令-响应对。
*   **配置**：查看 `config/train_config.yaml` 中的 `phase2` 部分。

```bash
./run_train.sh 2
```

## 📂 项目结构

```
Purr-Sight/
├── config/                 # 配置文件
│   └── train_config.yaml   # 统一训练配置
├── data/                   # 数据存储
├── models/                 # 预训练模型权重
├── purrsight/              # 核心库
│   ├── alignment/          # Phase 1: 对比学习对齐器
│   ├── encoder/            # 图像/音频/文本编码器
│   ├── LLM/                # Phase 2: 投影器 & 模型
│   ├── preprocess/         # 数据预处理逻辑
│   └── utils/              # 日志 & 工具
├── train/                  # 训练脚本
│   ├── train_alignment/    # Phase 1 训练逻辑
│   └── train_llm/          # Phase 2 训练逻辑
├── run_train.sh            # 统一训练入口脚本
└── train_runner.py         # 训练调度器
```

## 🛣️ 路线图 (Roadmap)

*   [x] **MVP**：单模态编码器 (Image/Text)
*   [x] **Phase 1**：多模态对齐 (Image/Audio/Text) 与 InfoNCE
*   [x] **Phase 2**：LLM 集成与指令微调
*   [ ] **优化**：INT8 量化以适应端侧推理
*   [ ] **部署**：ONNX Runtime / TensorRT 导出
*   [ ] **应用**：实时监控手机 App

## 🤝 贡献指南

欢迎贡献代码！请遵循 [Google Python 风格指南](https://zh-google-styleguide.readthedocs.io/en/latest/google-python-styleguide/python_style_rules/)。

1.  Fork 本项目
2.  创建特性分支 (`git checkout -b feature/AmazingFeature`)
3.  提交更改 (`git commit -m 'Add some AmazingFeature'`)
4.  推送到分支 (`git push origin feature/AmazingFeature`)
5.  提交 Pull Request

## 📄 许可证

本项目基于 Apache License 2.0 许可证分发。详情请参阅 `LICENSE` 文件。

---
<div align="center">
  <sub>Built with ❤️ for 🐱 by PhysicsBoy</sub>
</div>
