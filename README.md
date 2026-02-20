# Purr-Sight: Lightweight Multimodal AI for Edge Devices

<div align="center">

![Purr-Sight Logo](docs/assets/logo.png)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Code Style: Google](https://img.shields.io/badge/code%20style-google-blueviolet.svg)](https://github.com/google/styleguide/blob/gh-pages/pyguide.md)

**Real-time Cat Emotion & Behavior Understanding on the Edge**

[中文版](README_zh-CN.md) • [Features](#key-features) • [Architecture](#architecture) • [Getting Started](#getting-started) • [Training](#training) • [Roadmap](#roadmap)

</div>

---

## 📖 Introduction

**Purr-Sight** is an edge-first, lightweight multimodal AI system designed to decode your cat's emotions and behaviors in real-time. Unlike traditional surveillance systems that only "see," Purr-Sight "understands" by fusing visual cues (like ear position) with audio signals (like hissing or purring).

Built for **Raspberry Pi 5** and **NVIDIA Jetson**, it achieves millisecond-level inference with a compact **0.6B parameter** architecture, making advanced pet monitoring privacy-preserving and accessible.

## 🚀 Key Features

*   **Multimodal Understanding**: Fuses **Image** (MobileNetV4), **Audio** (PANNs), and **Text** (MiniLM) to detect subtle cues like "airplane ears" or "growling."
*   **Edge-Optimized**: Designed for ARM architectures with <200ms latency. Total model size <1GB.
*   **Fail-Safe Design**: Robust to missing modalities (e.g., broken camera or muted mic). It never crashes, just adapts.
*   **Two-Stage Training**:
    1.  **Alignment (Phase 1)**: Aligns visual and audio features into a shared semantic space using Contrastive Learning (InfoNCE).
    2.  **Instruction Tuning (Phase 2)**: Projects aligned features to a lightweight LLM (MatFormer-OLMo-0.5B) to generate structured JSON reports.

## 🏗️ Architecture

Purr-Sight employs an asymmetric three-tower architecture unified by a contrastive alignment mechanism.

```mermaid
graph TD
    %% Data Ingress
    subgraph Data [Data Ingress]
        Raw[Raw Input] -->|FFmpeg| Pre[Preprocessor]
        Pre -->|Tokenize| T_Dat[Text]
        Pre -->|Resize| I_Dat[Image (224px)]
        Pre -->|Mel Spec| A_Dat[Audio (Log-Mel)]
    end

    %% Encoders
    subgraph Encoders [Frozen Encoders]
        T_Dat -->|MiniLM| T_Enc[Text Enc (384d)]
        I_Dat -->|MobileNetV4| I_Enc[Image Enc (960d)]
        A_Dat -->|PANNs| A_Enc[Audio Enc (2048d)]
    end

    %% Phase 1
    subgraph Phase1 [Phase 1: Alignment]
        T_Enc & I_Enc & A_Enc -->|Projectors| Shared[Shared Space (512d)]
        Shared -->|InfoNCE Loss| Aligned[Aligned Features]
    end

    %% Phase 2
    subgraph Phase2 [Phase 2: Instruction Tuning]
        Aligned -->|Linear-GELU| Adapter[Multimodal Projector]
        Adapter -->|Soft Prompts| LLM[LLM (OLMo-0.5B)]
        LLM -->|Generation| Output[JSON Report]
    end
```

### Core Components

1.  **Encoders**:
    *   **Image**: MobileNetV4-ConvLarge (High accuracy/latency ratio).
    *   **Audio**: PANNs (CNN14) for robust sound event detection.
    *   **Text**: MiniLM-L6-v2 for efficient semantic embedding.
2.  **Aligner (Phase 1)**: Maps heterogeneous features (384d/960d/2048d) to a unified 512d hypersphere using learnable projection heads and temperature-scaled InfoNCE loss.
3.  **Projector (Phase 2)**: A simple MLP adapter that translates aligned features into soft prompt tokens for the LLM.

### 🔍 Code Audit & Optimizations (Latest Update)

We have performed a comprehensive "health check" on the codebase to ensure industrial-grade robustness:

*   **Tensor Shape Consistency**: Verified rigorous dimension mapping: `Encoders -> Aligner (512d) -> Projector (2048d hidden) -> LLM (1024d)`.
*   **Gradient Isolation**: Implemented strict freezing logic (`module.eval()`) for Encoders and Aligner in Phase 2 to prevent BatchNorm statistics drift and accidental gradient leaks.
*   **Data Pipeline Efficiency**: Added "Lazy Loading" and "Safety Buffers" in `InstructionDataset` to handle large JSONL files and prevent token truncation.
*   **Redundancy Removal**: Optimized `PurrSightMMLLM.forward` to skip audio encoding for empty inputs, saving compute resources.

## 🏁 Getting Started

### Prerequisites

*   Python 3.8+
*   FFmpeg (for video/audio processing)
*   CUDA-compatible GPU (recommended for training) or Apple Silicon (MPS)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/physicsboy/Purr-Sight.git
    cd Purr-Sight
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Download Pre-trained Weights:**
    Place model weights in the `models/` directory:
    *   `models/mobilenetv4/`
    *   `models/panns/`
    *   `models/mini-lm-l6-h384-uncased/`
    *   `models/Qwen1.5-1.8B-Chat/` (or your chosen LLM)

## 🏋️‍♂️ Training

Purr-Sight uses a unified entry point `run_train.sh` for both training phases. Configuration is managed via `config/train_config.yaml`.

### 1. Data Preparation (Offline Preprocessing)

**Crucial Step**: To maximize training speed, pre-process your raw video/audio data into tensors.

```bash
# Process raw JSONL data
python -m purrsight.preprocess.prepre \
  --input_file data_formal_alin/align_v0.jsonl \
  --output_dir data_formal_alin/preprocessed \
  --num_workers 8
```

### 2. Phase 1: Alignment Training

Aligns the encoders into a shared semantic space.

*   **Goal**: Minimize InfoNCE loss between matched Image-Text and Audio-Text pairs.
*   **Config**: Check `phase1` section in `config/train_config.yaml`.

```bash
./run_train.sh 1
```

**Output**:
- Logs: `mlflow ui` (http://localhost:5000)
- Checkpoints: `outputs/alignment_phase1_{timestamp}/checkpoints/`
  - `aligner.pt`: **Important!** This file is needed for Phase 2.

### 3. Phase 2: Instruction Tuning

Connects the aligned encoders to the LLM to generate text descriptions.

**⚠️ Transition Step**:
Before running Phase 2, you MUST update `config/train_config.yaml` to point to your trained Phase 1 weights.

1.  Locate your Phase 1 output file: `outputs/alignment_phase1_.../checkpoints/aligner.pt`
2.  Edit `config/train_config.yaml`:
    ```yaml
    phase2:
      # ...
      adapter_path: "outputs/alignment_phase1_20260126_XXXXXX/checkpoints/aligner.pt"
    ```

*   **Goal**: Fine-tune the Projector (and optionally LLM via LoRA) on instruction-response pairs.
*   **Config**: Check `phase2` section in `config/train_config.yaml`.

```bash
./run_train.sh 2
```

## 📂 Project Structure

```
Purr-Sight/
├── config/                 # Configuration files
│   └── train_config.yaml   # Unified training config
├── data/                   # Data storage
├── models/                 # Pre-trained model weights
├── purrsight/              # Core library
│   ├── alignment/          # Phase 1: Contrastive Aligner
│   ├── encoder/            # Image/Audio/Text Encoders
│   ├── LLM/                # Phase 2: Projector & Model
│   ├── preprocess/         # Data preprocessing logic
│   └── utils/              # Logging & helpers
├── train/                  # Training scripts
│   ├── train_alignment/    # Phase 1 training logic
│   └── train_llm/          # Phase 2 training logic
├── run_train.sh            # Unified training entry point
└── train_runner.py         # Training dispatcher
```

## 🛣️ Roadmap

*   [x] **MVP**: Single-modality encoders (Image/Text)
*   [x] **Phase 1**: Multimodal Alignment (Image/Audio/Text) with InfoNCE
*   [x] **Phase 2**: LLM Integration & Instruction Tuning
*   [ ] **Optimization**: INT8 Quantization for Edge Inference
*   [ ] **Deployment**: ONNX Runtime / TensorRT export
*   [ ] **App**: Mobile app for real-time monitoring

## 🤝 Contributing

Contributions are welcome! Please follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---
<div align="center">
  <sub>Built with ❤️ for 🐱 by Maxen</sub>
</div>
