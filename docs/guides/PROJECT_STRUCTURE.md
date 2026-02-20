# Purr-Sight 项目结构

## 📁 目录结构

```
Purr-Sight/
├── sub/                          # 运行脚本目录
│   ├── run_train.sh             # 训练脚本（Phase 1 & 2）
│   ├── run_pred.sh              # 推理脚本
│   ├── cluster_train.sh         # 集群训练脚本
│   └── README.md                # 脚本使用说明
│
├── train/                        # 训练代码目录
│   ├── train_alignment/         # Phase 1: 对齐训练
│   │   ├── train.py            # 训练主脚本
│   │   ├── dataset.py          # 数据集
│   │   ├── lightning_module.py # Lightning模块
│   │   └── ...
│   │
│   ├── train_llm/               # Phase 2: LLM微调
│   │   ├── train_phase2.py     # 训练主脚本
│   │   ├── multimodal_llm_module.py  # 多模态LLM模块
│   │   ├── checkpoint_manager.py     # Checkpoint管理
│   │   ├── lora_manager.py           # LoRA管理
│   │   └── ...
│   │
│   ├── inference_module.py      # 推理模块
│   └── train_runner.py          # 训练运行器
│
├── makeindex/                    # 数据索引脚本
│   ├── phase1/                  # Phase 1数据处理
│   │   ├── makeindex_ESC-50.py
│   │   └── makeindex_Laion-sub.py
│   └── phase2/                  # Phase 2数据处理
│       └── merge_datasets.py
│
├── tests/                        # 测试代码
│   ├── property/                # 属性测试
│   ├── unit/                    # 单元测试
│   ├── acceptance_test_phase1.py
│   ├── acceptance_test_phase2.py
│   └── ...
│
├── purrsight/                    # 核心库
│   ├── LLM/                     # LLM相关
│   ├── encoder/                 # 编码器
│   ├── alignment/               # 对齐模块
│   ├── preprocess/              # 预处理
│   └── utils/                   # 工具函数
│
├── config/                       # 配置文件
│   ├── train_config.yaml        # Phase 1配置
│   ├── phase2_example.yaml      # Phase 2配置
│   └── ...
│
├── data/                         # 数据目录
│   ├── instruction/             # 指令数据
│   ├── preprocessed/            # 预处理数据
│   ├── test_alignment/          # 测试数据
│   └── ...
│
├── checkpoints/                  # Checkpoint目录
│   ├── alignment/               # Phase 1 checkpoints
│   └── phase2/                  # Phase 2 checkpoints
│
├── mlruns/                       # MLflow日志
├── logs/                         # 训练日志
├── results/                      # 推理结果
│
├── docs/                         # 文档目录
│   ├── tasks/                   # 任务文档
│   ├── ACCEPTANCE_TEST_GUIDE.md
│   ├── QUICK_START_ACCEPTANCE.md
│   └── ...
│
├── models/                       # 预训练模型
│   ├── Qwen2.5-0.5B-Instruct/
│   ├── mobilenetv4/
│   └── ...
│
├── .kiro/                        # Kiro配置
│   └── specs/                   # 规格文档
│
├── README.md                     # 项目说明
├── PROJECT_STRUCTURE.md          # 本文档
└── pyproject.toml               # Python项目配置
```

## 🎯 核心目录说明

### `sub/` - 运行脚本

统一的训练和推理入口点，方便集群提交。

**使用方法：**
```bash
# Phase 1训练
./sub/run_train.sh 1

# Phase 2训练
./sub/run_train.sh 2

# 推理
./sub/run_pred.sh --checkpoint <path> --image <path>
```

### `train/` - 训练代码

所有训练相关的代码都在这里。

**Phase 1 (train_alignment/):**
- 对齐训练（Contrastive Learning）
- 训练projection heads
- 输出：aligner checkpoint

**Phase 2 (train_llm/):**
- LLM微调（LoRA）
- 加载Phase 1 checkpoint
- 输出：完整模型checkpoint

**推理 (inference_module.py):**
- 多模态推理
- 支持视频/图片/文字输入
- 输出：JSON格式结果

### `makeindex/` - 数据处理

数据索引和预处理脚本。

**Phase 1:**
- ESC-50音频数据索引
- LAION图像数据索引

**Phase 2:**
- 数据集合并
- 多模态数据准备

### `tests/` - 测试代码

完整的测试框架。

**属性测试 (property/):**
- 使用Hypothesis进行属性测试
- 验证通用属性和不变量

**单元测试 (unit/):**
- 测试具体功能
- 边缘情况测试

**验收测试:**
- Phase 1验收测试
- Phase 2验收测试
- 端到端测试

### `purrsight/` - 核心库

可复用的核心功能。

**主要模块：**
- `LLM/` - LLM相关（projectors, prompts）
- `encoder/` - 多模态编码器
- `alignment/` - 对齐模块
- `preprocess/` - 数据预处理
- `utils/` - 工具函数

### `config/` - 配置文件

YAML格式的训练配置。

**主要配置：**
- `train_config.yaml` - Phase 1配置
- `phase2_example.yaml` - Phase 2配置
- `validation_config.yaml` - 验证配置

### `checkpoints/` - Checkpoint存储

训练产生的checkpoint。

**结构：**
```
checkpoints/
├── alignment/
│   └── <run_id>/
│       ├── aligner.pt          # Aligner权重
│       └── model.ckpt          # 完整模型
└── phase2/
    └── checkpoint_*.pt         # Phase 2 checkpoints
```

### `docs/` - 文档

项目文档和指南。

**主要文档：**
- 验收测试指南
- 快速开始指南
- 部署指南
- 任务文档

## 🔄 工作流程

### 1. 数据准备

```bash
# Phase 1数据索引
python makeindex/phase1/makeindex_ESC-50.py
python makeindex/phase1/makeindex_Laion-sub.py

# Phase 2数据准备
python makeindex/phase2/merge_datasets.py
```

### 2. Phase 1训练

```bash
./sub/run_train.sh 1 --epochs 20
```

**输出：**
- `checkpoints/alignment/<run_id>/aligner.pt`
- `mlruns/` - MLflow日志

### 3. Phase 2训练

```bash
./sub/run_train.sh 2 --epochs 10
```

**输入：** Phase 1的aligner.pt  
**输出：**
- `checkpoints/phase2/checkpoint_*.pt`
- `mlruns/` - MLflow日志

### 4. 推理

```bash
./sub/run_pred.sh \
    --checkpoint checkpoints/phase2/best.pt \
    --image data/cat.png
```

**输出：**
- `results/inference_*.json`

## 📊 数据流

```
原始数据 (data/)
    ↓
数据索引 (makeindex/)
    ↓
预处理数据 (data/preprocessed/)
    ↓
Phase 1训练 (train/train_alignment/)
    ↓
Aligner Checkpoint (checkpoints/alignment/)
    ↓
Phase 2训练 (train/train_llm/)
    ↓
完整模型 (checkpoints/phase2/)
    ↓
推理 (train/inference_module.py)
    ↓
结果 (results/)
```

## 🔧 开发指南

### 添加新功能

1. **核心功能** → `purrsight/`
2. **训练逻辑** → `train/`
3. **测试** → `tests/`
4. **配置** → `config/`

### 运行测试

```bash
# 单元测试
pytest tests/unit/

# 属性测试
pytest tests/property/

# 验收测试
python tests/acceptance_test_phase1.py
```

### 查看日志

```bash
# 训练日志
tail -f logs/info.log

# MLflow UI
mlflow ui --backend-store-uri file://./mlruns
```

## 📝 文件命名规范

### Python文件
- 训练脚本：`train_*.py`
- 测试文件：`test_*.py`
- 模块文件：小写+下划线

### Shell脚本
- 运行脚本：`run_*.sh`
- 工具脚本：`*.sh`

### 配置文件
- YAML配置：`*_config.yaml`
- 示例配置：`*_example.yaml`

### Checkpoint
- Phase 1：`aligner.pt`, `model.ckpt`
- Phase 2：`checkpoint_epoch{N}_step{M}.pt`

## 🚀 快速开始

```bash
# 1. 克隆项目
git clone <repo_url>
cd Purr-Sight

# 2. 安装依赖
conda env create -f environment.yml
conda activate purrsight

# 3. 运行Phase 1训练
./sub/run_train.sh 1

# 4. 运行Phase 2训练
./sub/run_train.sh 2

# 5. 推理测试
./sub/run_pred.sh --checkpoint checkpoints/phase2/best.pt --image data/cat.png
```

## 📞 获取帮助

- **运行脚本帮助：** `./sub/run_train.sh` 或 `./sub/run_pred.sh --help`
- **文档：** 查看 `docs/` 目录
- **测试：** 查看 `tests/README.md`

---

**更新日期：** 2026-02-01  
**版本：** 1.0
