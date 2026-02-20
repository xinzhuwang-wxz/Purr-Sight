#!/bin/bash
# PurrSight Conda 环境安装脚本

set -e  # 遇到错误立即退出

ENV_NAME="${1:-purrsight}"  # 默认环境名称为 purrsight
PYTHON_VERSION="${2:-3.10}"  # 默认 Python 版本为 3.10

echo "=========================================="
echo "PurrSight Conda 环境安装脚本"
echo "=========================================="
echo "环境名称: $ENV_NAME"
echo "Python 版本: $PYTHON_VERSION"
echo "=========================================="
echo ""

# 检查 conda 是否安装
if ! command -v conda &> /dev/null; then
    echo "❌ 错误: 未找到 conda 命令"
    echo "请先安装 Anaconda 或 Miniconda"
    exit 1
fi

# 检查环境是否已存在
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "⚠️  环境 '$ENV_NAME' 已存在"
    read -p "是否删除并重新创建? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "删除现有环境..."
        conda env remove -n $ENV_NAME -y
    else
        echo "使用现有环境..."
        conda activate $ENV_NAME
        pip install -e .
        echo "✅ 安装完成!"
        exit 0
    fi
fi

# 创建新环境
echo "📦 创建 conda 环境: $ENV_NAME (Python $PYTHON_VERSION)"
conda create -n $ENV_NAME python=$PYTHON_VERSION -y

# 激活环境（需要 source conda.sh）
echo "🔧 激活环境..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

# 安装系统依赖
echo "📥 安装系统依赖 (libgfortran, ffmpeg, jpeg)..."
conda install -c conda-forge libgfortran ffmpeg jpeg libjpeg-turbo -y

# 安装科学计算库
echo "📥 安装科学计算库 (numpy, scipy)..."
conda install -c conda-forge "numpy>=1.26.2" "scipy>=1.11.4" -y

# 安装 PyTorch（从 conda 安装更稳定）
echo "📥 安装 PyTorch..."
conda install pytorch torchvision torchaudio -c pytorch -y

# 安装项目依赖
echo "📥 安装项目依赖..."
pip install -e .

# 验证安装
echo ""
echo "=========================================="
echo "验证安装..."
echo "=========================================="

python -c "import torch; print(f'✅ PyTorch {torch.__version__}')" || echo "❌ PyTorch 导入失败"
python -c "import numpy; print(f'✅ NumPy {numpy.__version__}')" || echo "❌ NumPy 导入失败"
python -c "import torchvision; print(f'✅ TorchVision {torchvision.__version__}')" || echo "❌ TorchVision 导入失败"

echo ""
echo "=========================================="
echo "✅ 安装完成!"
echo "=========================================="
echo ""
echo "激活环境: conda activate $ENV_NAME"
echo "运行测试: python test/test_preprocess.py"
echo ""

