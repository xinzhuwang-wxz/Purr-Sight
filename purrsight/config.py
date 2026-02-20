"""
配置模块：定义项目配置常量、枚举和常量类

包含：
- 通用目录配置（ROOT_DIR, LOGS_DIR, CHECKPOINTS_DIR等）
- 模态类型枚举（Modality）
- 特征键常量（FeatureKey）
"""

import os
from pathlib import Path
from enum import Enum

# Directories
ROOT_DIR = Path(__file__).parent.parent.absolute()
"""
项目根目录

无论从哪个目录运行 Python 脚本，ROOT_DIR 都会正确指向项目根目录。
例如：/Users/physicsboy/Documents/GitHub/Purr-Sight
"""

LOGS_DIR = Path(ROOT_DIR, "logs")
"""日志目录"""
LOGS_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINTS_DIR = Path(ROOT_DIR, "checkpoints")
"""模型检查点保存目录"""
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path(ROOT_DIR, "data")
"""数据目录"""
DATA_DIR.mkdir(parents=True, exist_ok=True)

MODELS_DIR = Path(ROOT_DIR, "models")
"""预训练模型权重目录"""
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# MLflow 配置
MLFLOW_DIR = Path(ROOT_DIR, "mlruns")
"""MLflow 实验跟踪目录"""
MLFLOW_DIR.mkdir(parents=True, exist_ok=True)

MLFLOW_TRACKING_URI = f"file://{MLFLOW_DIR.absolute()}"
"""MLflow tracking URI（本地文件系统）"""

# 训练监控：统一用 MLflow 记录所有指标，checkpoint 仅按此单一 metric 保存，避免多种 monitor 混用
CHECKPOINT_MONITOR = "train_loss"
"""Checkpoint 保存时监听的指标（全项目统一，其余指标仅通过 MLflow 记录）"""
CHECKPOINT_MONITOR_MODE = "min"
"""监听到的指标方向：min=越小越好，max=越大越好"""

# 可选：云存储配置（如果需要）
# 参考示例：支持 EFS 或其他共享存储
# EFS_DIR = Path(f"/efs/shared_storage/purrsight/{os.environ.get('USER', '')}")
# try:
#     EFS_DIR.mkdir(parents=True, exist_ok=True)
#     MLFLOW_DIR = Path(EFS_DIR, "mlflow")
#     MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
#     MLFLOW_TRACKING_URI = f"file://{MLFLOW_DIR.absolute()}"
# except OSError:
#     # 如果无法访问共享存储，回退到本地目录
#     pass


class Modality(str, Enum):
    """
    输入模态类型枚举
    
    用于标识输入数据的原始类型。
    
    Attributes:
        TEXT: 文本模态
        IMAGE: 图像模态
        VIDEO: 视频模态。用于标识输入为video文件（在线预处理阶段）。
               视频会被分解为IMAGE和AUDIO特征，VIDEO信息保存在`_video_metadata`中。
               离线预处理后，VIDEO信息只存在于metadata中，不再作为独立的模态。
               在LLM模块中，VIDEO作为占位符token保留（保持固定4个tokens）。
        AUDIO: 音频模态
    """
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"  # 用于标识输入类型，视频会被分解为IMAGE和AUDIO
    AUDIO = "audio"

    def __str__(self) -> str:
        """
        返回枚举值的字符串表示。
        
        Returns:
            枚举值的字符串表示
        """
        return self.value


class ModalitySource(str, Enum):
    """
    模态数据来源枚举
    
    用于标识最终使用的数据来源，显式化优先级规则。
    
    Attributes:
        VIDEO: 数据来源于视频文件
        IMAGE: 数据来源于独立图像文件
        AUDIO: 数据来源于独立音频文件
        TEXT: 数据来源于文本
        NONE: 无数据来源
    """
    VIDEO = "video"
    IMAGE = "image"
    AUDIO = "audio"
    TEXT = "text"
    NONE = None

    def __str__(self) -> str:
        """
        返回枚举值的字符串表示。
        
        Returns:
            枚举值的字符串表示，NONE返回"none"
        """
        return self.value if self.value is not None else "none"


class FeatureKey:
    """
    输出特征键常量
    
    用于访问处理后的特征。
    
    Attributes:
        TEXT: 文本特征键
        TEXT_ATTENTION_MASK: 文本attention mask键
        IMAGE: 图像特征键（包括单帧图像和16帧视频帧）
        AUDIO: 音频特征键（包括独立音频和视频中的音频）
    
    注意：视频在预处理阶段被分解为IMAGE和AUDIO特征，不再有独立的VIDEO特征键。
    """
    TEXT = "text"
    TEXT_ATTENTION_MASK = "text_attention_mask"
    IMAGE = "image"
    AUDIO = "audio"
