"""
对齐训练配置模块

定义对齐训练的配置参数（训练相关的超参数和设置）。

包含：
- AlignmentConfig: 对齐训练配置类

注意：通用目录配置在 purrsight.config 中定义。
"""

from dataclasses import dataclass
from typing import Optional
from pathlib import Path

from purrsight.config import CHECKPOINTS_DIR, MLFLOW_TRACKING_URI


@dataclass
class AlignmentConfig:
    """
    对齐训练配置
    
    定义对齐训练的所有超参数和设置。
    
    Attributes:
        data_path: 数据文件路径（必需）
        batch_size: 批次大小
        num_workers: DataLoader的worker数量
        val_split: 验证集比例
        use_preprocessed: 是否使用离线预处理
        preprocessed_dir: 预处理文件目录
        input_dim: 输入特征维度
        output_dim: 输出特征维度
        use_temperature_scaling: 是否使用温度缩放
        epochs: 训练轮数
        learning_rate: 学习率
        weight_decay: 权重衰减
        warmup_steps: 预热步数
        device: 设备选择（"auto", "cpu", "cuda", "mps"）
        save_dir: 检查点保存目录
        save_every: 每N个epoch保存一次
        mlflow_tracking_uri: MLflow tracking URI
        experiment_name: MLflow实验名称
        log_every: 每N步记录一次指标
    """

    # 数据
    data_path: str
    batch_size: int = 32
    num_workers: int = 4  # 编码器在 LightningModule 中，可以安全使用多进程
    # 注意：编码器在 LightningModule 中初始化一次，不在 Dataset 中，所以多进程不会重复加载
    # 🔧 性能优化：推荐设置为4-8（根据CPU核心数调整），0表示单线程（仅用于调试）
    val_split: float = 0.1
    
    # 预处理模式
    use_preprocessed: bool = False  # 是否使用离线预处理文件
    preprocessed_dir: Optional[str] = None  # 预处理文件目录，如果为None且use_preprocessed=True，默认使用data/preprocessed

    # 模型
    input_dim: int = 512
    output_dim: int = 512
    use_temperature_scaling: bool = True

    # 训练
    epochs: int = 10
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 1000

    # 设备
    device: str = "auto"

    # 保存（使用通用配置的默认值）
    save_dir: Optional[str] = None  # 如果为None，使用 CHECKPOINTS_DIR / "alignment"
    save_every: int = 1
    
    # 验证指标计算
    compute_val_metrics: bool = True  # 是否计算验证指标（O(B²)复杂度，batch_size大时较慢）
    val_metrics_every_n_epochs: int = 1  # 🔧 优化：每N个epoch计算一次验证指标（减少计算频率）

    # MLflow（使用通用配置的默认值）
    mlflow_tracking_uri: Optional[str] = None  # 如果为None，使用 purrsight.config.MLFLOW_TRACKING_URI
    experiment_name: str = "alignment_training"
    log_every: int = 100

    def __post_init__(self):
        """初始化后处理：设置默认值"""
        if self.save_dir is None:
            self.save_dir = str(CHECKPOINTS_DIR / "alignment")
        
        if self.mlflow_tracking_uri is None:
            self.mlflow_tracking_uri = MLFLOW_TRACKING_URI
        
        # 预处理模式默认值
        if self.use_preprocessed and self.preprocessed_dir is None:
            # 如果启用离线预处理但未指定目录，默认使用data/preprocessed
            from purrsight.config import ROOT_DIR
            self.preprocessed_dir = str(ROOT_DIR / "data" / "preprocessed")