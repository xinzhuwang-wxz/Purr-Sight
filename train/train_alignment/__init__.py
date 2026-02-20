"""
多模态对齐训练模块

使用 PyTorch Lightning 实现对比学习训练框架。
"""

from .train_align_conf import AlignmentConfig
from .dataset import AlignmentDataset
from .lightning_module import ContrastiveAlignmentModule
from .train import train_model, train_loop_per_worker

__all__ = [
    "AlignmentConfig",
    "AlignmentDataset",
    "ContrastiveAlignmentModule",
    "train_model",
    "train_loop_per_worker",
]