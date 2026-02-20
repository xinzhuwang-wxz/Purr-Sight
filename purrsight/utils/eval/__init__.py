"""
评估工具模块

提供模型评估相关的工具和指标。
"""

from .eval_align import ContrastiveMetrics

__all__ = [
    "ContrastiveMetrics",
]
