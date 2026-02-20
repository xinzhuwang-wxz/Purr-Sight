"""
语义对齐模块：基于ImageBind设计的多模态对比学习

将各模态特征映射到统一语义空间，实现跨模态语义对齐。

架构流程：
1. 输入：编码器输出的512维特征
2. 投影头：各模态独立投影，512维 → 512维（匹配LLM embedding维度）
3. 后处理器：L2归一化 + 温度缩放
4. 输出：统一语义空间嵌入（512维，已L2 normalize，匹配LLM embedding维度）
5. 对比损失：InfoNCE损失进行跨模态对比学习

关键特性：
- 对齐模块输出512维，直接匹配LLM embedding维度，无需投影层
- 每个模态作为独立token输入LLM，不融合
- LLM通过attention机制自行学习模态间的融合权重

参考ImageBind和UnIVAL设计，支持模态缺失的情况。
"""

from .projection_head import ProjectionHead, ProjectionHeads
from .postprocessor import (
    LearnableLogitScaling,
    Normalize,
    Postprocessors,
)
from .contrastive_loss import infonce_loss, ContrastiveLoss
from .aligner import ContrastiveAligner
__all__ = [
    # 投影头
    "ProjectionHead",
    "ProjectionHeads",
    # 后处理器
    "LearnableLogitScaling",
    "Normalize",
    "Postprocessors",
    # 对比损失
    "infonce_loss",
    "ContrastiveLoss",
    # 对齐器
    "ContrastiveAligner",
]
