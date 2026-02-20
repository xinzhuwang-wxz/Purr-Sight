"""对齐器模块：整合投影头和后处理器，实现多模态语义对齐。

参考 ImageBind 和 UnIVAL 设计，将编码器输出的特征投影到统一语义空间，
并进行 L2 归一化和温度缩放。输出维度匹配 LLM embedding 维度。

Typical usage example:

  aligner = ContrastiveAligner(output_dim=512)
  aligned_features = aligner(encoder_outputs)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Union
from purrsight.config import Modality
from .projection_head import ProjectionHeads
from .postprocessor import Postprocessors


def _ensure_tensor_and_device(
    value: Union[torch.Tensor, np.ndarray],
    model_device: torch.device,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """确保输入是 tensor 并移动到正确的 device。

    Args:
        value: 输入值，可以是 torch.Tensor 或 np.ndarray。
        model_device: 模型所在的 device。
        dtype: 目标数据类型。如果为 None，则根据 numpy dtype 自动推断。

    Returns:
        转换后的 tensor，位于正确的 device 上。

    Raises:
        TypeError: 如果输入类型不是 Tensor 或 ndarray。
    """
    if isinstance(value, np.ndarray):
        # 转换为tensor
        if dtype is None:
            if value.dtype == np.int64:
                tensor = torch.from_numpy(value).long()
            elif value.dtype == np.int32:
                tensor = torch.from_numpy(value).int()
            elif value.dtype in (np.float32, np.float64):
                tensor = torch.from_numpy(value).float()
            else:
                tensor = torch.from_numpy(value.astype(np.float32)).float()
        else:
            tensor = torch.from_numpy(value).to(dtype)
    elif isinstance(value, torch.Tensor):
        tensor = value
    else:
        raise TypeError(
            f"不支持的类型：{type(value)}，期望 torch.Tensor 或 np.ndarray"
        )
    
    # 移动到正确的device
    tensor = tensor.to(model_device)
    
    return tensor


def _normalize_encoder_outputs(
    encoder_outputs: Dict[str, Union[torch.Tensor, np.ndarray]],
    model_device: torch.device,
) -> Dict[str, torch.Tensor]:
    """规范化编码器输出：将 numpy array 转换为 tensor 并移动到正确的 device。

    Args:
        encoder_outputs: 编码器输出字典，值可以是 torch.Tensor 或 np.ndarray。
        model_device: 模型所在的 device。

    Returns:
        规范化后的字典，所有值都是 torch.Tensor 且位于正确的 device 上。
    """
    normalized_outputs = {}
    for key, value in encoder_outputs.items():
        if value is not None:
            normalized_outputs[key] = _ensure_tensor_and_device(
                value, model_device, dtype=torch.float32
            )
        else:
            normalized_outputs[key] = None
    
    return normalized_outputs


class ContrastiveAligner(nn.Module):
    """对比学习对齐器。

    整合投影头和后处理器，实现多模态特征到统一语义空间的映射。

    架构流程：
    1. 输入：编码器输出的特征（支持 numpy array 或 tensor）
    2. 输入验证：自动将 numpy array 转换为 tensor
    3. 设备管理：确保所有输入都在模型所在的 device 上
    4. 投影头：各模态独立投影，不同维度 -> 512维（匹配 LLM embedding 维度）
    5. 后处理器：L2 归一化 + 温度缩放
    6. 输出：统一语义空间嵌入（512维，已 L2 normalize）

    Attributes:
        projection_heads: 投影头模块。
        postprocessors: 后处理器模块。
        output_dim: 输出特征维度。
        use_temperature_scaling: 是否使用温度缩放。
    """
    
    def __init__(
        self,
        text_input_dim: int = 384,
        image_input_dim: int = 512,
        audio_input_dim: int = 2048,
        output_dim: int = 512,
        use_temperature_scaling: bool = True,
        text_logit_scale_init: float = 1.0,
        text_learnable: bool = False,
        image_logit_scale_init: float = 2.66,
        image_learnable: bool = False,
        audio_logit_scale_init: float = 2.66,
        audio_learnable: bool = False,
    ):
        """初始化对比学习对齐器。

        Args:
            text_input_dim: 文本输入特征维度，默认 384 (Text encoder 输出)。
            image_input_dim: 图像输入特征维度，默认 512 (Image encoder 输出)。
            audio_input_dim: 音频输入特征维度，默认 2048 (Audio encoder 输出)。
            output_dim: 输出特征维度，默认 512 (统一语义空间维度，匹配 LLM embedding 维度)。
            use_temperature_scaling: 是否使用温度缩放，默认 True。
            text_logit_scale_init: 文本模态的 logit scale 初始值，默认 2.66。
            text_learnable: 文本模态的 logit scale 是否可学习，默认 False。
            image_logit_scale_init: 图像模态的 logit scale 初始值，默认 1.0。
            image_learnable: 图像模态的 logit scale 是否可学习，默认 False。
            audio_logit_scale_init: 音频模态的 logit scale 初始值，默认 2.66。
            audio_learnable: 音频模态的 logit scale 是否可学习，默认 False。
        """
        super().__init__()
        self.output_dim = output_dim
        self.use_temperature_scaling = use_temperature_scaling

        # 投影头：各模态独立投影，不同维度 -> 512维
        self.projection_heads = ProjectionHeads(
            text_input_dim=text_input_dim,
            image_input_dim=image_input_dim,
            audio_input_dim=audio_input_dim,
            output_dim=output_dim,
        )

        # 后处理器：L2归一化 + 温度缩放
        self.postprocessors = Postprocessors(
            use_temperature_scaling=use_temperature_scaling,
            text_logit_scale_init=text_logit_scale_init,
            text_learnable=text_learnable,
            image_logit_scale_init=image_logit_scale_init,
            image_learnable=image_learnable,
            audio_logit_scale_init=audio_logit_scale_init,
            audio_learnable=audio_learnable,
        )
    
    def forward(
        self,
        encoder_outputs: Dict[str, Union[torch.Tensor, np.ndarray]],
        modality_presence: Optional[Dict[str, bool]] = None,
    ) -> Dict[str, torch.Tensor]:
        """前向传播：将编码器输出映射到统一语义空间。

        Args:
            encoder_outputs: 编码器输出的特征字典。
                键为模态名称，值为特征张量或 numpy 数组。
                Text: (B, 384), Image: (B, feature_dim), Audio: (B, 2048)。
                dtype=float32，支持 numpy.ndarray，会自动转换为 torch.Tensor 并移动到模型 device。
                注意：视频已分解为 Image 和 Audio，不再有独立的 Video 特征。
            modality_presence: 模态存在标记字典。
                键为模态名称，值为 bool。
                如果为 None，则根据 encoder_outputs 中的键自动推断。

        Returns:
            统一语义空间嵌入字典。
            键为模态名称，值为特征张量。
            形状为 (B, 512)，已归一化，dtype=float32。
            缺失的模态返回零向量。
        """
        # 获取模型所在的device
        model_device = next(self.parameters()).device
        
        # 输入验证和device管理
        normalized_outputs = _normalize_encoder_outputs(
            encoder_outputs, model_device
        )
        
        # 1. 投影
        projected_features = self.projection_heads(
            normalized_outputs, modality_presence
        )
        
        # 2. 后处理
        aligned_embeddings = self.postprocessors(
            projected_features, modality_presence
        )
        
        return aligned_embeddings
    
    def get_logit_scales(self) -> Dict[str, torch.Tensor]:
        """获取各模态的 logit scale 值（用于损失计算）。

        Returns:
            logit scale 字典，键为模态名称，值为 logit scale 张量。
        """
        return self.postprocessors.get_logit_scales()
