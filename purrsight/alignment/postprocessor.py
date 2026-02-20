"""
后处理器模块：L2归一化和温度缩放

参考ImageBind设计，所有模态都经过L2归一化，部分模态使用可学习的温度缩放。

包含：
- LearnableLogitScaling: 可学习的logit缩放类
- Normalize: L2归一化层类
- Postprocessors: 多模态后处理器集合类
- _get_device_from_features: 辅助函数，从特征字典获取设备

参考仓库：
- openai/CLIP (OpenCLIP)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from purrsight.config import Modality


def _get_device_from_features(features: Dict[str, torch.Tensor]) -> torch.device:
    """
    从特征字典中获取设备
    
    Args:
        features: 特征字典
    
    Returns:
        特征所在的设备，如果字典为空则返回CPU设备
    """
    return next(iter(features.values())).device if features else torch.device("cpu")


class LearnableLogitScaling(nn.Module):
    """
    可学习的logit缩放（温度参数）
    
    参考ImageBind设计，使用可学习的logit scale作为温度参数。
    在InfoNCE损失中，相似度矩阵会乘以exp(logit_scale)。
    
    Attributes:
        logit_scale: logit scale参数（Parameter或Buffer）
        learnable: 是否可学习
    """
    
    def __init__(
        self,
        logit_scale_init: float = 1.0,
        learnable: bool = True,
    ):
        """
        初始化可学习的logit缩放
        
        Args:
            logit_scale_init: logit scale的初始值，默认1.0
                通常使用log(1/0.07) ≈ 2.66作为初始值（CLIP风格）
            learnable: 是否可学习，默认True
        """
        super().__init__()
        if learnable:
            # 使用log空间初始化，避免数值不稳定
            self.logit_scale = nn.Parameter(torch.ones([]) * logit_scale_init)
        else:
            # 固定值，不参与梯度更新
            self.register_buffer("logit_scale", torch.tensor(logit_scale_init))
        self.learnable = learnable
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        应用logit缩放
        
        Args:
            x: 输入特征，形状为(B, D)，dtype=float32
        
        Returns:
            缩放后的特征，形状为(B, D)，dtype=float32
        """
        return x * self.logit_scale.exp()


class Normalize(nn.Module):
    """
    L2归一化层
    
    将特征向量归一化为单位向量，使得点积等于余弦相似度。
    对于零向量，保持为零向量（不归一化）。
    
    Attributes:
        dim: 归一化的维度
        eps: 防止除零的小值
    """
    
    def __init__(self, dim: int = -1, eps: float = 1e-8):
        """
        初始化归一化层
        
        Args:
            dim: 归一化的维度，默认-1（最后一维）
            eps: 防止除零的小值，默认1e-8
        """
        super().__init__()
        self.dim = dim
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        L2归一化
        
        Args:
            x: 输入特征，形状为(B, D)，dtype=float32
        
        Returns:
            归一化后的特征，形状为(B, D)，dtype=float32
        """
        # 检查零向量：如果某个样本的L2范数接近0，归一化后可能产生NaN
        # 对于零向量，直接返回零向量（不归一化）
        norms = x.norm(p=2, dim=self.dim, keepdim=True)  # (B, 1)
        # 如果范数小于eps，说明是零向量，保持原样
        x_normalized = F.normalize(x, p=2, dim=self.dim, eps=self.eps)
        # 对于零向量，归一化后可能产生NaN，需要特殊处理
        zero_mask = norms.squeeze(-1) < self.eps  # (B,)
        if zero_mask.any():
            # 零向量保持为零向量
            x_normalized[zero_mask] = x[zero_mask]
        return x_normalized


class Postprocessors(nn.Module):
    """
    多模态后处理器集合
    
    为每个模态创建后处理器，包含L2归一化和可选的温度缩放。
    参考ImageBind设计，所有模态都经过L2归一化，部分模态使用温度缩放。
    """
    
    def __init__(
        self,
        use_temperature_scaling: bool = True,
        text_logit_scale_init: float = 2.66,  # log(1/0.07)
        text_learnable: bool = True,
        image_logit_scale_init: float = 1.0,
        image_learnable: bool = True,
        audio_logit_scale_init: float = 2.66,  # log(1/0.07)，与text相同
        audio_learnable: bool = False,
    ):
        """
        初始化多模态后处理器
        
        Args:
            use_temperature_scaling: 是否使用温度缩放，默认True
            text_logit_scale_init: 文本模态的logit scale初始值，默认2.66
            text_learnable: 文本模态的logit scale是否可学习，默认True
            image_logit_scale_init: 图像模态的logit scale初始值，默认1.0
            image_learnable: 图像模态的logit scale是否可学习，默认False
            audio_logit_scale_init: 音频模态的logit scale初始值，默认2.66
            audio_learnable: 音频模态的logit scale是否可学习，默认False
        """
        super().__init__()
        self.use_temperature_scaling = use_temperature_scaling
        
        # 所有模态都使用L2归一化
        self.normalize = Normalize(dim=-1)
        
        # 为每个模态创建温度缩放（如果启用）
        if use_temperature_scaling:
            self.text_scaling = LearnableLogitScaling(
                logit_scale_init=text_logit_scale_init,
                learnable=text_learnable,
            )
            self.image_scaling = LearnableLogitScaling(
                logit_scale_init=image_logit_scale_init,
                learnable=image_learnable,
            )
            self.audio_scaling = LearnableLogitScaling(
                logit_scale_init=audio_logit_scale_init,
                learnable=audio_learnable,
            )
        else:
            # 不使用温度缩放时，创建Identity层
            self.text_scaling = nn.Identity()
            self.image_scaling = nn.Identity()
            self.audio_scaling = nn.Identity()
    
    def forward(
        self,
        features: Dict[str, torch.Tensor],
        modality_presence: Optional[Dict[str, bool]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播：对特征进行L2归一化和温度缩放
        
        Args:
            features: 投影后的特征字典，键为模态名称，值为特征张量
                形状为(B, 512)，dtype=float32（匹配LLM embedding维度）
            modality_presence: 模态存在标记字典，键为模态名称，值为bool
                如果为None，则根据features中的键自动推断
        
        Returns:
            后处理后的特征字典，键为模态名称，值为特征张量
            形状为(B, 512)，已L2 normalize，dtype=float32（匹配LLM embedding维度）
            缺失的模态返回零向量
        """
        # 如果没有提供modality_presence，根据features自动推断
        if modality_presence is None:
            modality_presence = {
                modality.value: modality.value in features and features[modality.value] is not None
                for modality in [Modality.TEXT, Modality.IMAGE, Modality.AUDIO]
            }
        
        outputs = {}
        
        for modality in [Modality.TEXT, Modality.IMAGE, Modality.AUDIO]:
            modality_key = modality.value
            # 🔧 修复：如果modality_presence为False，直接返回零向量（不归一化）
            if not modality_presence.get(modality_key, False):
                # 模态不存在，返回零向量（不归一化）
                device = _get_device_from_features(features)
                batch_size = next(iter(features.values())).shape[0] if features else 1
                output_dim = next(iter(features.values())).shape[1] if features else 512
                outputs[modality_key] = torch.zeros(
                    batch_size, output_dim, device=device, dtype=torch.float32
                )
            elif modality_key in features and features[modality_key] is not None:
                x = features[modality_key]
                
                # 🔧 修复：L2归一化（Normalize类已处理零向量）
                x = self.normalize(x)
                outputs[modality_key] = x
            else:
                # 模态存在但特征为None，返回零向量（不归一化）
                device = _get_device_from_features(features)
                batch_size = next(iter(features.values())).shape[0] if features else 1
                output_dim = next(iter(features.values())).shape[1] if features else 512
                outputs[modality_key] = torch.zeros(
                    batch_size, output_dim, device=device, dtype=torch.float32
                )
        
        return outputs
    
    def get_logit_scales(self) -> Dict[str, torch.Tensor]:
        """
        获取各模态的logit scale值（用于损失计算）
        
        Returns:
            logit scale字典，键为模态名称（字符串），值为logit scale张量
        """
        if not self.use_temperature_scaling:
            return {}
        
        return {
            Modality.TEXT.value: self.text_scaling.logit_scale.exp(),
            Modality.IMAGE.value: self.image_scaling.logit_scale.exp(),
            Modality.AUDIO.value: self.audio_scaling.logit_scale.exp(),
        }

