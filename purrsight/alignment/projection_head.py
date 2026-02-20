"""
投影头模块：将各模态特征投影到统一语义空间

参考ImageBind设计，每个模态有独立的投影层，将不同维度特征投影到512维统一语义空间。
输出维度匹配LLM embedding维度，无需额外投影层。

包含：
- ProjectionHead: 单个模态的投影头类
- ProjectionHeads: 多模态投影头集合类

参考仓库：
- openai/CLIP (OpenCLIP)
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
from purrsight.config import Modality


class ProjectionHead(nn.Module):
    """
    单个模态的投影头
    
    结构：
    input_dim → LayerNorm → Linear(input_dim→output_dim, bias=False) → output_dim
    
    参考ImageBind和UnIVAL设计，使用简单的LayerNorm + Linear结构。
    每个模态有独立的投影层以学习模态特定的映射。
    输出512维，匹配LLM embedding维度。
    
    Attributes:
        ln: LayerNorm层
        fc: 全连接层（无偏置）
        input_dim: 输入特征维度
        output_dim: 输出特征维度
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        output_dim: int = 512,  # 512维，匹配LLM embedding维度
        eps: float = 1e-6,
    ):
        """
        初始化投影头
        
        Args:
            input_dim: 输入特征维度，默认512（编码器输出维度）
            output_dim: 输出特征维度，默认512（统一语义空间维度，匹配LLM embedding维度）
            eps: LayerNorm的epsilon参数，默认1e-6
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # LayerNorm → Linear (无偏置)
        # 512 → 512，匹配LLM embedding维度
        self.ln = nn.LayerNorm(normalized_shape=input_dim, eps=eps)
        self.fc = nn.Linear(input_dim, output_dim, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：投影特征到统一维度
        
        Args:
            x: 输入特征，形状为(B, input_dim)，dtype=float32
        
        Returns:
            投影后的特征，形状为(B, output_dim)，dtype=float32
        """
        # LayerNorm → Linear
        x = self.ln(x)  # (B, input_dim)
        x = self.fc(x)  # (B, output_dim)
        return x


class ProjectionHeads(nn.Module):
    """
    多模态投影头集合

    为每个模态创建独立的投影头，将各模态特征投影到512维统一语义空间（匹配LLM embedding维度）。
    支持不同输入维度：Text(384) → 512, Image(feature_dim) → 512, Audio(2048) → 512
    支持模态缺失的情况（缺失模态返回零向量）。
    
    注意：视频在预处理阶段被分解为Image和Audio，使用对应的投影头处理。
    """
    
    def __init__(
        self,
        text_input_dim: int = 384,    # Text encoder output: 384
        image_input_dim: int = 512,   # Image encoder output: feature_dim
        audio_input_dim: int = 2048,  # Audio encoder output: 2048
        output_dim: int = 512,        # 512维，匹配LLM embedding维度
        eps: float = 1e-6,
    ):
        """
        初始化多模态投影头

        Args:
            text_input_dim: 文本输入特征维度，默认384（Text encoder输出）
            image_input_dim: 图像输入特征维度，默认512（Image encoder输出）
            audio_input_dim: 音频输入特征维度，默认2048（Audio encoder输出）
            output_dim: 输出特征维度，默认512（统一语义空间维度，匹配LLM embedding维度）
            eps: LayerNorm的epsilon参数，默认1e-6
        """
        super().__init__()
        self.output_dim = output_dim

        # 为每个模态创建独立的投影头（不同输入维度）
        self.text_head = ProjectionHead(text_input_dim, output_dim, eps)
        self.image_head = ProjectionHead(image_input_dim, output_dim, eps)
        self.audio_head = ProjectionHead(audio_input_dim, output_dim, eps)
        
    def forward(
        self,
        features: Dict[str, torch.Tensor],
        modality_presence: Optional[Dict[str, bool]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播：投影各模态特征到统一维度

        Args:
            features: 编码器输出的特征字典，键为模态名称，值为特征张量
                Text: (B, 384), Image: (B, feature_dim), Audio: (B, 2048)
                注意：Video已分解为Image和Audio，不再有独立的Video特征
                dtype=float32
            modality_presence: 模态存在标记字典，键为模态名称，值为bool
                如果为None，则根据features中的键自动推断

        Returns:
            投影后的特征字典，键为模态名称，值为特征张量
            形状为(B, 512)，dtype=float32（匹配LLM embedding维度）
            缺失的模态返回零向量
        """
        # 如果没有提供modality_presence，根据features自动推断
        if modality_presence is None:
            modality_presence = {
                modality.value: modality.value in features and features[modality.value] is not None
                for modality in [Modality.TEXT, Modality.IMAGE, Modality.AUDIO]
            }
        
        outputs = {}
        batch_size = None
        
        # 确定批次大小（从第一个存在的模态）
        for modality in [Modality.TEXT, Modality.IMAGE, Modality.AUDIO]:
            modality_key = modality.value
            if modality_presence.get(modality_key, False) and modality_key in features:
                if features[modality_key] is not None:
                    batch_size = features[modality_key].shape[0]
                    break
        
        if batch_size is None:
            raise ValueError("所有模态都缺失，无法确定批次大小")
        
        device = None
        for modality in [Modality.TEXT, Modality.IMAGE, Modality.AUDIO]:
            modality_key = modality.value
            if modality_presence.get(modality_key, False) and modality_key in features:
                if features[modality_key] is not None:
                    x = features[modality_key]
                    device = x.device

                    # 投影（各模态使用对应的投影头和维度验证）
                    if modality == Modality.TEXT:
                        expected_dim = self.text_head.ln.normalized_shape[0]
                        if x.dim() != 2 or x.shape[1] != expected_dim:
                            raise ValueError(
                                f"{modality_key}特征形状错误：期望(B, {expected_dim})，"
                                f"实际{x.shape}"
                            )
                        outputs[modality_key] = self.text_head(x)
                    elif modality == Modality.IMAGE:
                        expected_dim = self.image_head.ln.normalized_shape[0]
                        if x.dim() != 2 or x.shape[1] != expected_dim:
                            raise ValueError(
                                f"{modality_key}特征形状错误：期望(B, {expected_dim})，"
                                f"实际{x.shape}"
                            )
                        outputs[modality_key] = self.image_head(x)
                    elif modality == Modality.AUDIO:
                        expected_dim = self.audio_head.ln.normalized_shape[0]
                        if x.dim() != 2 or x.shape[1] != expected_dim:
                            raise ValueError(
                                f"{modality_key}特征形状错误：期望(B, {expected_dim})，"
                                f"实际{x.shape}"
                            )
                        outputs[modality_key] = self.audio_head(x)
                else:
                    # 模态存在但特征为None，返回零向量
                    if device is None:
                        device = next(self.parameters()).device
                    outputs[modality_key] = torch.zeros(
                        batch_size, self.output_dim, device=device, dtype=torch.float32
                    )
            else:
                # 模态缺失，返回零向量
                if device is None:
                    device = next(self.parameters()).device
                outputs[modality_key] = torch.zeros(
                    batch_size, self.output_dim, device=device, dtype=torch.float32
                )
        
        return outputs

