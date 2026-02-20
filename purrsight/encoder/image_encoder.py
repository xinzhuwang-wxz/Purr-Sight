"""
图像编码器：基于timm库的MobileNetV4-small实现

模型名称：mobilenetv4_conv_small.e2400_r224_in1k
输出特征维度：feature_dim维（动态，取决于模型）
本地权重路径：./models/mobilenetv4/pytorch_model.bin

包含：
- _ImageEncoder: 图像编码器类

参考仓库：
- timm (MobileNetV4)
"""

import torch
import torch.nn as nn
from timm import create_model
from purrsight.config import ROOT_DIR
from pathlib import Path


class _ImageEncoder(nn.Module):
    """
    图像编码器：基于MobileNetV4-small提取特征

    实现逻辑：
    1. 加载模型结构（包含分类头，但不加载分类头权重）
    2. 使用forward_features()跳过分类头，获取倒数第二层（特征层）输出
    3. 全局平均池化，输出feature_dim维特征向量
    
    Attributes:
        model: MobileNetV4模型实例
        feature_dim: 特征维度（动态推断）
        weight_path: 权重文件路径
    """
    
    def __init__(self, model_name: str = "mobilenetv4_conv_small.e2400_r224_in1k", weight_dir: str = "models/mobilenetv4"):
        """
        初始化图像编码器
        
        Args:
            model_name: 模型名称，默认mobilenetv4_conv_small.e2400_r224_in1k
            weight_dir: 权重文件目录，默认models/mobilenetv4
        
        Raises:
            FileNotFoundError: 当权重文件不存在时
        """
        super().__init__()
        
        # 优先使用safetensors格式，如果不存在则使用pytorch_model.bin
        weight_dir_path = Path(ROOT_DIR, weight_dir)
        safetensors_path = weight_dir_path / "model.safetensors"
        pytorch_bin_path = weight_dir_path / "pytorch_model.bin"
        
        if safetensors_path.exists():
            self.weight_path = safetensors_path
            self.use_safetensors = True
        elif pytorch_bin_path.exists():
            self.weight_path = pytorch_bin_path
            self.use_safetensors = False
        else:
            raise FileNotFoundError(
                f"权重文件不存在！\n"
                f"  查找路径: {weight_dir_path}\n"
                f"  期望文件: model.safetensors 或 pytorch_model.bin"
            )

        self.model = create_model(model_name, pretrained=False, features_only=False)
        self._load_local_weights()
        self._remove_classifier_head()
        self.feature_dim = self._get_feature_dim()
        self.eval()

    def _load_local_weights(self):
        """
        加载本地权重：只加载特征层权重，跳过分类头权重
        
        Raises:
            FileNotFoundError: 当权重文件不存在时
            EOFError: 当权重文件损坏或不完整时
        """
        if not self.weight_path.exists():
            raise FileNotFoundError(f"权重文件不存在！请下载至：\n{self.weight_path}\n")

        # Check file size (MobileNetV4 weights are approximately 20-30MB)
        file_size = self.weight_path.stat().st_size / (1024 * 1024)  # MB
        if file_size < 10:  # If less than 10MB, it might be incomplete
            raise FileNotFoundError(
                f"权重文件可能损坏或不完整！\n"
                f"  文件路径: {self.weight_path}\n"
                f"  文件大小: {file_size:.2f} MB (预期约20-30MB)\n"
                f"  请删除损坏的文件并重新下载。"
            )

        try:
            if self.use_safetensors:
                # 使用safetensors加载
                from safetensors.torch import load_file
                state_dict = load_file(self.weight_path)
            else:
                # 使用torch.load加载
                state_dict = torch.load(self.weight_path, map_location="cpu", weights_only=False)
        except EOFError as e:
            raise EOFError(
                f"权重文件损坏或不完整！\n"
                f"  文件路径: {self.weight_path}\n"
                f"  文件大小: {file_size:.2f} MB\n"
                f"  错误: {e}\n"
                f"  请删除损坏的文件并重新下载。"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"加载权重文件失败！\n"
                f"  文件路径: {self.weight_path}\n"
                f"  文件大小: {file_size:.2f} MB\n"
                f"  错误类型: {type(e).__name__}\n"
                f"  错误信息: {e}\n"
                f"  请检查文件是否完整或重新下载。"
            ) from e
        
        filtered_dict = {k: v for k, v in state_dict.items() if not k.startswith(("classifier.", "head."))}
        self.model.load_state_dict(filtered_dict, strict=False)
    
    def _remove_classifier_head(self):
        """
        移除分类头层，确保模型只用于特征提取
        """
        if hasattr(self.model, 'classifier'):
            delattr(self.model, 'classifier')
        if hasattr(self.model, 'head'):
            delattr(self.model, 'head')
        if hasattr(self.model, 'fc'):
            delattr(self.model, 'fc')

    def _get_feature_dim(self) -> int:
        """
        获取特征维度：通过前向传播推断特征层输出维度
        
        Returns:
            特征维度（通道数），例如960（MobileNetV4-small）
        """
        with torch.no_grad():
            test_input = torch.randn(1, 3, 224, 224)
            features = self.model.forward_features(test_input)
            return features.shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：提取feature_dim维特征

        Args:
            x: 输入图像张量，形状为(B, 3, 224, 224)，dtype=float32

        Returns:
            feature_dim维图像特征，形状为(B, feature_dim)，dtype=float32
        """
        with torch.no_grad():
            x = self.model.forward_features(x)
            x = x.mean(dim=[2, 3])
        return x  

