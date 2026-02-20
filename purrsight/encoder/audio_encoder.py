"""
音频编码器：基于本地集成的PANNs-CNN14（无第三方库依赖）

输入：64×256梅尔频谱（单通道，16kHz采样率）
输出：2048维音频特征
本地权重路径：./models/panns/cnn14_light_16k.pth

包含：
- _AudioEncoder: 音频编码器类

参考仓库：
- qiuqiangkong/audioset_tagging_cnn (PANNs)
"""

import torch
import torch.nn as nn
from purrsight.config import ROOT_DIR
from pathlib import Path

try:
    from .panns_cnn14 import Cnn14
except ImportError:
    from purrsight.encoder.panns_cnn14 import Cnn14


class _AudioEncoder(nn.Module):
    """
    PANNs-CNN14-light音频编码器，输出2048维特征
    
    Attributes:
        model: Cnn14模型实例
        weight_path: 权重文件路径
    """
    
    def __init__(self, weight_dir: str = "models/panns", weight_name: str = "cnn14_light_16k.pth"):
        """
        初始化音频编码器
        
        Args:
            weight_dir: 权重文件目录，默认models/panns
            weight_name: 权重文件名，默认cnn14_light_16k.pth
        
        Raises:
            FileNotFoundError: 当权重文件不存在时
        """
        super().__init__()
        self.weight_path = Path(ROOT_DIR, weight_dir, weight_name)

        self.model = Cnn14()
        self._load_local_weights()
        self.eval()  


    def _load_local_weights(self):
        """
        加载官方权重，缺失时提供下载命令
        """
        if not self.weight_path.exists():
            download_cmd = (
                f"mkdir -p {self.weight_path.parent}\n"
                f"wget -O {self.weight_path} "
                f"https://zenodo.org/record/3987831/files/Cnn14_16k_mAP%3D0.438.pth?download=1"
            )
            raise FileNotFoundError(f"PANNs权重文件不存在！请运行以下命令下载：\n{download_cmd}")

        # 检查文件大小（预期约350MB）
        file_size = self.weight_path.stat().st_size / (1024 * 1024)  # MB
        if file_size < 300:  # 如果小于300MB，可能不完整
            raise FileNotFoundError(
                f"权重文件可能损坏或不完整！\n"
                f"  文件路径: {self.weight_path}\n"
                f"  文件大小: {file_size:.2f} MB (预期约350MB)\n"
                f"  请重新下载：\n"
                f"  mkdir -p {self.weight_path.parent}\n"
                f"  wget -O {self.weight_path} "
                f"https://zenodo.org/record/3987831/files/Cnn14_16k_mAP%3D0.438.pth?download=1"
            )

        try:
            loaded_dict = torch.load(self.weight_path, map_location="cpu", weights_only=False)
        except EOFError as e:
            raise EOFError(
                f"权重文件损坏或不完整！\n"
                f"  文件路径: {self.weight_path}\n"
                f"  文件大小: {file_size:.2f} MB\n"
                f"  错误: {e}\n"
                f"  请删除损坏的文件并重新下载：\n"
                f"  rm {self.weight_path}\n"
                f"  mkdir -p {self.weight_path.parent}\n"
                f"  wget -O {self.weight_path} "
                f"https://zenodo.org/record/3987831/files/Cnn14_16k_mAP%3D0.438.pth?download=1"
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
        
        model_dict = self.model.state_dict()
        
        if 'model' in loaded_dict:
            state_dict = loaded_dict['model']
        else:
            state_dict = loaded_dict
        
        weight_keys = list(state_dict.keys())
        model_keys = list(model_dict.keys())
        
        prefix = None
        if weight_keys and not any(k in state_dict for k in model_keys):
            for p in ['model.', 'module.', '']:
                if any(k.startswith(p) and k[len(p):] in model_keys for k in weight_keys):
                    prefix = p
                    break
        
        if prefix:
            mapped_state_dict = {}
            for k in weight_keys:
                if k.startswith(prefix):
                    mapped_key = k[len(prefix):]
                    if mapped_key in model_dict:
                        mapped_state_dict[mapped_key] = state_dict[k]
                else:
                    mapped_state_dict[k] = state_dict[k]
            state_dict = mapped_state_dict
        
        filtered_state_dict = {}
        model_keys_set = set(model_dict.keys())
        for k, v in state_dict.items():
            if k.startswith(('spectrogram_extractor.', 'logmel_extractor.', 'fc_audioset.')):
                continue
            if k in model_keys_set:
                filtered_state_dict[k] = v
        
        self.model.load_state_dict(filtered_state_dict, strict=False)  


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：梅尔频谱→2048维特征

        Args:
            x: 预处理后的梅尔频谱，形状(64, 256)或(B, 64, 256)
               需满足：16kHz采样率，64梅尔bins，256时间步

        Returns:
            2048维音频特征，形状(B, 2048)
        """
        with torch.no_grad():
            x = self.model(x)
        return x

