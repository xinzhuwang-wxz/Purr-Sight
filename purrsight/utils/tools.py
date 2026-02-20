"""
工具函数模块：提供设备检测、tensor转换、seeds设置等通用工具函数

包含：
- get_available_device: 自动检测最佳可用设备
- to_tensor: 将numpy数组转换为PyTorch tensor
- set_seeds: 设置随机种子以确保可复现性
"""

from typing import Dict, Union
import numpy as np
import torch
import random
import os


def get_available_device() -> str:
    """
    自动检测并返回最佳可用设备，优先级：MPS > CPU > CUDA
    
    Returns:
        设备字符串："mps", "cpu" 或 "cuda"
    
    Example:
        >>> device = get_available_device()
    """
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def to_tensor(features: Dict[str, np.ndarray], device: Union[str, torch.device, None] = None) -> Dict[str, torch.Tensor]:
    """
    将预处理后的特征字典（numpy数组）转换为PyTorch tensor，用于模型输入
    
    Args:
        features: 特征字典，值为numpy数组
        device: 目标设备，可选：
            - None: 自动选择（优先级：MPS > CPU > CUDA）
            - "mps": Mac Apple Silicon GPU
            - "cpu": CPU
            - "cuda": NVIDIA GPU
            - torch.device对象
    
    Returns:
        特征字典，值为torch.Tensor
    
    Example:
        >>> from purrsight.preprocess import Preprocessor
        >>> from purrsight.utils.tools import to_tensor
        >>> features = Preprocessor.process({"text": "I love cat"})
        >>> tensor_features = to_tensor(features)
        >>> tensor_features = to_tensor(features, device="cpu")
    """
    if device is None:
        device = get_available_device()
    
    tensor_features = {}
    for key, value in features.items():
        if isinstance(value, np.ndarray):
            if value.dtype == np.int64:
                tensor_features[key] = torch.from_numpy(value).long()
            elif value.dtype == np.int32:
                tensor_features[key] = torch.from_numpy(value).int()
            elif value.dtype == np.float32 or value.dtype == np.float64:
                tensor_features[key] = torch.from_numpy(value).float()
            elif value.dtype == np.uint8:
                tensor_features[key] = torch.from_numpy(value).float()
            else:
                tensor_features[key] = torch.from_numpy(value.astype(np.float32)).float()
            
            tensor_features[key] = tensor_features[key].to(device)
        else:
            tensor_features[key] = value
    
    return tensor_features

def set_seeds(seed: int = 42):
    """
    设置随机种子以确保可复现性
    
    Args:
        seed: 随机种子值，默认42
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    eval("setattr(torch.backends.cudnn, 'deterministic', True)")
    eval("setattr(torch.backends.cudnn, 'benchmark', False)")
    os.environ["PYTHONHASHSEED"] = str(seed)