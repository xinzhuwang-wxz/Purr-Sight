"""
对齐训练数据集

加载多模态对齐数据，进行预处理和tensor化。
编码在 batch 级别进行（在 LightningModule 中），以利用 GPU 并行能力。

包含：
- AlignmentDataset: 多模态对齐数据集类
"""

from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path

from purrsight.preprocess import Preprocessor
from purrsight.config import FeatureKey, Modality
from purrsight.utils.logging import logger


class AlignmentDataset(Dataset):
    """
    多模态对齐数据集
    
    数据流：
    1. 加载原始数据（文件路径或内容）
    2. 预处理（Preprocessor.process）或加载预处理文件（Preprocessor.load_preprocessed）
    3. 返回numpy格式的单样本特征（无batch维度，无modality_masks）
    4. collate_batch合并为batch格式并创建modality_masks
    5. training_step在GPU上转换为tensor
    
    编码在 batch 级别进行（LightningModule 中），利用 GPU 并行能力。
    
    Attributes:
        data_list: 数据列表
        preprocessor: 预处理器实例（仅在非预处理模式下使用）
        device: 预处理设备（默认"cpu"）
        use_preprocessed: 是否使用离线预处理
        preprocessed_dir: 预处理文件目录（仅在use_preprocessed=True时使用）
    """

    def __init__(
        self,
        data_list: List[Dict[str, Any]],
        preprocessor: Optional[Preprocessor] = None,
        device: str = "cpu",  # Dataset 中默认使用 CPU，编码在 GPU 上进行
        use_preprocessed: bool = False,  # 是否使用预处理文件
        preprocessed_dir: Optional[Path] = None  # 预处理文件目录
    ):
        """
        初始化数据集

        Args:
            data_list: 数据列表，每个元素是一个字典，包含模态数据或预处理索引条目
            preprocessor: 预处理器实例，如果为None则创建新实例（仅在非预处理模式下使用）
            device: 设备字符串，默认"cpu"（预处理在 CPU 上进行）
            use_preprocessed: 是否使用预处理文件（如果True，data_list应为索引条目列表）
            preprocessed_dir: 预处理文件的基础目录（仅在use_preprocessed=True时使用）
        """
        self.data_list = data_list
        self.use_preprocessed = use_preprocessed
        self.preprocessed_dir = Path(preprocessed_dir) if preprocessed_dir else None
        
        if use_preprocessed:
            if preprocessed_dir is None:
                raise ValueError("使用预处理模式时，必须指定preprocessed_dir")
            self.preprocessor = None  # 预处理模式下不需要预处理器
        else:
            self.preprocessor = preprocessor or Preprocessor()
        
        self.device = device

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        获取单个样本

        Args:
            idx: 样本索引

        Returns:
            (features, metadata)元组
            features: 预处理后的特征字典（numpy格式），键为FeatureKey，值为numpy数组
                单样本格式，无batch维度。collate_batch会合并为batch格式。
            metadata: 元数据字典（可选），包含原始样本信息等
            
        注意：
        - 不进行编码，编码在 batch 级别进行（LightningModule 中）
        - 不创建modality_masks，collate_batch会统一创建和验证
        - 不添加batch维度，collate_batch会统一处理
        - 不转换为tensor，tensor转换在training_step中进行（GPU上转换，节省CPU内存）
        
        错误处理：如果数据加载失败（文件损坏等），返回空字典，
        collate_batch会统一处理缺失样本（创建零向量和False mask）。
        """
        sample = self.data_list[idx]

        # 根据模式选择加载方式
        try:
            if self.use_preprocessed:
                # 加载预处理文件
                if "preprocessed_files" not in sample:
                    raise ValueError(f"样本 {idx} 缺少 'preprocessed_files' 字段（预处理模式）")
                features = Preprocessor.load_preprocessed(
                    sample["preprocessed_files"],
                    self.preprocessed_dir
                )
            else:
                # 实时预处理（使用单样本方法，更高效）
                features = self.preprocessor.process(sample)
        except Exception as e:
            # 文件损坏或无法加载，静默跳过（使用debug级别，避免日志噪音）
            logger.debug(f"跳过样本 {idx}，加载失败: {e}")
            # 返回空字典，collate_batch会创建零向量和False mask
            return {}, {}
        
        # 🔧 P0修复：不创建modality_masks，collate_batch会统一创建
        # 🔧 P0修复：不添加batch维度，直接返回numpy格式的单样本特征
        # 🔧 内存优化：不转换为tensor，tensor转换在training_step中进行（GPU上转换）
        
        # 保存原始样本信息（用于调试）
        metadata = {"original_sample": sample.get("original_sample", sample)}
        
        return features, metadata