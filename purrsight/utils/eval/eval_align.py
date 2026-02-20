"""
对比学习评估指标模块

提供对比学习任务的评估指标，包括检索指标、相似度指标和对齐质量指标。

包含：
- ContrastiveMetrics: 对比学习评估指标计算器类
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np

from purrsight.config import Modality


class ContrastiveMetrics:
    """
    对比学习评估指标计算器
    
    提供多种评估指标：
    1. Retrieval metrics: Recall@K, MRR, Median Rank
    2. Similarity metrics: Positive/Negative pair similarity
    3. Alignment quality: Cross-modal alignment score
    
    Attributes:
        k_values: Recall@K的K值列表
    """
    
    def __init__(self, k_values: List[int] = [1, 5, 10]):
        """
        初始化评估指标计算器
        
        Args:
            k_values: Recall@K的K值列表，默认[1, 5, 10]
        """
        self.k_values = k_values
    
    def compute_retrieval_metrics(
        self,
        query_features: torch.Tensor,
        key_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        计算检索指标：Recall@K, MRR, Median Rank
        
        Args:
            query_features: Query特征，形状为(B, D)，已归一化
            key_features: Key特征，形状为(B, D)，已归一化
            mask: Sample-level mask，形状为(B,)，True表示有效样本
        
        Returns:
            包含检索指标的字典：
            - recall_at_k: Dict[int, float] - 各K值的Recall@K
            - mrr: float - Mean Reciprocal Rank
            - median_rank: float - Median Rank
        """
        B = query_features.shape[0]
        device = query_features.device
        
        # 应用mask（如果提供）
        if mask is not None:
            if mask.shape[0] != B:
                raise ValueError(f"Mask形状不匹配：mask {mask.shape} vs batch size {B}")
            if mask.device != device:
                mask = mask.to(device)
            valid_indices = torch.where(mask)[0]
            if len(valid_indices) < 2:
                # 有效样本不足，返回默认值
                return {
                    **{f"recall_at_{k}": 0.0 for k in self.k_values},
                    "mrr": 0.0,
                    "median_rank": float(B),
                }
            query_features = query_features[valid_indices]
            key_features = key_features[valid_indices]
            B = len(valid_indices)
        else:
            valid_indices = torch.arange(B, device=device)
        
        # 计算相似度矩阵 (B, B)
        # 对角线元素(i, i)是正样本对
        similarity_matrix = query_features @ key_features.T  # (B, B)
        
        # 🔧 性能优化：向量化rank计算，避免循环排序
        # 一次性对所有行排序（O(B log B)而不是B次O(B log B)）
        sorted_indices = torch.argsort(similarity_matrix, dim=1, descending=True)  # (B, B)
        
        # 找到每个正样本的rank：sorted_indices[i, rank-1] == i
        # 创建索引矩阵：每行的第i列应该是i（正样本位置）
        target_indices = torch.arange(B, device=device).unsqueeze(1).expand(-1, B)  # (B, B)
        # 找到每行中target_indices的位置
        rank_mask = (sorted_indices == target_indices)  # (B, B)，每行只有一个True
        ranks = rank_mask.nonzero(as_tuple=True)[1] + 1  # rank从1开始
        ranks = ranks.cpu().numpy()  # 转换为numpy数组
        
        # 🔧 性能优化：使用numpy批量计算，避免多次.item()调用
        # 计算Recall@K
        recall_at_k = {}
        for k in self.k_values:
            recall_at_k[f"recall_at_{k}"] = float((ranks <= k).mean())
        
        # 计算MRR (Mean Reciprocal Rank)
        reciprocal_ranks = 1.0 / ranks
        mrr = float(reciprocal_ranks.mean())
        
        # 计算Median Rank
        median_rank = float(np.median(ranks))
        
        return {
            **recall_at_k,
            "mrr": mrr,
            "median_rank": median_rank,
        }
    
    def compute_similarity_metrics(
        self,
        query_features: torch.Tensor,
        key_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        计算相似度指标：正样本对和负样本对的平均相似度
        
        Args:
            query_features: Query特征，形状为(B, D)，已归一化
            key_features: Key特征，形状为(B, D)，已归一化
            mask: Sample-level mask，形状为(B,)，True表示有效样本
        
        Returns:
            包含相似度指标的字典：
            - positive_similarity: float - 正样本对的平均相似度
            - negative_similarity: float - 负样本对的平均相似度
            - similarity_gap: float - 正负样本相似度差距
        """
        B = query_features.shape[0]
        device = query_features.device
        
        # 应用mask（如果提供）
        if mask is not None:
            if mask.shape[0] != B:
                raise ValueError(f"Mask形状不匹配：mask {mask.shape} vs batch size {B}")
            if mask.device != device:
                mask = mask.to(device)
            valid_indices = torch.where(mask)[0]
            if len(valid_indices) < 2:
                return {
                    "positive_similarity": 0.0,
                    "negative_similarity": 0.0,
                    "similarity_gap": 0.0,
                }
            query_features = query_features[valid_indices]
            key_features = key_features[valid_indices]
            B = len(valid_indices)
        
        # 计算相似度矩阵 (B, B)
        similarity_matrix = query_features @ key_features.T  # (B, B)
        
        # 🔧 性能优化：批量计算后再调用.item()，减少CPU-GPU同步
        # 正样本对：对角线元素
        positive_similarities = torch.diag(similarity_matrix)  # (B,)
        positive_similarity = positive_similarities.mean().item()
        
        # 负样本对：非对角线元素
        # 创建mask，排除对角线
        mask_matrix = ~torch.eye(B, dtype=torch.bool, device=device)
        negative_similarities = similarity_matrix[mask_matrix]  # (B*(B-1),)
        negative_similarity = negative_similarities.mean().item()
        
        # 相似度差距（在CPU上计算，避免GPU-CPU同步）
        similarity_gap = positive_similarity - negative_similarity
        
        return {
            "positive_similarity": positive_similarity,
            "negative_similarity": negative_similarity,
            "similarity_gap": similarity_gap,
        }
    
    def compute_alignment_score(
        self,
        features: Dict[str, torch.Tensor],
        modality_masks: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, float]:
        """
        计算对齐质量指标：跨模态对齐分数
        
        对于每个模态对，计算：
        1. 检索指标（Recall@K, MRR）
        2. 相似度指标（正负样本相似度差距）
        
        Args:
            features: 对齐后的特征字典，键为模态名称，值为(B, D)的tensor
            modality_masks: 模态mask字典，键为模态名称，值为(B,)的bool tensor
        
        Returns:
            包含所有模态对指标的字典，键格式为：
            - {modality1}_{modality2}_recall_at_k
            - {modality1}_{modality2}_mrr
            - {modality1}_{modality2}_median_rank
            - {modality1}_{modality2}_positive_similarity
            - {modality1}_{modality2}_negative_similarity
            - {modality1}_{modality2}_similarity_gap
        """
        metrics = {}
        
        # 获取所有模态对
        modalities = list(features.keys())
        
        for i, mod1 in enumerate(modalities):
            for mod2 in modalities[i+1:]:
                if mod1 not in features or mod2 not in features:
                    continue
                
                feat1 = features[mod1]
                feat2 = features[mod2]
                
                # 获取mask（如果提供）
                mask1 = modality_masks.get(mod1) if modality_masks else None
                mask2 = modality_masks.get(mod2) if modality_masks else None
                
                # 合并mask（两个模态都有效才计算）
                if mask1 is not None and mask2 is not None:
                    mask = mask1 & mask2
                elif mask1 is not None:
                    mask = mask1
                elif mask2 is not None:
                    mask = mask2
                else:
                    mask = None
                
                # 计算检索指标
                retrieval_metrics = self.compute_retrieval_metrics(feat1, feat2, mask)
                for key, value in retrieval_metrics.items():
                    metrics[f"{mod1}_{mod2}_{key}"] = value
                
                # 计算相似度指标
                similarity_metrics = self.compute_similarity_metrics(feat1, feat2, mask)
                for key, value in similarity_metrics.items():
                    metrics[f"{mod1}_{mod2}_{key}"] = value
        
        return metrics
    
    def compute_batch_metrics(
        self,
        features: Dict[str, torch.Tensor],
        modality_masks: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, float]:
        """
        计算batch级别的所有评估指标
        
        这是主要的接口函数，返回所有评估指标。
        
        Args:
            features: 对齐后的特征字典，键为模态名称，值为(B, D)的tensor
            modality_masks: 模态mask字典，键为模态名称，值为(B,)的bool tensor
        
        Returns:
            包含所有评估指标的字典
        """
        return self.compute_alignment_score(features, modality_masks)
