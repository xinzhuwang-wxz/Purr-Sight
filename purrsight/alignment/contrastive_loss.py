"""
对比损失模块：InfoNCE损失实现

参考ImageBind设计，使用InfoNCE损失进行跨模态对比学习。
支持对称损失和模态缺失的情况。

包含：
- infonce_loss: InfoNCE损失函数
- ContrastiveLoss: 对比损失类

参考仓库：
- openai/CLIP (OpenCLIP)
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


def infonce_loss(
    query_features: torch.Tensor,
    key_features: torch.Tensor,
    temperature: Optional[float] = None,
    logit_scale: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    InfoNCE损失函数
    
    计算query和key之间的对比损失。假设批次中第i个query和第i个key是正样本对，
    其他组合是负样本对。
    
    InfoNCE是对整个batch计算的，会计算所有样本之间的相似度矩阵(B, B)：
    - 对角线元素(i, i)是正样本对（同一个样本的不同模态）
    - 非对角线元素(i, j)是负样本对（不同样本的模态）
    
    公式（query→key方向）：
    L_q2k = (1/B) * Σ_i [-log(exp(sim(q_i, k_i) * τ) / Σ_j exp(sim(q_i, k_j) * τ))]
    
    其中：
    - q_i: 第i个query特征
    - k_i: 第i个key特征（正样本）
    - k_j: 第j个key特征（包括正样本j=i和负样本j≠i）
    - τ: 温度参数 = exp(logit_scale)
    - sim(·,·): 余弦相似度（特征已归一化，点积=余弦相似度）
    - Σ_j: 包含所有样本（正样本 + 负样本），不只是负样本
    
    对称损失：
    L = (L_q2k + L_k2q) / 2.0
    
    注意：
    - 温度缩放使用乘法：`similarity * logit_scale`，其中 `logit_scale = exp(logit_scale_param)`
    - 等价于除法形式：`similarity / temperature`，其中 `temperature = 1 / exp(logit_scale_param)`
    
    Args:
        query_features: Query特征，形状为(B, D)，已归一化，dtype=float32
            缺失的模态用零向量填充（padding 0）
        key_features: Key特征，形状为(B, D)，已归一化，dtype=float32
            缺失的模态用零向量填充（padding 0）
        temperature: 温度参数，如果提供，会覆盖logit_scale
        logit_scale: 可学习的logit scale（exp(logit_scale)作为温度），
            如果提供，会覆盖temperature
        mask: Sample-level mask，形状为(B,)，dtype=bool，True表示有效样本
            如果提供，只计算有效样本的损失（但整个batch仍参与相似度计算）
    
    Returns:
        InfoNCE损失值，标量张量，dtype=float32
    """
    B = query_features.shape[0]
    device = query_features.device
    
    # 验证输入形状
    if query_features.shape != key_features.shape:
        raise ValueError(
            f"Query和Key特征形状不匹配："
            f"query {query_features.shape} vs key {key_features.shape}"
        )
    
    # 🔧 优化：检查是否有NaN或Inf（零向量经过归一化后应该保持为零向量，不应该有NaN）
    valid_mask = ~torch.isnan(query_features.sum(dim=1)) & ~torch.isnan(key_features.sum(dim=1))
    valid_mask = valid_mask & ~torch.isinf(query_features.sum(dim=1)) & ~torch.isinf(key_features.sum(dim=1))
    
    # 🔧 优化：如果提供了mask，与valid_mask合并（mask=False的样本应该被完全排除）
    if mask is not None:
        if mask.shape[0] != B:
            raise ValueError(f"Mask形状不匹配：mask {mask.shape} vs batch size {B}")
        # 🔧 性能优化：检查mask设备，避免重复传输
        if mask.device != device:
            mask = mask.to(device)
        # mask=False的样本应该被完全排除（不参与相似度计算）
        valid_mask = valid_mask & mask
    
    # 检查有效样本数量
    valid_count = valid_mask.sum().item()
    if valid_count == 0:
        # 没有有效样本，返回0损失（需要梯度信息）
        from purrsight.utils.logging import logger
        logger.debug(f"所有样本被mask，跳过{query_features.shape[0]}个样本的InfoNCE损失计算")
        return torch.tensor(0.0, device=device, dtype=torch.float32, requires_grad=True)
    
    if valid_count < 2:
        # InfoNCE需要至少2个有效样本才能计算（需要正样本和负样本）
        from purrsight.utils.logging import logger
        logger.debug(f"有效样本数不足（{valid_count} < 2），跳过InfoNCE损失计算")
        return torch.tensor(0.0, device=device, dtype=torch.float32, requires_grad=True)
    
    # 🔧 优化：计算相似度矩阵 (B, B)
    # 注意：这里使用整个batch计算相似度矩阵，包括零向量样本
    # 零向量与其他向量的相似度是0，这是合理的
    # 但由于valid_mask会过滤，零向量样本不会参与损失计算
    # 由于特征已归一化，点积 = 余弦相似度
    logits = query_features @ key_features.T
    
    # 🔧 CRITICAL FIX: Mask invalid keys (columns) to avoid gradient dilution
    # Invalid keys (zero vectors) produce dot product 0, so exp(0)=1 in Softmax denominator.
    # This dilutes the gradient for valid positive pairs.
    # We must mask them to -inf so exp(-inf)=0.
    if valid_mask is not None:
        # valid_mask shape: (B,)
        # Mask columns where key is invalid
        logits[:, ~valid_mask] = -1e9
    
    # 应用温度缩放
    if logit_scale is not None:
        # 使用可学习的logit scale
        logits = logits * logit_scale
    elif temperature is not None:
        # 使用固定的温度参数
        logits = logits / temperature
    
    # 标签：对角线是正样本对
    # 🔧 修复：确保labels是long类型（cross_entropy需要）
    labels = torch.arange(B, device=device, dtype=torch.long)
    
    # 如果只有部分样本有效，只计算有效样本的损失
    if valid_count < B:
        # 只使用有效样本计算损失
        valid_logits = logits[valid_mask]  # (valid_count, B)
        valid_labels = labels[valid_mask]  # (valid_count,)
        
        # 计算loss_q2k：query->key方向
        # valid_logits的形状是(valid_count, B)，valid_labels的形状是(valid_count,)
        loss_q2k = F.cross_entropy(valid_logits, valid_labels)
        
        # 🔧 性能优化：cross_entropy通常返回float32，dtype检查移到最终loss统一处理
        
        # 🔧 性能优化：计算loss_k2q：key->query方向，使用直接索引避免转置
        # logits[i, j] 表示 query_i 和 key_j 的相似度
        # 对于 key->query 方向，我们需要 logits[j, i]（转置）
        # 但我们可以直接索引有效样本，避免创建转置视图
        valid_indices = valid_mask.nonzero(as_tuple=True)[0]  # (valid_count,)
        # 🔧 性能优化：直接提取有效样本的子矩阵，避免转置操作
        # logits[valid_indices][:, valid_indices] 等价于 logits.T[valid_indices][:, valid_indices]
        # 但更高效，因为避免了转置操作
        valid_logits_k2q = logits[valid_indices][:, valid_indices]  # (valid_count, valid_count)
        transposed_labels = torch.arange(valid_count, device=device, dtype=torch.long)  # (valid_count,)
        
        loss_k2q = F.cross_entropy(valid_logits_k2q, transposed_labels)
        
        # 🔧 性能优化：cross_entropy通常返回float32，只在必要时检查
        # 如果cross_entropy返回非float32，会在training_step中统一处理
    else:
        # 所有样本都有效，计算对称损失
        # 🔧 修复：确保labels是long类型（cross_entropy需要）
        if labels.dtype != torch.long:
            labels = labels.long()
        
        loss_q2k = F.cross_entropy(logits, labels)
        # 🔧 性能优化：使用torch.transpose而不是.T，更明确且可能更高效
        loss_k2q = F.cross_entropy(torch.transpose(logits, 0, 1), labels)
        
        # 🔧 性能优化：cross_entropy通常返回float32，只在必要时检查
        # 如果cross_entropy返回非float32，会在training_step中统一处理
    
    # 平均损失
    loss = (loss_q2k + loss_k2q) / 2.0
    
    # 🔧 性能优化：cross_entropy通常返回float32，这里只做最终检查
    # 如果cross_entropy返回非float32，会在training_step中统一处理
    # 这里保留检查以确保兼容性，但cross_entropy应该总是返回float32
    
    return loss


class ContrastiveLoss:
    """
    多模态对比损失计算器
    
    支持多模态之间的对比学习，自动处理模态缺失的情况。
    参考ImageBind设计，计算所有模态对之间的InfoNCE损失。
    
    Attributes:
        use_temperature_scaling: 是否使用温度缩放
        default_temperature: 默认温度参数
    """
    
    def __init__(
        self,
        use_temperature_scaling: bool = True,
        default_temperature: float = 0.07,
    ):
        """
        初始化对比损失计算器
        
        Args:
            use_temperature_scaling: 是否使用温度缩放，默认True
            default_temperature: 默认温度参数（当不使用温度缩放时），默认0.07
        """
        self.use_temperature_scaling = use_temperature_scaling
        self.default_temperature = default_temperature
    
    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        modality_presence: Optional[Dict[str, bool]] = None,
        logit_scales: Optional[Dict[str, torch.Tensor]] = None,
        modality_masks: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算多模态对比损失
        
        计算所有存在的模态对之间的InfoNCE损失，并返回总损失和各项损失。
        视频数据已分解为图像和音频，直接参与标准损失计算。
        
        Args:
            features: 后处理后的特征字典，键为模态名称，值为特征张量
                形状为(B, 512)，已归一化，dtype=float32（匹配LLM embedding维度）
            modality_presence: 模态存在标记字典，键为模态名称，值为bool
                如果为None，则根据features中的键自动推断
            logit_scales: logit scale字典，键为模态名称，值为logit scale张量
                如果为None且use_temperature_scaling=True，则使用default_temperature
            modality_masks: Sample-level模态mask字典，键为模态名称，值为形状为(B,)的bool tensor
                表示batch中每个样本的模态是否存在，用于精确控制损失计算
        
        Returns:
            (total_loss, loss_dict)元组：
            - total_loss: 总损失值，标量张量
            - loss_dict: 各项损失字典，键为"modality1_modality2"，值为损失值
        """
        # 🔧 优化：优先使用传入的modality_presence，避免重复推断
        # 如果没有提供modality_presence，根据features和modality_masks自动推断
        if modality_presence is None:
            from purrsight.config import Modality
            modality_presence = {}
            for modality in [Modality.TEXT, Modality.IMAGE, Modality.AUDIO]:
                modality_key = modality.value
                # 优先使用modality_masks判断（更准确）
                if modality_masks is not None and modality_key in modality_masks:
                    mask = modality_masks[modality_key]
                    # 如果mask中至少有一个True，说明batch中至少有一个样本有该模态
                    modality_presence[modality_key] = mask.any().item() if mask.numel() > 0 else False
                else:
                    # 回退到features判断
                    modality_presence[modality_key] = (
                        modality_key in features
                        and features[modality_key] is not None
                        and not torch.all(features[modality_key] == 0)  # 排除零向量
                    )
        
        # 获取存在的模态列表
        present_modalities = [
            modality
            for modality, present in modality_presence.items()
            if present
        ]
        
        if len(present_modalities) < 2:
            # 至少需要两个模态才能计算对比损失
            device = next(iter(features.values())).device
            from purrsight.utils.logging import logger
            logger.debug(f"模态数量不足（{len(present_modalities)} < 2），跳过对比损失计算")
            return torch.tensor(0.0, device=device, dtype=torch.float32, requires_grad=True), {}
        
        losses = {}
        loss_pairs = []

        # 计算所有模态对之间的损失（视频样本的image和audio与独立样本一样参与计算）
        for i, modality1 in enumerate(present_modalities):
            for modality2 in present_modalities[i + 1:]:
                feat1 = features[modality1]
                feat2 = features[modality2]
                
                # 确定温度参数
                temperature = None
                logit_scale = None
                
                if self.use_temperature_scaling:
                    if logit_scales is not None:
                        # 使用两个模态的平均logit scale
                        scale1 = logit_scales.get(modality1, None)
                        scale2 = logit_scales.get(modality2, None)
                        if scale1 is not None and scale2 is not None:
                            logit_scale = (scale1 + scale2) / 2.0
                        elif scale1 is not None:
                            logit_scale = scale1
                        elif scale2 is not None:
                            logit_scale = scale2
                        else:
                            temperature = self.default_temperature
                    else:
                        temperature = self.default_temperature
                else:
                    temperature = self.default_temperature
                
                # 创建sample-level mask：两个模态都有效的样本
                mask = None
                if modality_masks is not None:
                    mask1 = modality_masks.get(modality1)
                    mask2 = modality_masks.get(modality2)
                    if mask1 is not None and mask2 is not None:
                        # 确保mask在正确的设备上（与特征相同）
                        device = feat1.device
                        B = feat1.shape[0]
                        
                        # 验证并修正mask1形状
                        if mask1.dim() != 1:
                            # 如果不是1维，尝试squeeze或reshape
                            if mask1.numel() == B:
                                mask1 = mask1.view(B)
                            else:
                                raise ValueError(
                                    f"modality mask {modality1} 形状不正确: "
                                    f"shape={mask1.shape}, numel={mask1.numel()}, expected ({B},)"
                                )
                        elif mask1.shape[0] != B:
                            raise ValueError(
                                f"modality mask {modality1} batch size不匹配: "
                                f"shape={mask1.shape}, expected ({B},)"
                            )
                        
                        # 验证并修正mask2形状
                        if mask2.dim() != 1:
                            # 如果不是1维，尝试squeeze或reshape
                            if mask2.numel() == B:
                                mask2 = mask2.view(B)
                            else:
                                raise ValueError(
                                    f"modality mask {modality2} 形状不正确: "
                                    f"shape={mask2.shape}, numel={mask2.numel()}, expected ({B},)"
                                )
                        elif mask2.shape[0] != B:
                            raise ValueError(
                                f"modality mask {modality2} batch size不匹配: "
                                f"shape={mask2.shape}, expected ({B},)"
                            )
                        
                        # 🔧 性能优化：检查设备，避免重复传输
                        if mask1.device != device:
                            mask1 = mask1.to(device)
                        if mask2.device != device:
                            mask2 = mask2.to(device)
                        mask = mask1 & mask2  # 两个模态都有效的样本

                # 计算InfoNCE损失
                pair_loss = infonce_loss(
                    feat1, feat2, temperature=temperature, logit_scale=logit_scale, mask=mask
                )
                
                pair_name = f"{modality1}_{modality2}"
                losses[pair_name] = pair_loss
                loss_pairs.append(pair_loss)

        # 计算平均损失
        if len(loss_pairs) > 0:
            total_loss = sum(loss_pairs) / len(loss_pairs)
            # 🔧 修复：确保loss的dtype是float32（mixed precision训练需要）
            if total_loss.dtype != torch.float32:
                total_loss = total_loss.float()
        else:
            device = next(iter(features.values())).device
            total_loss = torch.tensor(0.0, device=device, dtype=torch.float32, requires_grad=True)
        
        return total_loss, losses


    def __call__(
        self,
        features: Dict[str, torch.Tensor],
        modality_presence: Optional[Dict[str, bool]] = None,
        logit_scales: Optional[Dict[str, torch.Tensor]] = None,
        modality_masks: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        调用compute_loss方法
        
        Args:
            features: 后处理后的特征字典
            modality_presence: 模态存在标记字典
            logit_scales: logit scale字典
            modality_masks: Sample-level模态mask字典，键为模态名称，值为形状为(B,)的bool tensor
        
        Returns:
            (total_loss, loss_dict)元组
        """
        return self.compute_loss(features, modality_presence, logit_scales, modality_masks)

