"""
Batch处理工具：支持batch内不同样本有不同的模态组合

提供统一的batch处理功能，配合 Preprocessor.process_batch() 使用。

包含：
- prepare_batch_features: 将预处理输出转换为tensor并创建sample-level的modality masks

注意：当前训练流程（train/train_alignment）不再使用此函数。
训练流程现在使用：
1. Dataset.__getitem__() - 返回numpy格式单样本特征
2. collate_batch() - 合并为batch格式numpy数组，创建modality_masks（numpy格式）
3. training_step() - 在GPU上转换为tensor（节省CPU内存）
4. encode_batch() - 编码特征
5. forward() - 对齐特征

此函数保留用于其他场景（测试、其他训练脚本等）。
"""

import torch
import numpy as np
from typing import Dict, Optional, Union, Tuple
from purrsight.config import FeatureKey, Modality, ModalitySource
from purrsight.utils.logging import logger


def prepare_batch_features(
    batch_features: Dict[str, np.ndarray],
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    将Preprocessor.process_batch()的输出转换为tensor并创建sample-level的modality masks
    
    ⚠️ 注意：当前训练流程（train/train_alignment）不再使用此函数。
    训练流程现在由collate_batch()和training_step()处理，以优化内存使用
    （tensor转换在GPU上进行，而不是在CPU上）。
    
    此函数保留用于其他场景（测试、其他训练脚本等）。
    
    功能：
    1. 将numpy数组转换为PyTorch tensor
    2. 创建sample-level的模态mask（每个样本的模态存在标记）
    
    Args:
        batch_features: Preprocessor.process_batch()的输出，键为FeatureKey，值为batch格式的numpy数组
            所有特征形状为(B, ...)
        device: 目标设备，如果为None则自动选择
    
    Returns:
        (tensor_features, modality_masks)元组：
        - tensor_features: 转换后的特征字典，键为FeatureKey，值为形状为(B, ...)的tensor
        - modality_masks: sample-level的模态mask字典，键为模态名称，值为形状为(B,)的bool tensor
            每个元素表示对应样本是否包含该模态
    
    Example:
        >>> from purrsight.preprocess import Preprocessor
        >>> batch_inputs = [
        ...     {"text": "Cat playing", "image": "/path/to/cat1.jpg"},
        ...     {"text": "Cat sleeping"},
        ...     {"image": "/path/to/cat2.jpg"},
        ... ]
        >>> features = Preprocessor.process_batch(batch_inputs)  # numpy batch格式
        >>> tensor_features, masks = prepare_batch_features(features)
    """
    if len(batch_features) == 0:
        raise ValueError("batch_features不能为空")
    
    # 确定batch大小（从第一个特征推断）
    batch_size = None
    for feat in batch_features.values():
        if feat is not None and feat.ndim > 0:
            batch_size = feat.shape[0]
            break
    
    if batch_size is None:
        raise ValueError("无法从batch_features推断batch大小")
    
    # 确定device
    if device is None:
        from purrsight.utils.tools import get_available_device
        device = get_available_device()
    if isinstance(device, str):
        device = torch.device(device)
    
    # 从batch_features中提取modality_masks（预处理阶段创建）
    # Preprocessor.process_batch() 总是创建 _modality_masks，但保留向后兼容逻辑
    if "_modality_masks" in batch_features:
        modality_masks_dict = batch_features["_modality_masks"]
        # 移除特殊键，不参与后续处理
        del batch_features["_modality_masks"]
        
        # 🔧 修复：验证mask与video_metadata的一致性
        video_metadata = batch_features.get("_video_metadata", {})
        if video_metadata:
            for idx_str, meta in video_metadata.items():
                idx = int(idx_str)  # 确保是int
                if idx >= batch_size:
                    continue  # 跳过超出batch范围的索引
                
                # 如果视频帧提取失败，确保IMAGE mask为False
                if not meta.get("image_valid", True):
                    if Modality.IMAGE.value in modality_masks_dict:
                        modality_masks_dict[Modality.IMAGE.value][idx] = False
                        logger.debug(
                            f"样本{idx}: 视频帧提取失败，IMAGE mask已设置为False"
                        )
                
                # 如果视频音频无效且audio_source是video，确保AUDIO mask为False
                if not meta.get("audio_valid", True):
                    audio_source = meta.get("audio_source")
                    if audio_source == ModalitySource.VIDEO.value:
                        if Modality.AUDIO.value in modality_masks_dict:
                            modality_masks_dict[Modality.AUDIO.value][idx] = False
                            logger.debug(
                                f"样本{idx}: 视频音频无效且source=video，AUDIO mask已设置为False"
                            )
    else:
        # 🔧 清理：向后兼容分支（主要用于测试场景）
        # 注意：当前版本的Preprocessor.process_batch()总是创建_modality_masks
        # 此分支主要用于测试或直接调用prepare_batch_features的场景
        logger.warning(
            "batch_features中未找到'_modality_masks'，将根据特征值推断模态存在情况。"
            "这可能是bug，请检查预处理逻辑。如果这是测试场景，可以忽略此警告。"
        )
        modality_masks_dict = {}
        video_metadata = batch_features.get("_video_metadata", {})

        # 简化推断：基于特征存在性和video_metadata
        for modality in [Modality.TEXT, Modality.IMAGE, Modality.AUDIO]:
            modality_key = modality.value
            if modality == Modality.TEXT:
                has_modality = (
                    FeatureKey.TEXT in batch_features
                    and FeatureKey.TEXT_ATTENTION_MASK in batch_features
                )
                if has_modality:
                    mask = np.any(batch_features[FeatureKey.TEXT_ATTENTION_MASK] != 0, axis=1)
                else:
                    mask = np.zeros(batch_size, dtype=np.bool_)
            elif modality == Modality.IMAGE:
                has_modality = FeatureKey.IMAGE in batch_features
                if has_modality:
                    # 简化：检查video_metadata或特征值不为全零
                    img_feat = batch_features[FeatureKey.IMAGE]
                    
                    # 处理16帧格式和单帧格式
                    if img_feat.ndim == 5:
                        # 16帧格式：(B, 16, 3, 224, 224)
                        # Sum掉所有空间和时间维度，只保留batch维度
                        img_mask = np.sum(np.abs(img_feat), axis=(1, 2, 3, 4)) > 1e-6
                    elif img_feat.ndim == 4:
                        # 单帧格式：(B, 3, 224, 224)
                        # Sum掉所有空间维度，只保留batch维度
                        img_mask = np.sum(np.abs(img_feat), axis=(1, 2, 3)) > 1e-6
                    else:
                        # 未知格式，尝试sum掉所有非batch维度
                        sum_axes = tuple(range(1, img_feat.ndim))
                        img_mask = np.sum(np.abs(img_feat), axis=sum_axes) > 1e-6
                    
                    if video_metadata:
                        # 视频样本总是有IMAGE（16帧）
                        video_mask = np.array([
                            idx in video_metadata and video_metadata[idx].get("has_video", False)
                            for idx in range(batch_size)
                        ], dtype=np.bool_)
                        # 合并：视频或独立图像
                        mask = video_mask | img_mask
                    else:
                        mask = img_mask
                else:
                    mask = np.zeros(batch_size, dtype=np.bool_)
            elif modality == Modality.AUDIO:
                has_modality = FeatureKey.AUDIO in batch_features
                if has_modality:
                    # 优化：使用更高效的方法检查非零（避免np.isclose的开销）
                    mask = np.sum(np.abs(batch_features[FeatureKey.AUDIO]), axis=(1, 2)) > 1e-6
                else:
                    mask = np.zeros(batch_size, dtype=np.bool_)

            modality_masks_dict[modality_key] = mask.astype(np.bool_)
    
    # 将numpy modality_masks转换为torch tensor
    # 注意：VIDEO mask不再需要，视频已分解为IMAGE和AUDIO
    modality_masks = {}
    for modality in [Modality.TEXT, Modality.IMAGE, Modality.AUDIO]:
        modality_key = modality.value
        if modality_key in modality_masks_dict:
            mask_np = modality_masks_dict[modality_key]
            # 🔧 修复：确保mask_np是1维数组，避免转换为0维tensor
            if isinstance(mask_np, np.ndarray):
                if mask_np.ndim == 0:
                    # 0维标量，转换为1维数组
                    mask_np = np.array([mask_np.item()], dtype=np.bool_)
                elif mask_np.ndim > 1:
                    # 多维数组，flatten为1维
                    mask_np = mask_np.flatten()
            else:
                # 非数组类型（如bool），转换为1维数组
                mask_np = np.array([bool(mask_np)], dtype=np.bool_)
            mask = torch.from_numpy(mask_np).to(device)
            # 确保mask是1维的
            if mask.dim() == 0:
                mask = mask.unsqueeze(0)
        else:
            mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        modality_masks[modality_key] = mask
    
    # 转换为tensor
    tensor_features = {}
    for key, feat in batch_features.items():
        # 跳过特殊键（不是numpy数组）
        if key == "_video_metadata" or key == "_modality_sources":
            tensor_features[key] = feat  # 保留metadata字典
            continue
        
        if feat is not None:
            # 确保feat是numpy数组
            if not isinstance(feat, np.ndarray):
                # 如果不是numpy数组，跳过或记录警告
                continue
            
            # 转换为tensor
            if feat.dtype == np.int64:
                tensor_feat = torch.from_numpy(feat).long()
            elif feat.dtype == np.int32:
                tensor_feat = torch.from_numpy(feat).int()
            else:
                tensor_feat = torch.from_numpy(feat.astype(np.float32)).float()
            
            tensor_feat = tensor_feat.to(device)
            tensor_features[key] = tensor_feat
    
    return tensor_features, modality_masks
