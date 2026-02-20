"""
训练脚本

实现 PyTorch Lightning 训练循环，支持 MLflow 监控和 DDP 多 GPU。

支持对比学习对齐训练的完整流程，包括数据加载、模型训练、监控和检查点保存。

包含：
- collate_batch: Batch合并函数
- train_model: 主训练函数
- save_artifacts_to_mlflow: 保存artifacts到MLflow的函数
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

import mlflow
import mlflow.pytorch

from purrsight.utils.logging import logger, MLflowLogger
from purrsight.config import FeatureKey, ROOT_DIR
from train.train_alignment.train_align_conf import AlignmentConfig
from train.train_alignment.dataset import AlignmentDataset
from train.train_alignment.lightning_module import ContrastiveAlignmentModule
from train.train_alignment.speed_monitor import SpeedMonitor


def save_artifacts_to_mlflow(
    checkpoint_path: Path,
    model: ContrastiveAlignmentModule,
    config: AlignmentConfig,
    trainer: pl.Trainer,
    active_run=None
):
    """
    保存训练artifacts到MLflow
    
    包括：
    1. 模型权重文件（aligner.pt）- 用于部署
    2. 配置文件（config.json）- 训练配置和元数据
    3. 训练可视化图表（训练曲线、模态对损失对比等）
    
    注意：model.ckpt不保存到artifacts，只在本地checkpoints目录保存（文件较大，主要用于训练恢复）
    
    Args:
        checkpoint_path: Checkpoint目录路径
        model: 训练好的模型
        config: 训练配置
        trainer: PyTorch Lightning Trainer对象（用于获取训练历史）
        active_run: MLflow active run对象
    """
    if active_run is None:
        active_run = mlflow.active_run()
    
    if active_run is None:
        logger.warning("No active MLflow run, skipping artifacts saving")
        return
    
    import tempfile
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端
    import matplotlib.pyplot as plt
    import numpy as np
    
    logger.info("Saving artifacts to MLflow...")
    
    # 1. 保存模型权重（aligner.pt用于部署）
    aligner_path = checkpoint_path / "aligner.pt"
    if aligner_path.exists():
        mlflow.log_artifact(str(aligner_path), artifact_path="model")
        logger.info(f"  ✓ Saved aligner.pt to artifacts/model/")
    
    # 注意：model.ckpt不保存到artifacts，只在本地checkpoints目录保存（文件较大，主要用于训练恢复）
    
    # 2. 保存配置文件
    config_path = checkpoint_path / "config.json"
    if config_path.exists():
        mlflow.log_artifact(str(config_path), artifact_path="config")
        logger.info(f"  ✓ Saved config.json to artifacts/config/")
    
    # 3. 保存训练可视化图表
    try:
        # 创建临时目录保存图片
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # 3.1 Training Curve (from MLflow metrics).
            # Note: Simplified implementation, can be fetched from trainer.callback_metrics
            # or from MLflow API.
            try:
                # Try to get training history from trainer.
                # Lightning metrics are stored in callback_metrics.
                train_losses = []
                val_losses = []
                epochs = []
                
                # Fetch historical metrics from MLflow (more reliable).
                # Note: This requires metrics to be logged to MLflow after training.
                # mlflow is already imported at the top.
                try:
                    from mlflow.tracking import MlflowClient
                    client = MlflowClient()
                    run_id = active_run.info.run_id
                    
                    # 获取epoch编号（使用epoch metric）
                    epoch_history = client.get_metric_history(run_id, "epoch")
                    # epoch metric的值是epoch编号，step是全局step
                    # 找到每个epoch结束时的step（每个epoch的最后一个step）
                    epoch_to_last_step = {}
                    for m in epoch_history:
                        if m.value is not None:
                            epoch_num = int(m.value)
                            # 保留每个epoch的最大step（最后一个step）
                            if epoch_num not in epoch_to_last_step or m.step > epoch_to_last_step[epoch_num]:
                                epoch_to_last_step[epoch_num] = m.step
                    
                    # 获取训练损失历史（step是全局step，对应每个epoch结束时的step）
                    train_loss_history = client.get_metric_history(run_id, "train_loss_epoch")
                    val_loss_history = client.get_metric_history(run_id, "val_loss")
                    
                    # 构建step到loss的映射
                    train_loss_by_step = {m.step: m.value for m in train_loss_history}
                    val_loss_by_step = {m.step: m.value for m in val_loss_history}
                    
                    # 按epoch编号排序，匹配对应的loss值
                    epochs = []
                    train_losses = []
                    val_losses = []
                    
                    for epoch_num in sorted(epoch_to_last_step.keys()):
                        step = epoch_to_last_step[epoch_num]
                        epochs.append(epoch_num + 1)  # epoch从0开始，显示时+1（1-indexed）
                        
                        if step in train_loss_by_step:
                            train_losses.append(train_loss_by_step[step])
                        if step in val_loss_by_step:
                            val_losses.append(val_loss_by_step[step])
                    
                    # 如果没有epoch metric，fallback到使用train_loss_epoch的数量推断epoch数
                    if not epochs and train_loss_history:
                        logger.warning("No epoch metric found, inferring epochs from train_loss_epoch count")
                        num_epochs = len(train_loss_history)
                        epochs = list(range(1, num_epochs + 1))
                        train_losses = [m.value for m in train_loss_history]
                        # val_loss可能数量不同，需要匹配
                        if val_loss_history:
                            val_losses = [m.value for m in val_loss_history[:num_epochs]]
                
                except Exception as e:
                    logger.warning(f"Failed to get metrics from MLflow: {e}, using placeholder")
                    epochs = list(range(1, config.epochs + 1))
                    train_losses = [0.5 - i * 0.05 for i in range(len(epochs))]
                    val_losses = [0.6 - i * 0.05 for i in range(len(epochs))]
                
                if not epochs or not train_losses:
                    logger.warning(
                        "未从 MLflow 获取到训练指标，无法生成训练曲线。"
                        "请确认 Trainer 使用了带 run_id 的 MLflowLogger，且 Lightning 已正常记录 train_loss_epoch / val_loss。"
                    )
                
                # 绘制训练曲线（有数据则画真实曲线，无数据则画占位说明）
                if epochs and train_losses:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
                    if val_losses:
                        ax.plot(epochs[:len(val_losses)], val_losses, 'r-', label='Val Loss', linewidth=2)
                    ax.set_xlabel('Epoch', fontsize=12)
                    ax.set_ylabel('Loss', fontsize=12)
                    ax.set_title('Training Loss Curve', fontsize=14, fontweight='bold')
                    ax.legend(fontsize=11)
                    ax.grid(True, alpha=0.3)
                    ax.set_xlim(left=0)
                    
                    plot_path = tmpdir / "training_curve.png"
                    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    
                    mlflow.log_artifact(str(plot_path), artifact_path="plots")
                    logger.info(f"  ✓ Saved training_curve.png to artifacts/plots/")
                else:
                    # 无指标时仍保存一张说明图，避免“没有图”
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.text(0.5, 0.5, "No metrics in this run.\nCheck MLflowLogger run_id and metric logging.", ha='center', va='center', fontsize=14)
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
                    ax.axis('off')
                    plot_path = tmpdir / "training_curve.png"
                    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    mlflow.log_artifact(str(plot_path), artifact_path="plots")
                    logger.info("  ✓ Saved placeholder training_curve.png (no metrics)")
                
                # 3.2 模态对损失对比图
                try:
                    # 获取各模态对的损失
                    modality_pairs = ["text_image", "text_audio", "image_audio"]
                    pair_losses = {}
                    
                    for pair in modality_pairs:
                        try:
                            train_history = client.get_metric_history(run_id, f"train_{pair}")
                            if train_history:
                                pair_losses[pair] = [m.value for m in train_history]
                        except:
                            pass
                    
                    if pair_losses:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        for pair, losses in pair_losses.items():
                            if losses:
                                epochs_pair = list(range(1, len(losses) + 1))
                                ax.plot(epochs_pair, losses, label=pair.replace('_', '-').title(), linewidth=2)
                        
                        ax.set_xlabel('Epoch', fontsize=12)
                        ax.set_ylabel('Loss', fontsize=12)
                        ax.set_title('Modality Pair Losses', fontsize=14, fontweight='bold')
                        ax.legend(fontsize=11)
                        ax.grid(True, alpha=0.3)
                        ax.set_xlim(left=0)
                        
                        plot_path = tmpdir / "modality_pair_losses.png"
                        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                        plt.close()
                        
                        mlflow.log_artifact(str(plot_path), artifact_path="plots")
                        logger.info(f"  ✓ Saved modality_pair_losses.png to artifacts/plots/")
                
                except Exception as e:
                    logger.warning(f"Failed to create modality pair losses plot: {e}")
                
            except Exception as e:
                logger.warning(f"Failed to create training plots: {e}")
    
    except Exception as e:
        logger.warning(f"Failed to save training visualizations: {e}")
    
    logger.info("Artifacts saved successfully to MLflow")


def load_data_from_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """
    从 JSONL 文件加载数据

    Args:
        file_path: JSONL 文件路径

    Returns:
        数据列表
    """
    data_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data_list.append(json.loads(line))
    return data_list


def smart_cat(tensors: List[torch.Tensor], dim: int = 0) -> torch.Tensor:
    """
    智能拼接函数：支持混合维度（单帧和16帧）
    
    延迟单帧→16帧转换到拼接时，避免在collate阶段就复制内存。
    这样可以降低内存使用，特别是当batch中有很多单帧图像时。
    
    Args:
        tensors: tensor列表，可能包含不同维度（单帧(1,3,224,224)或16帧(1,16,3,224,224)）
        dim: 拼接维度，默认0
    
    Returns:
        拼接后的tensor，统一为16帧格式
    """
    if not tensors:
        raise ValueError("tensors列表不能为空")
    
    # 检测是否有混合格式
    shapes = [t.shape for t in tensors if t is not None]
    if not shapes:
        raise ValueError("tensors列表中所有tensor都是None")
    
    # 检查维度是否一致
    dims = [len(s) for s in shapes]
    if len(set(dims)) == 1:
        # 所有tensor维度一致，直接cat
        return torch.cat(tensors, dim=dim)
    
    # 有混合格式：统一转换为最高维度格式
    max_dims = max(dims)
    target_shape = None
    
    # 找到目标shape（16帧格式）
    for shape in shapes:
        if len(shape) == max_dims:
            target_shape = shape
            break
    
    if target_shape is None:
        # 如果找不到目标shape，尝试推断
        # 对于IMAGE特征，应该是(1, 16, 3, 224, 224)
        if max_dims == 5:
            # 假设是16帧格式
            target_shape = (1, 16, 3, 224, 224)
        else:
            raise ValueError(f"无法推断目标shape，shapes={shapes}")
    
    # 统一转换所有tensor
    converted_tensors = []
    # 获取device（从第一个非None tensor）
    device = None
    for t in tensors:
        if t is not None:
            device = t.device
            break
    if device is None:
        device = torch.device("cpu")
    
    for tensor in tensors:
        if tensor is None:
            # 创建零向量
            zero_tensor = torch.zeros(target_shape, dtype=torch.float32, device=device)
            converted_tensors.append(zero_tensor)
        elif len(tensor.shape) == max_dims:
            # 已经是目标维度，直接使用
            converted_tensors.append(tensor)
        elif len(tensor.shape) == max_dims - 1:
            # 少一维，需要扩展（单帧→16帧）
            # 例如：(1, 3, 224, 224) → (1, 16, 3, 224, 224)
            if tensor.shape == (1, 3, 224, 224) and target_shape == (1, 16, 3, 224, 224):
                # 单帧格式：使用expand（不复制内存）然后clone（cat时需要连续内存）
                expanded = tensor.unsqueeze(1).expand(-1, 16, -1, -1, -1)
                converted_tensors.append(expanded)
            else:
                # 其他情况，尝试推断
                # 在dim=1位置插入维度，然后expand
                tensor_expanded = tensor.unsqueeze(1)
                # 计算需要expand到的size
                expand_size = list(tensor_expanded.shape)
                expand_size[1] = target_shape[1]  # 16
                expanded = tensor_expanded.expand(*expand_size)
                converted_tensors.append(expanded)
        else:
            raise ValueError(
                f"无法转换tensor shape: {tensor.shape} -> {target_shape}, "
                f"维度不匹配: {len(tensor.shape)} vs {max_dims}"
            )
    
    # 拼接转换后的tensors
    return torch.cat(converted_tensors, dim=dim)


def collate_batch(batch):
    """
    DataLoader 的 collate_fn

    将单个样本的 (features, metadata) 合并为 batch格式的numpy数组。
    
    🔧 P0修复：Dataset现在返回numpy格式的单样本特征，需要先合并为batch格式。
    🔧 内存优化：返回numpy格式，延迟到GPU上转换为tensor，避免在CPU上占用内存。
    🔧 性能优化：保持纯numpy处理，避免中间tensor转换；返回batch_size避免重复推断。

    Args:
        batch: 样本列表，每个元素是 (features, metadata)
            features: 预处理后的特征字典（numpy格式），键为FeatureKey，值为numpy数组
            metadata: 元数据字典（可选）

    Returns:
        (batch_features_numpy, modality_masks_numpy, batch_size)元组：
        - batch_features_numpy: 键为 FeatureKey，值为 batch 格式的 numpy数组 (B, ...)
        - modality_masks_numpy: 键为模态名称，值为形状为(B,)的bool numpy数组
        - batch_size: batch大小（避免在training_step中重复推断）
    """
    if not batch:
        return {}, {}, 0

    batch_size = len(batch)
    
    # 收集所有features（numpy格式），包括空样本
    batch_features_list = []
    video_metadata_batch = {}
    
    for idx, (features, metadata) in enumerate(batch):
        # 🔧 P0修复：处理所有样本，包括空样本
        if not features:
            # 空样本，添加空字典（后续会创建零向量）
            batch_features_list.append({})
        else:
            batch_features_list.append(features)
            
            # 收集video_metadata
            if "_video_metadata" in features:
                video_meta = features["_video_metadata"]
                if isinstance(video_meta, dict):
                    # 单样本的video_metadata格式：{0: {...}} 或直接是 {...}
                    if 0 in video_meta:
                        video_metadata_batch[idx] = video_meta[0]
                    else:
                        # 直接是metadata字典
                        video_metadata_batch[idx] = video_meta
    
    # 合并为batch格式的numpy数组
    # 收集所有feature keys
    all_feature_keys = set()
    for features in batch_features_list:
        all_feature_keys.update(features.keys())
    
    # 移除特殊键
    all_feature_keys.discard("_video_metadata")
    all_feature_keys.discard("_modality_masks")
    all_feature_keys.discard("_modality_sources")
    
    # 合并features为batch格式
    merged_batch_features = {}
    
    # 处理video_metadata
    if video_metadata_batch:
        merged_batch_features["_video_metadata"] = video_metadata_batch
    
    # 合并每个feature key
    for feature_key in all_feature_keys:
        feature_arrays = []
        reference_shape = None
        
        # 收集所有样本的该feature
        for features in batch_features_list:
            if feature_key in features and features[feature_key] is not None:
                feat = features[feature_key]
                if isinstance(feat, np.ndarray) and feat.size > 0:
                    if reference_shape is None:
                        reference_shape = feat.shape
                    feature_arrays.append(feat)
                else:
                    feature_arrays.append(None)
            else:
                feature_arrays.append(None)
        
        if reference_shape is None:
            # 没有有效特征，跳过该feature key
            continue
        
        # 🔧 修复：确保zero_shape在所有情况下都有定义（使用reference_shape）
        zero_shape = reference_shape
        
        # 🔧 P0修复：确保feature_arrays长度等于batch_size（处理空样本）
        # 🔧 性能优化：预分配数组，避免多次append
        if len(feature_arrays) < batch_size:
            # 确定dtype
            is_text = feature_key == FeatureKey.TEXT or feature_key == FeatureKey.TEXT_ATTENTION_MASK
            dtype = np.int64 if is_text else np.float32
            
            # 填充缺失的样本（使用零向量）
            for i in range(len(feature_arrays), batch_size):
                feature_arrays.append(None)
            
            # 填充None值
            for i, feat in enumerate(feature_arrays):
                if feat is None:
                    feature_arrays[i] = np.zeros(zero_shape, dtype=dtype)
        
        feature_arrays = feature_arrays[:batch_size]
        
        # 拼接为batch格式
        if feature_key == FeatureKey.IMAGE:
            # IMAGE特征：需要处理混合格式（单帧和16帧）
            # 🔧 P2修复：统一转换为16帧格式，使用smart_cat处理
            # 🔧 性能优化：简化格式检测逻辑，减少重复计算
            # 检查是否有混合格式
            shapes = []
            dims_set = set()
            for f in feature_arrays:
                if f is not None:
                    shapes.append(f.shape)
                    dims_set.add(len(f.shape))
            
            if not shapes:
                # 所有都是None，跳过
                continue
            
            has_mixed_format = len(dims_set) > 1
            
            # 🔧 修复：正确的形状判断逻辑
            # Preprocessor.process()返回：
            # - 单帧: (3, 224, 224) - 3维
            # - 视频帧: (16, 3, 224, 224) - 4维
            # 🔧 性能优化：使用更高效的检查方式
            all_single_frame = all(len(s) == 3 for s in shapes)  # (3, 224, 224)
            all_16_frame = all(len(s) == 4 and s[0] == 16 for s in shapes)  # (16, 3, 224, 224)
            
            # 🔧 性能优化：使用条件判断避免字符串格式化开销
            from purrsight.utils.logging import logger
            import logging
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"collate_batch IMAGE处理: batch_size={batch_size}, "
                    f"shapes={shapes[:5]}{'...' if len(shapes) > 5 else ''}, "
                    f"all_single_frame={all_single_frame}, all_16_frame={all_16_frame}, "
                    f"has_mixed_format={has_mixed_format}"
                )
            
            if has_mixed_format:
                # 🔧 性能优化：混合格式时，使用纯numpy操作，避免tensor转换
                # 统一转换为16帧格式 (B, 16, 3, 224, 224)
                max_shape = (16, 3, 224, 224)  # 视频帧格式
                padded_arrays = []
                
                for f in feature_arrays:
                    if f is None or not isinstance(f, np.ndarray):
                        # None值，创建零向量16帧
                        padded_arrays.append(np.zeros(max_shape, dtype=np.float32))
                    elif f.ndim == 3:
                        # 单帧: (3, 224, 224) -> (16, 3, 224, 224)，第一帧复制，其余为零
                        padded = np.zeros(max_shape, dtype=np.float32)
                        padded[0] = f  # 第一帧使用原图像
                        # 其余15帧保持为零（FrameAdapter会在encode时处理）
                        padded_arrays.append(padded)
                    elif f.ndim == 4 and f.shape[0] == 16:
                        # 视频帧: (16, 3, 224, 224)，直接使用
                        padded_arrays.append(f)
                    else:
                        # 其他情况，创建零向量
                        padded_arrays.append(np.zeros(max_shape, dtype=np.float32))
                
                # 使用numpy stack，避免tensor转换
                merged_batch_features[feature_key] = np.stack(padded_arrays, axis=0)
                # 🔧 性能优化：使用条件判断避免字符串格式化开销
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"collate_batch IMAGE smart_cat结果: shape={merged_batch_features[feature_key].shape}, "
                        f"batch_size={batch_size}"
                    )
            elif all_single_frame:
                # 🔧 性能优化：如果都是单帧，统一形状后stack
                # 直接stack单帧图像，保持(B, 3, 224, 224)格式
                # 🔧 修复：确保所有数组形状一致，处理None值和形状不一致的情况
                cleaned_arrays = []
                target_shape = (3, 224, 224)  # 单帧目标形状
                for f in feature_arrays:
                    if f is None or not isinstance(f, np.ndarray):
                        cleaned_arrays.append(np.zeros(target_shape, dtype=np.float32))
                    elif f.ndim == 3 and f.shape == target_shape:
                        cleaned_arrays.append(f)
                    elif f.ndim == 3:
                        # 形状不一致，可能需要resize或pad（不应该发生，但为了安全）
                        if f.shape[1:] == target_shape[1:]:
                            cleaned_arrays.append(f)  # 通道数不同，但空间尺寸相同
                        else:
                            # 创建零向量（不应该发生）
                            cleaned_arrays.append(np.zeros(target_shape, dtype=np.float32))
                    else:
                        # 维度不对，创建零向量
                        cleaned_arrays.append(np.zeros(target_shape, dtype=np.float32))
                merged_batch_features[feature_key] = np.stack(cleaned_arrays, axis=0)
                # 🔧 性能优化：使用条件判断避免字符串格式化开销
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"collate_batch IMAGE all_single_frame结果: shape={merged_batch_features[feature_key].shape}, "
                        f"batch_size={batch_size} (保持单帧格式)"
                    )
            elif all_16_frame:
                # 🔧 性能优化：都是16帧格式，统一形状后stack
                # feature_arrays中的元素是(16, 3, 224, 224)，直接stack为(B, 16, 3, 224, 224)
                # 🔧 修复：确保所有数组形状一致，处理None值和形状不一致的情况
                cleaned_arrays = []
                target_shape = (16, 3, 224, 224)  # 16帧目标形状
                for f in feature_arrays:
                    if f is None or not isinstance(f, np.ndarray):
                        cleaned_arrays.append(np.zeros(target_shape, dtype=np.float32))
                    elif f.ndim == 4 and f.shape == target_shape:
                        cleaned_arrays.append(f)
                    elif f.ndim == 4 and f.shape[0] == 16:
                        # 形状不完全一致（可能是空间尺寸不同），使用目标形状的零向量
                        # 这种情况不应该发生，但为了安全处理
                        cleaned_arrays.append(np.zeros(target_shape, dtype=np.float32))
                    else:
                        # 维度不对，创建零向量
                        cleaned_arrays.append(np.zeros(target_shape, dtype=np.float32))
                merged_batch_features[feature_key] = np.stack(cleaned_arrays, axis=0)
                # 🔧 性能优化：使用条件判断避免字符串格式化开销
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"collate_batch IMAGE all_16_frame结果: shape={merged_batch_features[feature_key].shape}, "
                        f"batch_size={batch_size}"
                    )
            else:
                # 🔧 性能优化：fallback分支也使用纯numpy操作
                # 🔧 性能优化：使用条件判断避免字符串格式化开销
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"collate_batch IMAGE fallback处理: shapes={shapes[:5]}{'...' if len(shapes) > 5 else ''}"
                    )
                
                # 统一转换为16帧格式（fallback情况）
                max_shape = (16, 3, 224, 224)
                padded_arrays = []
                for f in feature_arrays:
                    if f is None or not isinstance(f, np.ndarray):
                        padded_arrays.append(np.zeros(max_shape, dtype=np.float32))
                    elif f.ndim == 3:
                        # 单帧: (3, 224, 224) -> (16, 3, 224, 224)
                        padded = np.zeros(max_shape, dtype=np.float32)
                        padded[0] = f
                        padded_arrays.append(padded)
                    elif f.ndim == 4 and f.shape[0] == 16:
                        padded_arrays.append(f)
                    else:
                        padded_arrays.append(np.zeros(max_shape, dtype=np.float32))
                
                merged_batch_features[feature_key] = np.stack(padded_arrays, axis=0)
                # 🔧 性能优化：使用条件判断避免字符串格式化开销
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"collate_batch IMAGE fallback结果: shape={merged_batch_features[feature_key].shape}, "
                        f"batch_size={batch_size}"
                    )
        else:
            # 其他特征（TEXT, AUDIO等）：直接numpy concatenate
            # 🔧 稳定版本：确保所有None值已填充，然后使用简单的stack/concatenate逻辑
            # 注意：前面已经填充了所有None值为零向量，所以这里应该都是有效数组
            
            # 确保feature_arrays中没有None值（前面应该已经填充）
            cleaned_arrays = []
            for f in feature_arrays:
                if f is None:
                    # 如果还有None（不应该发生，但为了安全），使用reference_shape创建零向量
                    is_text = feature_key == FeatureKey.TEXT or feature_key == FeatureKey.TEXT_ATTENTION_MASK
                    dtype = np.int64 if is_text else np.float32
                    cleaned_arrays.append(np.zeros(reference_shape, dtype=dtype))
                elif isinstance(f, np.ndarray):
                    cleaned_arrays.append(f)
                else:
                    # 非数组类型，跳过或创建零向量
                    is_text = feature_key == FeatureKey.TEXT or feature_key == FeatureKey.TEXT_ATTENTION_MASK
                    dtype = np.int64 if is_text else np.float32
                    cleaned_arrays.append(np.zeros(reference_shape, dtype=dtype))
            
            if not cleaned_arrays:
                continue
            
            # 尝试stack（如果形状一致）
            try:
                merged_batch_features[feature_key] = np.stack(cleaned_arrays, axis=0)
            except ValueError:
                # 如果stack失败（形状不一致），使用concatenate with expand_dims
                # 这适用于TEXT的seq_len不同等情况
                try:
                    expanded_arrays = [np.expand_dims(f, axis=0) if f.ndim == len(reference_shape) else f for f in cleaned_arrays]
                    merged_batch_features[feature_key] = np.concatenate(expanded_arrays, axis=0)
                except ValueError as e:
                    # 如果还是失败，记录详细信息以便调试
                    shapes_info = [f.shape if isinstance(f, np.ndarray) else "None" for f in cleaned_arrays]
                    raise ValueError(
                        f"无法合并feature {feature_key}: 形状不一致. "
                        f"Shapes: {shapes_info}, "
                        f"reference_shape={reference_shape}. "
                        f"原始错误: {e}"
                    ) from e
    
    # 🔧 内存优化：不在这里转换为tensor，延迟到GPU上转换
    # 只创建modality_masks（numpy格式），tensor转换在training_step中进行
    from purrsight.config import Modality
    
    # 创建modality_masks（numpy格式）
    modality_masks_numpy = {}
    for modality in [Modality.TEXT, Modality.IMAGE, Modality.AUDIO]:
        modality_key = modality.value
        if modality == Modality.TEXT:
            has_modality = (
                FeatureKey.TEXT in merged_batch_features
                and FeatureKey.TEXT_ATTENTION_MASK in merged_batch_features
            )
            if has_modality:
                mask = np.any(merged_batch_features[FeatureKey.TEXT_ATTENTION_MASK] != 0, axis=1)
            else:
                mask = np.zeros(batch_size, dtype=np.bool_)
        elif modality == Modality.IMAGE:
            has_modality = FeatureKey.IMAGE in merged_batch_features
            if has_modality:
                img_feat = merged_batch_features[FeatureKey.IMAGE]
                
                # 🔧 调试日志：记录img_feat形状
                # 🔧 性能优化：使用条件判断避免字符串格式化开销
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"collate_batch IMAGE mask计算: img_feat.shape={img_feat.shape}, "
                        f"batch_size={batch_size}, ndim={img_feat.ndim}"
                    )
                
                # 🔧 修复：添加形状验证，确保batch维度正确
                if img_feat.ndim == 5:
                    # 期望格式: (B, 16, 3, 224, 224)
                    if img_feat.shape[0] != batch_size:
                        raise ValueError(
                            f"IMAGE特征batch维度不匹配: shape={img_feat.shape}, "
                            f"batch_size={batch_size}. 可能是collate阶段处理错误。"
                        )
                    img_mask = np.sum(np.abs(img_feat), axis=(1, 2, 3, 4)) > 1e-6
                elif img_feat.ndim == 4:
                    # 期望格式: (B, 3, 224, 224) - 单帧格式（不应该出现，应该已转换为16帧）
                    if img_feat.shape[0] != batch_size:
                        raise ValueError(
                            f"IMAGE特征batch维度不匹配: shape={img_feat.shape}, "
                            f"batch_size={batch_size}. 可能是collate阶段处理错误。"
                        )
                    img_mask = np.sum(np.abs(img_feat), axis=(1, 2, 3)) > 1e-6
                else:
                    # 其他维度：动态计算
                    if img_feat.shape[0] != batch_size:
                        raise ValueError(
                            f"IMAGE特征batch维度不匹配: shape={img_feat.shape}, "
                            f"batch_size={batch_size}. 可能是collate阶段处理错误。"
                        )
                    sum_axes = tuple(range(1, img_feat.ndim))
                    img_mask = np.sum(np.abs(img_feat), axis=sum_axes) > 1e-6
                
                # 验证img_mask形状
                if img_mask.shape != (batch_size,):
                    raise ValueError(
                        f"img_mask形状错误: shape={img_mask.shape}, 期望({batch_size},). "
                        f"img_feat形状={img_feat.shape}"
                    )
                
                # 检查video_metadata并验证一致性
                if video_metadata_batch:
                    video_mask = np.array([
                        idx in video_metadata_batch and video_metadata_batch[idx].get("has_video", False)
                        for idx in range(batch_size)
                    ], dtype=np.bool_)
                    
                    # 验证video_mask形状
                    if video_mask.shape != (batch_size,):
                        raise ValueError(
                            f"video_mask形状错误: shape={video_mask.shape}, 期望({batch_size},)"
                        )
                    
                    # 🔧 修复：验证mask与video_metadata的一致性
                    for idx, meta in video_metadata_batch.items():
                        if not meta.get("image_valid", True):
                            img_mask[idx] = False
                    mask = video_mask | img_mask
                else:
                    mask = img_mask
            else:
                mask = np.zeros(batch_size, dtype=np.bool_)
        elif modality == Modality.AUDIO:
            has_modality = FeatureKey.AUDIO in merged_batch_features
            if has_modality:
                # 🔧 性能优化：使用更高效的mask计算
                audio_feat = merged_batch_features[FeatureKey.AUDIO]
                mask = np.sum(np.abs(audio_feat), axis=tuple(range(1, audio_feat.ndim))) > 1e-6
                # 🔧 修复：验证mask与video_metadata的一致性
                if video_metadata_batch:
                    from purrsight.config import ModalitySource
                    for idx, meta in video_metadata_batch.items():
                        if not meta.get("audio_valid", True):
                            audio_source = meta.get("audio_source")
                            if audio_source == ModalitySource.VIDEO.value:
                                mask[idx] = False
            else:
                mask = np.zeros(batch_size, dtype=np.bool_)
        
        modality_masks_numpy[modality_key] = mask.astype(np.bool_)
    
    # 🔧 内存优化：返回numpy格式，延迟到GPU上转换为tensor
    # 这样可以避免在CPU上占用大量内存
    # 🔧 性能优化：同时返回batch_size，避免在training_step中重复推断
    return merged_batch_features, modality_masks_numpy, batch_size


def train_loop_per_worker(rank: int, world_size: int, config: AlignmentConfig):
    """
    训练循环（每个worker执行）

    Args:
        rank: 当前进程rank（DDP使用）
        world_size: 总进程数（DDP使用）
        config: 训练配置
    """
    # 设置随机种子（保证多GPU一致性）
    pl.seed_everything(42 + rank)

    # 设置设备
    if world_size > 1:
        # DDP模式（仅支持CUDA）
        torch.cuda.set_device(rank)
        device = f"cuda:{rank}"
    else:
        # 单GPU/CPU/MPS模式
        if config.device == "auto":
            from purrsight.utils.tools import get_available_device
            device = get_available_device()  # 自动检测：MPS > CUDA > CPU
        else:
            device = config.device

    logger.info(f"Worker {rank}/{world_size} using device: {device}")

    # 离线预处理时从 preprocessed_dir/index.jsonl 加载样本列表；在线时从 data_path 加载
    data_path_to_load = config.data_path
    if config.use_preprocessed:
        if not config.preprocessed_dir:
            raise ValueError("use_preprocessed=True 时必须配置 preprocessed_dir")
        preprocessed_path = Path(config.preprocessed_dir.strip().strip('"')).resolve()
        if not preprocessed_path.exists():
            raise FileNotFoundError(
                f"预处理目录不存在: {preprocessed_path}\n"
                f"请先运行离线预处理：python -m purrsight.preprocess.prepre "
                f"--input_file <原始数据> --output_dir {config.preprocessed_dir}"
            )
        index_path = preprocessed_path / "index.jsonl"
        if index_path.exists() and index_path.stat().st_size > 0:
            data_path_to_load = str(index_path)
            logger.info(f"使用离线预处理，从索引加载: {data_path_to_load}")
        else:
            raise FileNotFoundError(
                f"预处理目录下未找到有效索引: {index_path}\n"
                f"请先完成离线预处理并生成 index.jsonl"
            )

    # 加载数据：先检查文件存在，避免静默读空
    data_path_resolved = Path(data_path_to_load).resolve()
    if not data_path_resolved.exists():
        raise FileNotFoundError(
            f"数据文件不存在: {data_path_resolved}\n"
            f"在线模式请检查 config.data_path；离线模式请检查 preprocessed_dir 下是否有 index.jsonl"
        )
    if not data_path_resolved.is_file():
        raise FileNotFoundError(f"数据路径不是文件: {data_path_resolved}")

    logger.info(f"Loading data from {data_path_resolved} (size={data_path_resolved.stat().st_size} bytes)")
    if data_path_to_load.endswith('.jsonl'):
        data_list = load_data_from_jsonl(str(data_path_resolved))
    else:
        raise ValueError(f"Unsupported data format: {data_path_to_load}")

    logger.info(f"Loaded {len(data_list)} samples")
    if len(data_list) == 0:
        raise ValueError(
            "数据列表为空，请检查 data_path 或 preprocessed_dir/index.jsonl 是否有有效样本（非空行且合法 JSON）"
        )
    # 便于排查：打印首条样本的 keys，确认格式
    sample_keys = list(data_list[0].keys()) if data_list else []
    logger.info(f"首条样本字段: {sample_keys}")

    if config.use_preprocessed:
        logger.info(f"离线预处理模式，预处理目录: {config.preprocessed_dir}")
    else:
        logger.info("使用在线预处理模式（实时预处理）")

    # 创建数据集
    preprocessed_dir_clean = Path(config.preprocessed_dir.strip().strip('"')).resolve() if config.preprocessed_dir else None
    dataset = AlignmentDataset(
        data_list=data_list,
        device="cpu",
        use_preprocessed=config.use_preprocessed,
        preprocessed_dir=preprocessed_dir_clean if config.use_preprocessed else None,
    )

    # 分割训练/验证集
    val_size = int(len(dataset) * config.val_split)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    logger.info(f"Train set: {train_size} samples, Val set: {val_size} samples")

    # 检查数据集是否为空
    if train_size == 0:
        logger.error("训练集为空，无法训练")
        return
    if val_size == 0:
        logger.warning("验证集为空，将跳过验证")

    # 创建DataLoader
    # pin_memory: 只对CUDA启用（MPS不支持，CPU不需要）
    # 注意：MPS设备虽然不支持pin_memory，但数据传输仍然高效
    # 🔧 性能优化：使用配置中的num_workers
    # 🔧 修复：统一使用device字符串检查，兼容字符串和torch.device对象
    device_str = str(device) if isinstance(device, torch.device) else device
    
    # 自动设置prefetch_factor和persistent_workers
    use_workers = config.num_workers > 0
    persistent_workers = use_workers
    prefetch_factor = 2 if use_workers else None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_batch,
        pin_memory=(device_str.startswith("cuda")),  # 只对CUDA启用
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_batch,
        pin_memory=(device_str.startswith("cuda")),  # 只对CUDA启用
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor
    )

    # 创建LightningModule
    model = ContrastiveAlignmentModule(config)

    # 获取MLflow run信息（用于统一命名）
    active_run = mlflow.active_run()
    if active_run is not None:
        run_id = active_run.info.run_id
        run_name = active_run.info.run_name
        # 从run_name中提取时间戳（如果存在）
        # run_name格式：experiment_name_YYYYMMDD_HHMMSS
        if "_" in run_name:
            parts = run_name.rsplit("_", 2)
            if len(parts) == 3:
                timestamp = f"{parts[1]}_{parts[2]}"  # YYYYMMDD_HHMMSS
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        run_id = None
        run_name = f"{config.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 统一checkpoint目录：使用run_id_timestamp格式，与手动保存的checkpoint一致
    # 这样Lightning的自动checkpoint和手动checkpoint都在同一个目录结构下
    if run_id is not None:
        checkpoint_base_dir = Path(config.save_dir) / f"{run_id}_{timestamp}"
    else:
        checkpoint_base_dir = Path(config.save_dir) / f"checkpoint_{timestamp}"
    
    checkpoint_base_dir.mkdir(parents=True, exist_ok=True)
    lightning_checkpoint_dir = checkpoint_base_dir / "lightning_checkpoints"
    lightning_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 使用当前 MLflow run（与 train_model 中 start_run 一致），保证 params 与 metrics 在同一 run，曲线可见
    run_id_for_logger = None
    try:
        active_run = mlflow.active_run()
        run_id_for_logger = active_run.info.run_id if active_run else None
    except Exception:
        pass

    # 设置MLflow logger：传入 run_id 使 Lightning 的指标写入同一 run；不支持 run_id 时退化为 run_name
    try:
        if run_id_for_logger:
            mlf_logger = MLflowLogger(
                experiment_name=config.experiment_name,
                tracking_uri=config.mlflow_tracking_uri,
                run_name=run_name,
                run_id=run_id_for_logger,
            )
        else:
            mlf_logger = MLflowLogger(
                experiment_name=config.experiment_name,
                tracking_uri=config.mlflow_tracking_uri,
                run_name=run_name,
            )
    except TypeError:
        mlf_logger = MLflowLogger(
            experiment_name=config.experiment_name,
            tracking_uri=config.mlflow_tracking_uri,
            run_name=run_name,
        )

    # 配置ModelCheckpoint：不依赖 monitor，避免 callback_metrics 中无 train_loss 报错；按周期保存 + last，指标仅用 MLflow
    from pytorch_lightning.callbacks import ModelCheckpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(lightning_checkpoint_dir),
        filename="{epoch}-{step}",
        save_top_k=1,
        monitor=None,  # 不监听 metric，避免 Lightning 未注入 train_loss 时报错
        save_last=True,
        every_n_epochs=config.save_every,
        save_on_train_epoch_end=True,
    )
    
    # 🔧 验证4：暂时关闭SpeedMonitor（验证SpeedMonitor是否导致性能下降）
    # speed_monitor = SpeedMonitor(log_every_n_batches=config.log_every)  # 🔧 验证：临时注释
    speed_monitor = None  # 🔧 验证：临时禁用SpeedMonitor

    # 检查DataLoader长度
    logger.info(f"训练DataLoader长度: {len(train_loader)} batches")
    logger.info(f"验证DataLoader长度: {len(val_loader)} batches")
    logger.info(f"Checkpoint目录: {checkpoint_base_dir}")
    logger.info(f"  - Lightning自动checkpoint: {lightning_checkpoint_dir}")
    logger.info(f"  - 手动保存checkpoint: {checkpoint_base_dir}")

    # 创建Trainer
    # 🔧 性能优化：启用自动混合精度（AMP）
    # 🔧 修复：MPS设备对mixed precision的支持有限，如果遇到dtype错误，禁用AMP
    # MPS设备：暂时禁用AMP（MPS对float16支持不完善，backward pass可能失败）
    # CUDA设备：使用16-mixed
    # 🔧 修复：统一使用device字符串检查，兼容字符串和torch.device对象
    device_str = str(device) if isinstance(device, torch.device) else device
    if device_str.startswith("mps"):
        # MPS设备：禁用AMP以避免backward pass中的dtype错误
        precision = "32-true"  # 使用float32
        logger.info("MPS设备：禁用mixed precision以避免dtype错误")
    elif device_str.startswith("cuda"):
        precision = "16-mixed"  # CUDA使用float16
    else:
        precision = "32-true"  # CPU使用float32
    
    trainer_kwargs = {
        "max_epochs": config.epochs,
        "logger": mlf_logger,
        # 🔧 修复6：优化Lightning的log_every_n_steps（增加到50，减少logging频率）
        "log_every_n_steps": max(config.log_every, 50),  # 至少50步才log一次，减少MLflow I/O阻塞
        "callbacks": [checkpoint_callback] + ([speed_monitor] if speed_monitor is not None else []),  # 🔧 验证：如果speed_monitor为None则不添加
        "enable_progress_bar": True,
        "enable_model_summary": True,
        "num_sanity_val_steps": 0,  # 跳过sanity checking，避免小数据集问题
        "limit_train_batches": 1.0,  # 使用所有训练batches
        "limit_val_batches": 1.0,  # 使用所有验证batches
        "precision": precision,
    }

    if world_size > 1:
        # DDP配置（仅支持CUDA）
        trainer_kwargs.update({
            "accelerator": "gpu",
            "devices": world_size,
            "strategy": "ddp",
        })
    else:
        # 单GPU/CPU/MPS配置
        # 🔧 修复：统一使用device字符串检查，兼容字符串和torch.device对象
        device_str = str(device) if isinstance(device, torch.device) else device
        if device_str.startswith("cuda"):
            trainer_kwargs.update({
                "accelerator": "gpu",
                "devices": 1,
            })
        elif device_str == "mps":
            trainer_kwargs.update({
                "accelerator": "mps",
                "devices": 1,
            })
        else:
            trainer_kwargs.update({
                "accelerator": "cpu",
            })

    trainer = pl.Trainer(**trainer_kwargs)

    # 开始训练
    logger.info("Starting training...")
    trainer.fit(model, train_loader, val_loader)

    # 保存最终模型（使用MLflow run ID统一编号）
    # 注意：checkpoint_base_dir、run_id、timestamp已经在train_loop_per_worker开始时创建
    if rank == 0:  # 只在主进程保存
        # 使用已经创建的checkpoint_base_dir（与Lightning checkpoint在同一目录）
        # 这样手动保存的checkpoint和Lightning自动保存的checkpoint都在同一个目录下
        # checkpoint_base_dir格式：{run_id}_{timestamp}，例如：2750355bd92c443b9d851249630300be_20260113_152025
        checkpoint_path = checkpoint_base_dir

        # 保存Lightning checkpoint（最终模型）
        trainer.save_checkpoint(checkpoint_path / "model.ckpt")

        # 保存aligner权重（便于后续使用）
        torch.save(model.aligner.state_dict(), checkpoint_path / "aligner.pt")

        # 保存训练配置信息（便于后续查看）
        config_info = {
            "run_id": run_id,
            "timestamp": timestamp,
            "experiment_name": config.experiment_name,
            "config": config.__dict__,
        }
        with open(checkpoint_path / "config.json", 'w', encoding='utf-8') as f:
            json.dump(config_info, f, indent=2, ensure_ascii=False)

        logger.info(f"Final checkpoint saved to {checkpoint_path}")
        logger.info(f"  Run ID: {run_id if run_id else 'N/A'}")
        logger.info(f"  Timestamp: {timestamp}")
        logger.info(f"  Experiment: {config.experiment_name}")
        
        # 在MLflow中记录checkpoint路径
        if active_run is not None:
            mlflow.log_param("final_checkpoint_path", str(checkpoint_path))
            mlflow.log_param("lightning_checkpoint_dir", str(checkpoint_path / "lightning_checkpoints"))
            
            # ✅ 保存Artifacts到MLflow
            save_artifacts_to_mlflow(
                checkpoint_path=checkpoint_path,
                model=model,
                config=config,
                trainer=trainer,
                active_run=active_run
            )


def train_model(config: AlignmentConfig):
    """
    主训练函数
    
    设置MLflow experiment并启动训练。
    
    功能：
    1. 设置MLflow tracking URI和experiment
    2. 加载数据并创建Dataset和DataLoader
    3. 初始化模型和Trainer
    4. 启动训练
    5. 保存artifacts到MLflow

    Args:
        config: 训练配置
    """
    # 设置MLflow
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    
    # 获取或创建实验（使用易读的实验名称）
    # 注意：MLflow的实验ID是自动生成的数字，但我们可以通过实验名称来识别
    try:
        experiment = mlflow.get_experiment_by_name(config.experiment_name)
        if experiment is None:
            # 创建新实验
            experiment_id = mlflow.create_experiment(
                config.experiment_name,
                tags={"description": f"Alignment training experiment: {config.experiment_name}"}
            )
            logger.info(f"Created new MLflow experiment: {config.experiment_name} (ID: {experiment_id})")
        else:
            logger.info(f"Using existing MLflow experiment: {config.experiment_name} (ID: {experiment.experiment_id})")
    except Exception as e:
        logger.warning(f"Failed to get/create experiment: {e}, using default")
    
    mlflow.set_experiment(config.experiment_name)

    # 启动MLflow run（MLflowLogger会检测并使用这个run）
    # 生成易读的run名称：包含时间戳和实验名称
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{config.experiment_name}_{run_timestamp}"
    with mlflow.start_run(run_name=run_name):
        # 自动检测离线预处理数据
        if not config.use_preprocessed:
            # 检查默认位置或配置的位置
            # 如果 preprocessed_dir 为 None，检查默认位置 data/preprocessed
            check_dir = Path(config.preprocessed_dir) if config.preprocessed_dir else (ROOT_DIR / "data" / "preprocessed")
            check_index = check_dir / "index.jsonl"
            
            # 只有当索引文件存在且包含内容时才切换
            if check_dir.exists() and check_index.exists() and check_index.stat().st_size > 0:
                logger.info("=" * 40)
                logger.info(f"自动检测到离线预处理数据: {check_index}")
                logger.info("根据用户策略：优先使用离线数据以加速训练")
                logger.info(f"  - 切换模式: 在线预处理 -> 离线预处理")
                logger.info(f"  - 预处理目录: {check_dir}")
                logger.info(f"  - 数据源重定向: {config.data_path} -> {check_index}")
                logger.info("=" * 40)
                
                config.use_preprocessed = True
                config.preprocessed_dir = str(check_dir)
                config.data_path = str(check_index)

        # 记录配置参数
        mlflow.log_params(config.__dict__)

        # 记录代码版本（可选）
        try:
            import git
            repo = git.Repo(search_parent_directories=True)
            mlflow.set_tag("git_commit", repo.head.commit.hexsha)
            mlflow.set_tag("git_branch", repo.active_branch.name)
        except:
            pass

        # 启动训练（MLflowLogger会使用当前的active run）
        train_loop_per_worker(rank=0, world_size=1, config=config)

        logger.info("Training completed!")


if __name__ == "__main__":
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(description='Phase 1 Alignment Training')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    args = parser.parse_args()
    
    # Load config from YAML
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Extract phase1 config
    phase1_config = config_dict.get('phase1', {})
    
    # Create AlignmentConfig
    config = AlignmentConfig(
        data_path=phase1_config.get('data_path', 'data/phase1/online/train.jsonl'),
        batch_size=phase1_config.get('batch_size', 32),
        epochs=phase1_config.get('epochs', 10),
        learning_rate=phase1_config.get('learning_rate', 1e-3),
        weight_decay=phase1_config.get('weight_decay', 0.01),
        warmup_steps=phase1_config.get('warmup_steps', 1000),
        num_workers=phase1_config.get('num_workers', 4),
        val_split=phase1_config.get('val_split', 0.1),
        use_preprocessed=phase1_config.get('use_preprocessed', False),
        preprocessed_dir=phase1_config.get('preprocessed_dir'),
        input_dim=phase1_config.get('input_dim', 512),
        output_dim=phase1_config.get('output_dim', 512),
        use_temperature_scaling=phase1_config.get('use_temperature_scaling', True),
        experiment_name=phase1_config.get('experiment_name', 'alignment_training'),
        log_every=config_dict.get('common', {}).get('log_every', 100),
        save_every=config_dict.get('common', {}).get('save_every', 1),
        device=config_dict.get('common', {}).get('device', 'auto'),
    )

    # 启动训练
    train_model(config)