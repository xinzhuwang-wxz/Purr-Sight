"""
PyTorch Lightning 模块

封装对比学习对齐训练的 LightningModule，实现多模态特征对齐的端到端训练。

包含：
- adapt_image_frames: FrameAdapter函数（已废弃，仅用于向后兼容）
- ContrastiveAlignmentModule: 对比学习对齐训练的Lightning模块

数据流（最新版本）：
1. Dataset.__getitem__() 返回numpy格式单样本特征（无batch维度，无modality_masks）
2. collate_batch() 合并为batch格式numpy数组，创建modality_masks（numpy格式）
3. training_step() 在GPU上转换为tensor
4. encode_batch() 进行batch编码（利用GPU并行，16帧格式已统一）
5. forward() 对齐特征，推断modality_presence
6. loss_fn() 计算InfoNCE损失
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
from typing import Dict, Tuple, Any, Optional, List, Union

from purrsight.alignment import ContrastiveAligner, ContrastiveLoss
from purrsight.encoder import _ImageEncoder, _TextEncoder, _AudioEncoder
from purrsight.config import Modality, FeatureKey
from purrsight.utils.eval.eval_align import ContrastiveMetrics
from purrsight.utils.logging import logger
from .train_align_conf import AlignmentConfig


def adapt_image_frames(image_tensor: torch.Tensor) -> Tuple[torch.Tensor, str]:
    """
    适配图像帧格式，统一为16帧格式（FrameAdapter）
    
    ⚠️ 已废弃：此函数仅用于向后兼容。
    
    主要逻辑已内联到encode_batch()中。格式转换现在在collate阶段（smart_cat）完成，
    此函数仅在向后兼容分支中使用（当检测到单帧格式时）。
    
    Args:
        image_tensor: 图像tensor，可能是：
            - (B, 3, H, W): 单帧格式（向后兼容）
            - (B, 16, 3, H, W): 16帧格式（正常情况）
    
    Returns:
        (adapted_tensor, frame_mode)元组：
        - adapted_tensor: 适配后的tensor，统一为(B, 16, 3, H, W)
        - frame_mode: 帧模式，"single_frame_expanded" 或 "video_frames"
    """
    # 🔧 P2修复：collate阶段已统一格式，这里只做验证和向后兼容处理
    if image_tensor.dim() == 5 and image_tensor.shape[1] == 16:
        # 已经是16帧格式（collate阶段已处理）
        adapted_tensor = image_tensor
        # 检测是否为单帧扩展：检查16帧是否相同
        frame_mode = "video_frames"  # 默认假设是视频帧
        if image_tensor.shape[0] > 0:
            first_frame = image_tensor[0, 0]
            last_frame = image_tensor[0, -1]
            if torch.allclose(first_frame, last_frame, atol=1e-6):
                frame_mode = "single_frame_expanded"
    elif image_tensor.dim() == 4:
        # 单帧格式（不应该出现，collate阶段应已转换）
        # 但为了向后兼容，仍然处理
        adapted_tensor = image_tensor.unsqueeze(1).expand(-1, 16, -1, -1, -1)
        frame_mode = "single_frame_expanded"
    else:
        # 未知格式，尝试直接使用
        adapted_tensor = image_tensor
        frame_mode = "unknown"
    
    return adapted_tensor, frame_mode


class ContrastiveAlignmentModule(pl.LightningModule):
    """
    对比学习对齐的 LightningModule
    
    在 batch 级别进行编码，利用 GPU 并行能力。
    
    Attributes:
        config: 训练配置
        image_encoder: 图像编码器（冻结）
        text_encoder: 文本编码器（冻结）
        audio_encoder: 音频编码器（冻结）
        aligner: 对比学习对齐器
        loss_fn: 对比损失函数
        metrics: 评估指标计算器
    """

    def __init__(self, config: AlignmentConfig):
        """
        初始化 LightningModule

        Args:
            config: 训练配置
        """
        super().__init__()

        # 保存配置
        self.config = config

        # 初始化编码器（冻结，eval模式）
        # 编码器在 LightningModule 中初始化一次，所有 batch 共享
        self._init_encoders()

        # 获取编码器实际输出维度
        text_dim = 384  # MiniLM固定输出
        image_dim = self.image_encoder.feature_dim  # 从ImageEncoder获取
        audio_dim = 2048  # Cnn14固定输出

        # 初始化对齐器（传入各模态的实际维度）
        self.aligner = ContrastiveAligner(
            text_input_dim=text_dim,
            image_input_dim=image_dim,
            audio_input_dim=audio_dim,
            output_dim=config.output_dim,
            use_temperature_scaling=config.use_temperature_scaling,
        )

        # 初始化损失函数
        self.loss_fn = ContrastiveLoss(
            use_temperature_scaling=config.use_temperature_scaling
        )

        # 初始化评估指标计算器
        self.metrics = ContrastiveMetrics(k_values=[1, 5, 10])

        # 🔧 性能优化：缓存帧权重，避免重复计算
        self._single_frame_weights = None
        self._video_frame_weights = None
        
        # 🔧 性能优化：dtype检查标志，只在第一次或出现问题时检查
        self._dtype_check_needed = True  # 第一次batch需要检查
        
        # 🔧 优化：验证指标计算控制
        self.compute_val_metrics = config.compute_val_metrics
        self.val_metrics_every_n_epochs = config.val_metrics_every_n_epochs

        # 保存超参数（自动记录到 MLflow）
        self.save_hyperparameters(config.__dict__)

    def _init_encoders(self):
        """
        初始化并冻结编码器
        
        编码器会自动移动到模型所在的设备（GPU/CPU），并设置为eval模式。
        所有编码器参数被冻结，不参与梯度更新。
        """
        # 编码器会自动移动到模型所在的设备（GPU/CPU）
        self.image_encoder = _ImageEncoder().eval()
        self.text_encoder = _TextEncoder().eval()
        self.audio_encoder = _AudioEncoder().eval()

        # 冻结编码器参数
        for encoder in [self.image_encoder, self.text_encoder, self.audio_encoder]:
            for param in encoder.parameters():
                param.requires_grad = False

    def encode_batch(
        self,
        tensor_features: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Batch 级别编码：将预处理后的 tensor_features 编码为 encoder_outputs
        
        使用FrameAdapter统一处理单帧图像和16帧视频，确保所有图像输入到编码器都是16帧格式。
        
        Args:
            tensor_features: 预处理后的特征字典，键为FeatureKey，值为形状为(B, ...)的tensor
                数据在CPU上，会在函数内部批量传输到GPU/MPS

        Returns:
            encoder_outputs: 编码器输出字典，键为模态名称，值为不同形状的tensor
                Text: (B, 384), Image: (B, feature_dim), Audio: (B, 2048)
            
        注意：视频已分解为Image和Audio，不再有独立的Video编码输出。
        """
        encoder_outputs = {}
        device = self.device  # LightningModule 的设备

        # 🔧 性能优化：批量传输所有tensor到device，检查设备避免重复传输
        # 跳过非tensor类型（如_video_metadata字典）
        tensor_features_gpu = {}
        for key, value in tensor_features.items():
            if value is not None:
                if isinstance(value, torch.Tensor):
                    # 🔧 性能优化：检查设备，避免重复传输
                    if value.device != device:
                        tensor_features_gpu[key] = value.to(device)
                    else:
                        tensor_features_gpu[key] = value  # 已在目标设备，跳过传输
                else:
                    # 非tensor类型（如_video_metadata字典）直接保留，不传输到GPU
                    tensor_features_gpu[key] = value
            else:
                tensor_features_gpu[key] = None

        # Batch 编码（利用 GPU 并行能力）
        with torch.no_grad():
            # 文本编码
            if FeatureKey.TEXT in tensor_features_gpu:
                text_input = tensor_features_gpu[FeatureKey.TEXT]
                text_mask = tensor_features_gpu.get(FeatureKey.TEXT_ATTENTION_MASK)
                encoder_outputs[Modality.TEXT.value] = self.text_encoder(text_input, text_mask)

            # 图像编码
            # 🔧 P2修复：格式转换已在collate阶段（smart_cat）完成，这里直接处理16帧格式
            if FeatureKey.IMAGE in tensor_features_gpu:
                image_input = tensor_features_gpu[FeatureKey.IMAGE]
                
                # 🔧 P2修复：collate阶段已统一为16帧格式，这里直接处理
                # 验证输入格式（应该是16帧格式(B, 16, 3, 224, 224)）
                if image_input.dim() == 5 and image_input.shape[1] == 16:
                    # 16帧格式（包括单帧扩展和视频帧）
                    # 🔧 性能优化：简化frame_mode检测，只在batch_size > 0时检测
                    # 使用更快的比较方法（比较sum而不是allclose）
                    frame_mode = "video_frames"  # 默认假设是视频帧
                    if image_input.shape[0] > 0:
                        # 检查第一个样本的第一帧和最后一帧是否相同（单帧扩展的特征）
                        # 使用sum比较比allclose更快，且对于单帧扩展（完全相同）足够准确
                        first_frame = image_input[0, 0]
                        last_frame = image_input[0, -1]
                        # 比较sum比allclose快，且对于完全相同的情况足够准确
                        if torch.equal(first_frame, last_frame) or (first_frame.sum() - last_frame.sum()).abs() < 1e-4:
                            frame_mode = "single_frame_expanded"
                    
                    B, T, C, H, W = image_input.shape
                    # 🔧 修复：使用reshape而不是view（因为expand创建的是非连续视图）
                    # Reshape为(B*T, C, H, W)
                    image_input_flat = image_input.reshape(B * T, C, H, W)
                    # 编码：输出(B*T, feature_dim)
                    encoded_frames = self.image_encoder(image_input_flat)
                    # Reshape回(B, T, feature_dim)
                    encoded_frames = encoded_frames.reshape(B, T, -1)
                    
                    # 🔧 P1修复：单帧图像使用均匀权重，16帧视频使用线性递增权重
                    # 🔧 性能优化：缓存权重，避免重复计算
                    if frame_mode == "single_frame_expanded":
                        # 单帧图像：使用均匀权重（数学等价：mean([x, x, ..., x]) = x）
                        if self._single_frame_weights is None or self._single_frame_weights.shape[0] != T:
                            self._single_frame_weights = torch.ones(T, device=encoded_frames.device) / T
                        weights = self._single_frame_weights
                    else:
                        # 16帧视频：使用线性递增权重（后面的帧权重更高），保留时序信息
                        # 权重范围 [0.5, 1.0]，归一化后求和
                        if self._video_frame_weights is None or self._video_frame_weights.shape[0] != T:
                            weights_raw = torch.linspace(0.5, 1.0, T, device=encoded_frames.device)
                            self._video_frame_weights = weights_raw / weights_raw.sum()  # 归一化
                        weights = self._video_frame_weights
                    
                    encoder_outputs[Modality.IMAGE.value] = (encoded_frames * weights.view(1, T, 1)).sum(dim=1)
                elif image_input.dim() == 4:
                    # 🔧 修复：单帧格式是正常的（如果batch都是单帧，collate阶段会保持单帧格式）
                    # 使用adapt_image_frames进行转换
                    adapted_image, frame_mode = adapt_image_frames(image_input)
                    if adapted_image.dim() == 5 and adapted_image.shape[1] == 16:
                        B, T, C, H, W = adapted_image.shape
                        image_input_flat = adapted_image.reshape(B * T, C, H, W)
                        encoded_frames = self.image_encoder(image_input_flat)
                        encoded_frames = encoded_frames.reshape(B, T, -1)
                        # 🔧 性能优化：使用缓存的权重
                        if frame_mode == "single_frame_expanded":
                            if self._single_frame_weights is None or self._single_frame_weights.shape[0] != T:
                                self._single_frame_weights = torch.ones(T, device=encoded_frames.device) / T
                            weights = self._single_frame_weights
                        else:
                            if self._video_frame_weights is None or self._video_frame_weights.shape[0] != T:
                                weights_raw = torch.linspace(0.5, 1.0, T, device=encoded_frames.device)
                                self._video_frame_weights = weights_raw / weights_raw.sum()
                            weights = self._video_frame_weights
                        encoder_outputs[Modality.IMAGE.value] = (encoded_frames * weights.view(1, T, 1)).sum(dim=1)
                    else:
                        encoder_outputs[Modality.IMAGE.value] = self.image_encoder(adapted_image)
                else:
                    # 未知格式，尝试直接编码
                    logger.warning(f"未知的图像输入格式: shape={image_input.shape}, dim={image_input.dim()}")
                    encoder_outputs[Modality.IMAGE.value] = self.image_encoder(image_input)

            # 音频编码
            if FeatureKey.AUDIO in tensor_features_gpu:
                audio_input = tensor_features_gpu[FeatureKey.AUDIO]
                encoder_outputs[Modality.AUDIO.value] = self.audio_encoder(audio_input)

        # 🔧 修复：确保所有模态的batch size一致（不应该不一致）
        batch_size = None
        for tensor in encoder_outputs.values():
            if tensor is not None:
                batch_size = tensor.shape[0]
                break

        if batch_size is None:
            raise ValueError("Batch 中没有任何有效模态")

        # 🔧 修复：验证一致性，如果不一致直接报错（不应该发生）
        for modality_key, tensor in encoder_outputs.items():
            if tensor is not None and tensor.shape[0] != batch_size:
                raise ValueError(
                    f"编码器输出batch size不一致（这是bug，不应该发生）: "
                    f"{modality_key} shape={tensor.shape}, expected batch_size={batch_size}. "
                    f"请检查预处理和编码逻辑。可能的原因："
                    f"1. 预处理阶段batch size不一致"
                    f"2. 编码器处理了不同数量的样本"
                )

        # 为缺失的模态创建零向量（注意：VIDEO已分解，不再需要）
        for modality in [Modality.TEXT, Modality.IMAGE, Modality.AUDIO]:
            if modality.value not in encoder_outputs:
                # 根据模态确定正确的维度
                if modality == Modality.TEXT:
                    dim = 384
                elif modality == Modality.IMAGE:
                    dim = self.image_encoder.feature_dim
                elif modality == Modality.AUDIO:
                    dim = 2048
                else:
                    dim = 512  # fallback
                
                encoder_outputs[modality.value] = torch.zeros(
                    batch_size, dim, device=device, dtype=torch.float32
                )

        return encoder_outputs

    def forward(
        self,
        encoder_outputs: Dict[str, torch.Tensor],
        modality_masks: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, bool]]:
        """
        前向传播：对齐特征到统一语义空间

        Args:
            encoder_outputs: 编码器输出字典，键为模态名称，值为不同形状的tensor
            modality_masks: 模态mask字典，可选（sample-level mask，形状为(B,)）

        Returns:
            (aligned_features, modality_presence)元组：
            - aligned_features: 对齐后的特征字典，键为模态名称，值为形状为(B, 512)的tensor
            - modality_presence: batch-level模态存在标记字典，用于传递给loss_fn
        """
        # 🔧 优化：统一在forward中推断modality_presence，避免在loss_fn中重复推断
        if modality_masks is not None:
            modality_presence = {
                k: bool(v.any().item()) for k, v in modality_masks.items()
            }
        else:
            # 如果没有mask，根据encoder_outputs推断
            modality_presence = {
                k: v is not None and not torch.all(v == 0)
                for k, v in encoder_outputs.items()
            }

        aligned_features = self.aligner(encoder_outputs, modality_presence)
        return aligned_features, modality_presence

    def training_step(
        self,
        batch: Tuple[Dict[str, Union[np.ndarray, torch.Tensor]], Dict[str, Union[np.ndarray, torch.Tensor]], int],
        batch_idx: int
    ) -> torch.Tensor:
        """
        训练步骤

        Args:
            batch: (batch_features, modality_masks, batch_size)元组
                batch_features: 预处理后的特征字典（numpy或tensor格式），键为FeatureKey
                modality_masks: 模态mask字典（numpy或tensor格式），键为模态名称
                batch_size: batch大小（由collate_batch提供，避免重复推断）
            batch_idx: batch索引

        Returns:
            损失值
        """
        # 1. 记录GPU内存使用情况（只在 rank 0 记录）
        if self.trainer.is_global_zero and batch_idx % 50 == 0:
             # GPU Memory
             if torch.cuda.is_available():
                 max_memory = torch.cuda.max_memory_allocated() / 1024**3
                 current_memory = torch.cuda.memory_allocated() / 1024**3
                 self.log("gpu_memory_max_gb", max_memory, prog_bar=False)
                 self.log("gpu_memory_current_gb", current_memory, prog_bar=False)

        batch_features, modality_masks, batch_size = batch

        # 🔧 内存优化：在GPU上转换为tensor，避免在CPU上占用内存
        # 🔧 性能优化：批量转换，减少多次.to(device)调用
        device = self.device
        
        # 转换batch_features为GPU tensor
        tensor_features = {}
        # 🔧 性能优化：先收集所有需要转换的numpy数组，然后批量转换
        numpy_items = []
        for key, feat in batch_features.items():
            if key == "_video_metadata" or key == "_modality_sources":
                tensor_features[key] = feat  # 保留metadata字典
                continue
            
            if feat is not None:
                if isinstance(feat, np.ndarray):
                    # 🔧 性能优化：使用torch.as_tensor进行零拷贝转换（如果可能）
                    # 🔧 修复：确保所有非整数类型都转换为float32（mixed precision训练需要）
                    if feat.dtype == np.int64:
                        tensor_feat = torch.from_numpy(feat).long()
                    elif feat.dtype == np.int32:
                        tensor_feat = torch.from_numpy(feat).int()
                    else:
                        # 处理int8, uint8, int16, uint16, float16等类型，统一转换为float32
                        if feat.dtype in [np.int8, np.uint8, np.int16, np.uint16, np.float16]:
                            tensor_feat = torch.from_numpy(feat.astype(np.float32)).float()
                        elif feat.dtype == np.float32:
                            tensor_feat = torch.as_tensor(feat).float()
                        else:
                            tensor_feat = torch.from_numpy(feat.astype(np.float32)).float()
                    numpy_items.append((key, tensor_feat))
                elif isinstance(feat, torch.Tensor):
                    # 🔧 修复：确保tensor的dtype正确（mixed precision训练需要float32/int64/int32/bool）
                    if feat.dtype not in [torch.float32, torch.int64, torch.int32, torch.bool]:
                        # 如果不是标准类型，转换为float32
                        if feat.dtype in [torch.int8, torch.uint8, torch.int16, torch.uint16, torch.float16]:
                            feat = feat.float()
                        else:
                            feat = feat.float()
                    numpy_items.append((key, feat))
                else:
                    tensor_features[key] = feat
        
        # 🔧 性能优化：批量传输到GPU，使用异步传输（CUDA）或同步传输（MPS/CPU）
        # 🔧 修复：device是torch.device对象，使用device.type而不是startswith
        if device.type == "cuda":
            # CUDA设备：使用non_blocking异步传输，提高效率
            for key, tensor_feat in numpy_items:
                tensor_features[key] = tensor_feat.to(device, non_blocking=True)
        else:
            # MPS或CPU设备：同步传输
            for key, tensor_feat in numpy_items:
                tensor_features[key] = tensor_feat.to(device)
        
        # 转换modality_masks为GPU tensor
        modality_masks_gpu = {}
        mask_items = []
        for k, v in modality_masks.items():
            if isinstance(v, np.ndarray):
                mask_items.append((k, torch.from_numpy(v).bool()))
            elif isinstance(v, torch.Tensor):
                # 🔧 修复：确保mask tensor是bool类型
                if v.dtype != torch.bool:
                    v = v.bool()
                mask_items.append((k, v))
            else:
                modality_masks_gpu[k] = v
        
        # 🔧 性能优化：批量传输masks到GPU，使用异步传输（CUDA）
        # 🔧 修复：device是torch.device对象，使用device.type而不是startswith
        if device.type == "cuda":
            for k, v in mask_items:
                modality_masks_gpu[k] = v.to(device, non_blocking=True)
        else:
            for k, v in mask_items:
                modality_masks_gpu[k] = v.to(device)

        # 检查batch是否为空
        if not tensor_features:
            # 🔧 改进：改为debug级别，因为这是预期的边界情况（所有样本mask=False时）
            logger.debug(f"训练步骤 {batch_idx}: tensor_features 为空（可能是所有样本mask=False）")
            return torch.tensor(0.0, device=self.device, dtype=torch.float32, requires_grad=True)

        # Batch 编码（利用 GPU 并行能力）
        try:
            encoder_outputs = self.encode_batch(tensor_features)
        except Exception as e:
            logger.error(f"训练步骤 {batch_idx}: 编码失败 - {e}", exc_info=True)
            raise

        # 对齐特征
        try:
            aligned_features, modality_presence = self.forward(encoder_outputs, modality_masks_gpu)
        except Exception as e:
            logger.error(f"训练步骤 {batch_idx}: 对齐失败 - {e}", exc_info=True)
            raise

        # 计算损失
        try:
            # 🔧 优化：使用forward返回的modality_presence，避免重复推断
            total_loss, loss_dict = self.loss_fn(
                aligned_features,
                modality_presence=modality_presence,  # 使用forward推断的结果
                logit_scales=self.aligner.get_logit_scales() if self.aligner.use_temperature_scaling else None,
                modality_masks=modality_masks_gpu,  # Sample-level mask，用于精确控制损失计算
            )
        except Exception as e:
            logger.error(f"训练步骤 {batch_idx}: 损失计算失败 - {e}", exc_info=True)
            raise

        # 🔧 性能优化：条件dtype检查（只在第一次或出现问题时检查）
        if self._dtype_check_needed or total_loss.dtype != torch.float32:
            if total_loss.dtype != torch.float32:
                logger.warning(f"训练步骤 {batch_idx}: 损失dtype不正确 ({total_loss.dtype})，转换为float32")
                total_loss = total_loss.float()
                self._dtype_check_needed = True  # 出现问题，继续检查
            else:
                self._dtype_check_needed = False  # 正常，后续跳过检查
        
        # 确保损失有效且需要梯度
        if not total_loss.requires_grad:
            logger.warning(f"训练步骤 {batch_idx}: 损失不需要梯度，可能有问题")
            # 确保损失需要梯度
            total_loss = total_loss.detach().requires_grad_(True)
        
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            logger.warning(f"训练步骤 {batch_idx}: 损失为 NaN 或 Inf，使用零损失")
            total_loss = torch.tensor(0.0, device=total_loss.device, dtype=torch.float32, requires_grad=True)

        # 🔧 性能优化：batch_size已从collate_batch传入，无需重复推断
        # batch_size已在函数开始时从batch元组中解包

        # 仅通过 Lightning 的 MLflowLogger 记录（统一用 MLflow）
        self.log("train_loss", total_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, batch_size=batch_size)
        for loss_name, loss_value in loss_dict.items():
            self.log(f"train_{loss_name}", loss_value, prog_bar=False, logger=True, on_step=False, on_epoch=True, batch_size=batch_size)
        if self.trainer is not None and self.trainer.optimizers:
            lr = self.trainer.optimizers[0].param_groups[0].get("lr")
            if lr is not None:
                self.log("learning_rate", float(lr), prog_bar=False, logger=True, on_step=True, on_epoch=False)

        # 🔧 性能优化：不在每个batch计算指标，改为在epoch结束时计算
        # 指标计算移到on_train_epoch_end中，避免每个batch都计算（O(B²)复杂度）

        # 🔧 最终检查：确保返回的是float32 tensor（只在第一次或出现问题时检查）
        if batch_idx == 0 or self._dtype_check_needed:
            if not isinstance(total_loss, torch.Tensor):
                raise TypeError(f"训练步骤 {batch_idx}: loss不是tensor，而是{type(total_loss)}")
            if total_loss.dtype != torch.float32:
                logger.error(f"训练步骤 {batch_idx}: loss dtype不是float32，而是{total_loss.dtype}，强制转换")
                total_loss = total_loss.float()
                self._dtype_check_needed = True
            if not total_loss.requires_grad:
                logger.warning(f"训练步骤 {batch_idx}: loss不需要梯度，强制设置")
                total_loss = total_loss.detach().requires_grad_(True)

        # 🔧 优化：显式清理中间变量，帮助GC回收（防止内存泄漏）
        del tensor_features, modality_masks_gpu, encoder_outputs, aligned_features, modality_presence, loss_dict
        
        # 🔧 验证3：强制释放batch引用（多模态任务中非常关键）
        # 强制释放batch中的所有引用
        if isinstance(batch_features, dict):
            for k in list(batch_features.keys()):
                del batch_features[k]
        del batch_features
        if isinstance(modality_masks, dict):
            for k in list(modality_masks.keys()):
                del modality_masks[k]
        del modality_masks
        del batch  # 释放整个batch元组引用
        
        # 🔧 GC Optimization: Removed explicit gc.collect() calls.
        # Calling gc.collect() every N steps is a performance anti-pattern.
        # It pauses the entire training loop and clears CPU memory, not GPU memory.
        
        # 注意：不删除total_loss，因为需要返回

        return total_loss

    def validation_step(
        self,
        batch: Tuple[Dict[str, Union[np.ndarray, torch.Tensor]], Dict[str, Union[np.ndarray, torch.Tensor]], int],
        batch_idx: int
    ) -> torch.Tensor:
        """
        验证步骤

        Args:
            batch: (batch_features, modality_masks, batch_size)元组
                batch_features: 预处理后的特征字典，键为FeatureKey（可能是numpy数组或tensor）
                modality_masks: 模态mask字典，键为模态名称（可能是numpy数组或tensor）
                batch_size: batch大小（由collate_batch提供，避免重复推断）
            batch_idx: batch索引

        Returns:
            损失值
        """
        batch_features, modality_masks, batch_size = batch

        # 🔧 修复：validation_step也需要将numpy数组转换为tensor（与training_step一致）
        # 🔧 内存优化：在GPU上转换为tensor，避免在CPU上占用内存
        # 🔧 性能优化：批量转换，减少多次.to(device)调用（与training_step保持一致）
        device = self.device
        
        # 转换batch_features为GPU tensor
        tensor_features = {}
        # 🔧 性能优化：先收集所有需要转换的numpy数组，然后批量转换
        numpy_items = []
        for key, feat in batch_features.items():
            if key == "_video_metadata" or key == "_modality_sources":
                tensor_features[key] = feat  # 保留metadata字典
                continue
            
            if feat is not None:
                if isinstance(feat, np.ndarray):
                    # 🔧 性能优化：使用torch.as_tensor进行零拷贝转换（如果可能）
                    # 🔧 修复：确保所有非整数类型都转换为float32（mixed precision训练需要）
                    if feat.dtype == np.int64:
                        tensor_feat = torch.from_numpy(feat).long()
                    elif feat.dtype == np.int32:
                        tensor_feat = torch.from_numpy(feat).int()
                    else:
                        # 处理int8, uint8, int16, uint16, float16等类型，统一转换为float32
                        if feat.dtype in [np.int8, np.uint8, np.int16, np.uint16, np.float16]:
                            tensor_feat = torch.from_numpy(feat.astype(np.float32)).float()
                        elif feat.dtype == np.float32:
                            tensor_feat = torch.as_tensor(feat).float()
                        else:
                            tensor_feat = torch.from_numpy(feat.astype(np.float32)).float()
                    numpy_items.append((key, tensor_feat))
                elif isinstance(feat, torch.Tensor):
                    # 🔧 修复：确保tensor的dtype正确（mixed precision训练需要float32/int64/int32/bool）
                    if feat.dtype not in [torch.float32, torch.int64, torch.int32, torch.bool]:
                        # 如果不是标准类型，转换为float32
                        if feat.dtype in [torch.int8, torch.uint8, torch.int16, torch.uint16, torch.float16]:
                            feat = feat.float()
                        else:
                            feat = feat.float()
                    numpy_items.append((key, feat))
                else:
                    tensor_features[key] = feat
        
        # 🔧 性能优化：批量传输到GPU，使用异步传输（CUDA）或同步传输（MPS/CPU）
        # 🔧 修复：device是torch.device对象，使用device.type而不是startswith
        if device.type == "cuda":
            # CUDA设备：使用non_blocking异步传输，提高效率
            for key, tensor_feat in numpy_items:
                tensor_features[key] = tensor_feat.to(device, non_blocking=True)
        else:
            # MPS或CPU设备：同步传输
            for key, tensor_feat in numpy_items:
                tensor_features[key] = tensor_feat.to(device)
        
        # 转换modality_masks为GPU tensor
        modality_masks_gpu = {}
        mask_items = []
        for k, v in modality_masks.items():
            if isinstance(v, np.ndarray):
                mask_items.append((k, torch.from_numpy(v).bool()))
            elif isinstance(v, torch.Tensor):
                # 🔧 修复：确保mask tensor是bool类型
                if v.dtype != torch.bool:
                    v = v.bool()
                mask_items.append((k, v))
            else:
                modality_masks_gpu[k] = v
        
        # 🔧 性能优化：批量传输masks到GPU，使用异步传输（CUDA）
        # 🔧 修复：device是torch.device对象，使用device.type而不是startswith
        if device.type == "cuda":
            for k, v in mask_items:
                modality_masks_gpu[k] = v.to(device, non_blocking=True)
        else:
            for k, v in mask_items:
                modality_masks_gpu[k] = v.to(device)

        # 检查batch是否为空
        if not tensor_features:
            # 🔧 改进：改为debug级别，因为这是预期的边界情况（所有样本mask=False时）
            logger.debug(f"验证步骤 {batch_idx}: tensor_features 为空（可能是所有样本mask=False）")
            return torch.tensor(0.0, device=self.device, dtype=torch.float32, requires_grad=False)

        # Batch 编码（利用 GPU 并行能力）
        try:
            encoder_outputs = self.encode_batch(tensor_features)
        except Exception as e:
            logger.error(f"验证步骤 {batch_idx}: 编码失败 - {e}", exc_info=True)
            raise

        # 对齐特征（modality_masks_gpu已在上面转换）
        try:
            aligned_features, modality_presence = self.forward(encoder_outputs, modality_masks_gpu)
        except Exception as e:
            logger.error(f"验证步骤 {batch_idx}: 对齐失败 - {e}", exc_info=True)
            raise

        # 计算损失
        try:
            # 🔧 优化：使用forward返回的modality_presence，避免重复推断
            total_loss, loss_dict = self.loss_fn(
                aligned_features,
                modality_presence=modality_presence,  # 使用forward推断的结果
                logit_scales=self.aligner.get_logit_scales() if self.aligner.use_temperature_scaling else None,
                modality_masks=modality_masks_gpu  # Sample-level mask，用于精确控制损失计算
            )
        except Exception as e:
            logger.error(f"验证步骤 {batch_idx}: 损失计算失败 - {e}", exc_info=True)
            raise

        # 🔧 性能优化：条件dtype检查（只在第一次或出现问题时检查）
        if batch_idx == 0 or self._dtype_check_needed:
            if total_loss.dtype != torch.float32:
                logger.warning(f"验证步骤 {batch_idx}: 损失dtype不正确 ({total_loss.dtype})，转换为float32")
                total_loss = total_loss.float()
                self._dtype_check_needed = True
            else:
                self._dtype_check_needed = False
        
        # 确保损失有效
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            logger.warning(f"验证步骤 {batch_idx}: 损失为 NaN 或 Inf，使用零损失")
            total_loss = torch.tensor(0.0, device=total_loss.device, dtype=torch.float32, requires_grad=False)

        # 🔧 性能优化：batch_size已从collate_batch传入，无需重复推断
        # batch_size已在函数开始时从batch元组中解包

        # 仅通过 Lightning 的 MLflowLogger 记录（统一用 MLflow）
        self.log("val_loss", total_loss, prog_bar=True, logger=True, on_step=False, on_epoch=True, batch_size=batch_size)
        for loss_name, loss_value in loss_dict.items():
            self.log(f"val_{loss_name}", loss_value, prog_bar=False, logger=True, on_step=False, on_epoch=True, batch_size=batch_size)

        # 🔧 性能优化：只在最后一个validation batch计算指标，避免每个batch都计算
        # 🔧 优化：根据配置决定是否计算指标，以及计算频率
        should_compute_metrics = False
        if self.compute_val_metrics and self.trainer is not None:
            # 检查是否是最后一个batch
            is_last_batch = False
            if hasattr(self.trainer, 'is_last_batch'):
                is_last_batch = self.trainer.is_last_batch
            elif hasattr(self.trainer, 'num_val_batches') and self.trainer.num_val_batches is not None:
                if self.trainer.num_val_batches > 0:
                    is_last_batch = (batch_idx == self.trainer.num_val_batches - 1)
            
            # 检查是否应该在这个epoch计算指标（根据val_metrics_every_n_epochs）
            if is_last_batch:
                current_epoch = self.trainer.current_epoch if hasattr(self.trainer, 'current_epoch') else 0
                if current_epoch % self.val_metrics_every_n_epochs == 0:
                    should_compute_metrics = True
        
        if should_compute_metrics:
            try:
                # 🔧 优化：记录指标计算时间（用于性能监控）
                import time
                metrics_start_time = time.time()
                
                eval_metrics = self.metrics.compute_batch_metrics(
                    aligned_features,
                    modality_masks_gpu
                )
                
                metrics_time = time.time() - metrics_start_time
                # 记录到SpeedMonitor（如果可用）
                if hasattr(self, 'trainer') and self.trainer is not None:
                    for callback in self.trainer.callbacks:
                        if hasattr(callback, 'val_metrics_times'):
                            callback.val_metrics_times.append(metrics_time)
                            # 限制历史记录数量
                            if len(callback.val_metrics_times) > callback.max_history:
                                callback.val_metrics_times = callback.val_metrics_times[-callback.max_history:]
                
                logger.debug(f"验证指标计算耗时: {metrics_time*1000:.1f}ms")
                
                # 记录评估指标（只在epoch结束时记录）
                for metric_name, metric_value in eval_metrics.items():
                    self.log(
                        f"val_{metric_name}",
                        metric_value,
                        prog_bar=False,
                        logger=True,
                        on_step=False,
                        on_epoch=True,
                        batch_size=batch_size
                    )
            except Exception as e:
                logger.warning(f"验证步骤 {batch_idx}: 评估指标计算失败 - {e}")

        # 🔧 优化：显式清理中间变量，帮助GC回收（防止内存泄漏）
        del tensor_features, modality_masks_gpu, encoder_outputs, aligned_features, modality_presence, loss_dict
        # 注意：不删除total_loss，因为需要返回

        return total_loss

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        配置优化器和学习率调度器

        Returns:
            包含optimizer和scheduler的字典
        """
        # 优化器：只优化对齐模块的参数（编码器已冻结）
        optimizer = torch.optim.AdamW(
            self.aligner.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        # 学习率调度器：余弦退火
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.epochs,
            eta_min=self.config.learning_rate * 0.1
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1
            }
        }

    def on_train_epoch_end(self) -> None:
        """Epoch 结束时记录聚合指标，由 MLflowLogger 写入 MLflow。"""
        if self.trainer is None:
            return
        cm = self.trainer.callback_metrics
        for key in ("train_loss", "train_loss_epoch"):
            if key in cm:
                self.log("train_loss_epoch", cm[key], prog_bar=False, logger=True, on_step=False, on_epoch=True)
                break
        if self.trainer.optimizers:
            lr = self.trainer.optimizers[0].param_groups[0].get("lr")
            if lr is not None:
                self.log("learning_rate_epoch", float(lr), prog_bar=False, logger=True, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self) -> None:
        """Validation epoch 结束时记录 val_loss，由 MLflowLogger 写入 MLflow。"""
        if self.trainer is None:
            return
        cm = self.trainer.callback_metrics
        for key in ("val_loss", "val_loss_epoch"):
            if key in cm:
                self.log("val_loss_epoch", cm[key], prog_bar=False, logger=True, on_step=False, on_epoch=True)
                break