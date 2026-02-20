"""
图像预处理模块：支持单图和视频帧批处理

针对MobileNetV4-small编码器优化，确保预处理参数与编码器匹配。

包含：
- _ImageProcessor: 图像预处理器类

参考仓库：
- pytorch/vision (TorchVision.transforms)
- timm (MobileNetV4)
"""

import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
from typing import Union, Optional, List


class _ImageProcessor:
    """
    图像预处理器：支持单图和视频帧批处理
    
    预处理参数与MobileNetV4-small编码器对齐：
    - 输入尺寸：224x224
    - 归一化：ImageNet标准参数（mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]）
    
    Attributes:
        mean: 归一化均值
        std: 归一化标准差
        input_size: 输入图像尺寸
        transform: 图像变换pipeline
    """
    
    def __init__(self, model_name: Optional[str] = None):
        """
        初始化图像预处理变换
        
        Args:
            model_name: 可选的模型名称，如果提供且timm可用，会尝试从模型配置获取预处理参数。
                        目前的参数是与MobileNetV4-small对齐的
        """
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.input_size = 224
        
        if model_name:
            try:
                import timm
                model = timm.create_model(model_name, pretrained=False)
                if hasattr(model, 'default_cfg') and model.default_cfg:
                    cfg = model.default_cfg
                    if 'mean' in cfg and cfg['mean']:
                        self.mean = cfg['mean']
                    if 'std' in cfg and cfg['std']:
                        self.std = cfg['std']
                    if 'input_size' in cfg and cfg['input_size']:
                        input_size = cfg['input_size']
                        if isinstance(input_size, (list, tuple)):
                            self.input_size = input_size[-1]
                        else:
                            self.input_size = input_size
            except (ImportError, Exception):
                pass
        
        resize_size = int(self.input_size * 1.143)
        self.transform = T.Compose([
            T.Resize(resize_size),
            T.CenterCrop(self.input_size),
            T.ToTensor(),
            T.Normalize(mean=self.mean, std=self.std)
        ])

    def process_image(self, img: Union[Image.Image, np.ndarray, List[Image.Image], List[np.ndarray]]) -> np.ndarray:
        """
        图像预处理：Resize→Crop→Normalize
        
        Args:
            img: PIL图像对象、numpy数组，或它们的列表（batch）
            
        Returns:
            处理后的numpy数组：
            - 单图：形状为(3, 224, 224)，dtype=float32
            - Batch：形状为(B, 3, 224, 224)，dtype=float32
        """
        is_batch = isinstance(img, list)
        if not is_batch:
            img = [img]
        
        processed_images = []
        for single_img in img:
            if isinstance(single_img, np.ndarray):
                single_img = Image.fromarray(single_img.astype('uint8'), 'RGB')
            elif isinstance(single_img, Image.Image):
                if single_img.mode != 'RGB':
                    if single_img.mode == 'RGBA':
                        rgb_img = Image.new('RGB', single_img.size, (255, 255, 255))
                        rgb_img.paste(single_img, mask=single_img.split()[3])
                        single_img = rgb_img
                    else:
                        single_img = single_img.convert('RGB')
            else:
                raise ValueError(f"不支持的图像类型: {type(single_img)}")
            
            tensor_result = self.transform(single_img)
            processed_images.append(tensor_result.numpy())
        
        result = np.stack(processed_images)
        
        # 如果不是batch，去掉batch维度
        if not is_batch:
            result = result.squeeze(0)
        
        return result

    def process_video_frames(self, frames: np.ndarray, frame_mask: np.ndarray = None) -> dict[str, np.ndarray]:
        """
        视频帧批处理：统一转为标准格式
        
        Args:
            frames: 视频帧numpy数组，形状为(T, H, W, C)或(T, C, H, W)，dtype=uint8或float32
            frame_mask: 帧mask，形状为(T,)，1=真实帧，0=补全帧。如果为None，则所有帧都是真实的
            
        Returns:
            包含"frames"和"frame_mask"的字典：
            - frames: numpy数组 (T, 3, 224, 224)，dtype=float32
            - frame_mask: numpy数组 (T,)，dtype=int64，1=真实帧，0=补全帧
        """
        num_frames = frames.shape[0]
        
        if frames.ndim == 4:
            if frames.shape[1] == 3 or frames.shape[1] == 1:
                frames = np.transpose(frames, (0, 2, 3, 1))
            elif frames.shape[-1] not in (1, 3, 4):
                raise ValueError(f"无法识别的视频帧格式: {frames.shape}")
        else:
            raise ValueError(f"视频帧维度错误，期望4维，实际{frames.ndim}维: {frames.shape}")

        processed_frames = []
        for t in range(num_frames):
            frame_array = frames[t]
            
            if frame_array.dtype != np.uint8:
                if frame_array.max() <= 1.0:
                    frame_array = (frame_array * 255).astype(np.uint8)
                else:
                    frame_array = np.clip(frame_array, 0, 255).astype(np.uint8)
            
            frame_array = np.clip(frame_array, 0, 255).astype(np.uint8)
            
            if frame_array.shape[2] != 3:
                if frame_array.shape[2] == 1:
                    frame_array = np.repeat(frame_array, 3, axis=2)
                elif frame_array.shape[2] == 4:
                    frame_array = frame_array[:, :, :3]
                else:
                    raise ValueError(f"不支持的通道数: {frame_array.shape[2]}")

            frame_pil = Image.fromarray(frame_array, 'RGB')
            processed_frame = self.transform(frame_pil)
            processed_frames.append(processed_frame.numpy())

        processed_frames_array = np.stack(processed_frames)

        if frame_mask is None:
            frame_mask = np.ones(num_frames, dtype=np.int64)
        else:
            if isinstance(frame_mask, torch.Tensor):
                frame_mask = frame_mask.numpy()
            frame_mask = frame_mask.astype(np.int64)

        return {
            "frames": processed_frames_array,
            "frame_mask": frame_mask
        }
