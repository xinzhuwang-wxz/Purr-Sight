"""预处理模块公共接口：多模态输入自动识别与处理。

支持字典输入和直接输入，集成模态双重校验、动态关键帧选择、内存流音频处理。

包含：
- Preprocessor: 多模态预处理器主类

功能：
- 单样本预处理：process()
- Batch预处理：process_batch()
- 离线预处理加载：load_preprocessed()
- 视频分解：视频文件自动分解为 IMAGE 和 AUDIO 特征
- Fail-Safe 策略：所有失败情况都不会导致程序 crash
"""

import os
import subprocess
import torch
import numpy as np
from io import BytesIO
from PIL import Image
from typing import Union, Dict, Any, List
from pathlib import Path

from .image import _ImageProcessor
from .audio import _AudioProcessor
from .text import _TextProcessor
from purrsight.config import Modality, FeatureKey, ModalitySource
from purrsight.utils.logging import logger


class Preprocessor:
    """多模态预处理器：统一接口处理图像、视频、音频、文本输入。
    
    支持字典输入和直接输入，输出标记化特征字典（numpy 数组）。
    预处理返回 numpy 数组以节省内存，输入模型前使用 to_tensor() 转换为 tensor。
    
    Attributes:
        image_processor: 图像预处理器实例。
        audio_processor: 音频预处理器实例。
        text_processor: 文本预处理器实例。
        _instance: 单例实例（类属性）。
    """
    
    _instance = None

    def __init__(self):
        """初始化所有模态预处理器。
        
        Raises:
            RuntimeError: 当预处理器初始化失败时。
        """
        # System Check: 检查关键依赖和权重是否存在
        import shutil
        if not shutil.which("ffmpeg"):
            logger.warning("系统未检测到 ffmpeg！视频/音频处理将无法正常工作。请安装 ffmpeg。")
            
        try:
            self.image_processor = _ImageProcessor()
            self.audio_processor = _AudioProcessor()
            self.text_processor = _TextProcessor()
        except Exception as e:
            raise RuntimeError(f"初始化预处理器失败: {str(e)}")
    
    @classmethod
    def process(cls, input_data: Union[Dict[str, Any], str, Image.Image]) -> Dict[str, np.ndarray]:
        """类方法：便捷接口，无需实例化即可调用。
        
        Args:
            input_data: 输入数据，支持三种格式：
                - 字典：{"modality": data_path_or_content}
                - 字符串：文件路径（自动推断模态）
                - PIL.Image：图像对象
        
        Returns:
            模态标记化特征字典（numpy 数组），使用 FeatureKey 访问。
        
        Example:
            >>> features = Preprocessor.process({"text": "I love cat"})
            >>> features = Preprocessor.process("/path/to/image.jpg")
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance._process(input_data)
    
    @classmethod
    def load_preprocessed(cls, preprocessed_files: Dict[str, str], base_dir: Path) -> Dict[str, np.ndarray]:
        """类方法：加载预处理后的特征文件。
        
        Args:
            preprocessed_files: 预处理文件路径字典，键为模态名称，值为相对路径。
            base_dir: 预处理文件的基础目录。
            
        Returns:
            特征字典（numpy 数组），格式与 process() 相同。
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance._load_preprocessed(preprocessed_files, base_dir)
    
    def _load_preprocessed(self, preprocessed_files: Dict[str, str], base_dir: Path) -> Dict[str, np.ndarray]:
        """加载预处理后的特征文件。
        
        Args:
            preprocessed_files: 预处理文件路径字典。
            base_dir: 预处理文件的基础目录。
            
        Returns:
            特征字典（numpy 数组）。
            
        Raises:
            ValueError: 如果关键模态文件（文本、图像、音频）损坏。
        """
        features = {}
        
        # 加载文本特征
        if "text" in preprocessed_files:
            text_file = base_dir / preprocessed_files["text"]
            if text_file.exists():
                try:
                    # 🔧 验证2：立刻关闭mmap（验证mmap文件句柄泄漏问题）
                    # 不要做size判断，先验证稳定性
                    use_mmap = False  # 🔧 验证：临时关闭mmap
                    
                    # 🔧 性能优化：添加mmap=True以加速大文件加载，减少内存拷贝开销
                    text_data = torch.load(text_file, map_location="cpu", mmap=use_mmap, weights_only=False)
                    features[FeatureKey.TEXT] = text_data["input_ids"].numpy()
                    features[FeatureKey.TEXT_ATTENTION_MASK] = text_data["attention_mask"].numpy()
                    
                    # 🔧 修复：显式删除引用，帮助GC回收（防止内存泄漏）
                    del text_data
                except (EOFError, RuntimeError, ValueError) as e:
                    # 🔧 修复：离线预处理模式下，如果关键模态文件损坏，抛出异常跳过整个样本
                    # 文本是必需的，如果损坏应该跳过整个样本
                    raise ValueError(f"损坏的文本文件 {text_file}: {e}") from e
        
        # 加载图像特征（可能是视频帧）
        if "image" in preprocessed_files:
            image_file = base_dir / preprocessed_files["image"]
            if image_file.exists():
                try:
                    # 🔧 验证2：立刻关闭mmap（验证mmap文件句柄泄漏问题）
                    # 不要做size判断，先验证稳定性
                    use_mmap = False  # 🔧 验证：临时关闭mmap
                    
                    # 🔧 性能优化：添加mmap=True以加速大文件加载，减少内存拷贝开销
                    image_tensor = torch.load(image_file, map_location="cpu", mmap=use_mmap, weights_only=True)
                    features[FeatureKey.IMAGE] = image_tensor.numpy()
                    
                    # 🔧 修复：显式删除引用，帮助GC回收（防止内存泄漏）
                    del image_tensor
                except (EOFError, RuntimeError, ValueError) as e:
                    # 🔧 修复：离线预处理模式下，如果关键模态文件损坏，抛出异常跳过整个样本
                    # 而不是返回部分特征（避免零向量填充）
                    raise ValueError(f"损坏的图像文件 {image_file}: {e}") from e
        
        # 加载音频特征
        if "audio" in preprocessed_files:
            audio_file = base_dir / preprocessed_files["audio"]
            if audio_file.exists():
                try:
                    # 🔧 验证2：立刻关闭mmap（验证mmap文件句柄泄漏问题）
                    # 不要做size判断，先验证稳定性
                    use_mmap = False  # 🔧 验证：临时关闭mmap
                    
                    # 🔧 性能优化：添加mmap=True以加速大文件加载，减少内存拷贝开销
                    audio_tensor = torch.load(audio_file, map_location="cpu", mmap=use_mmap, weights_only=True)
                    features[FeatureKey.AUDIO] = audio_tensor.numpy()
                    
                    # 🔧 修复：显式删除引用，帮助GC回收（防止内存泄漏）
                    del audio_tensor
                except (EOFError, RuntimeError, ValueError) as e:
                    # 🔧 修复：离线预处理模式下，如果关键模态文件损坏，抛出异常跳过整个样本
                    # 如果索引文件中包含audio，说明原始样本有audio，文件损坏应该跳过整个样本
                    raise ValueError(f"损坏的音频文件 {audio_file}: {e}") from e
        
        # 加载视频元数据
        if "video_metadata" in preprocessed_files:
            metadata_file = base_dir / preprocessed_files["video_metadata"]
            if metadata_file.exists():
                try:
                    # 🔧 修复：对于小文件，不使用mmap可能更快，且避免文件句柄泄漏
                    file_size_mb = metadata_file.stat().st_size / (1024 * 1024)
                    use_mmap = file_size_mb > 10  # 只对大文件使用mmap
                    
                    # 🔧 性能优化：添加mmap=True以加速大文件加载，减少内存拷贝开销
                    video_metadata = torch.load(metadata_file, map_location="cpu", mmap=use_mmap, weights_only=False)
                    features["_video_metadata"] = video_metadata
                    
                    # 🔧 修复：显式删除引用，帮助GC回收（防止内存泄漏）
                    # 注意：video_metadata是字典，不需要del，因为已经赋值给features
                except (EOFError, RuntimeError, ValueError) as e:
                    logger.warning(f"跳过损坏的视频元数据文件 {metadata_file}: {e}")
        
        return features
    
    @classmethod
    def process_batch(cls, batch_inputs: List[Union[Dict[str, Any], str, Image.Image]], inference_mode: bool = False) -> Dict[str, np.ndarray]:
        """类方法：Batch 预处理接口，处理多个样本。
        
        Args:
            batch_inputs: 输入数据列表，每个元素支持三种格式：
                - 字典：{"modality": data_path_or_content}
                - 字符串：文件路径（自动推断模态）
                - PIL.Image：图像对象
                不同样本可能有不同的模态组合。
            inference_mode: 是否为推理模式。如果是，则尝试跳过不必要的计算（如单帧图像扩展为16帧）。
        
        Returns:
            模态标记化特征字典（numpy 数组），使用 FeatureKey 访问。
            所有特征都是 batch 格式，形状为 (B, ...)。
        
        Example:
            >>> batch_inputs = [
            ...     {"text": "Cat playing", "image": "/path/to/cat1.jpg"},
            ...     {"text": "Cat sleeping"},
            ...     {"image": "/path/to/cat2.jpg"},
            ... ]
            >>> features = Preprocessor.process_batch(batch_inputs, inference_mode=True)
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance._process_batch(batch_inputs, inference_mode=inference_mode)
    
    def _process(self, input_data: Union[Dict[str, Any], str, Image.Image]) -> Dict[str, np.ndarray]:
        """核心处理方法：输入解析 -> 模态识别 -> 特征提取 -> 返回标记化特征。
        
        Args:
            input_data: 输入数据，支持三种格式：
                - 字典：{"modality": data_path_or_content}
                - 字符串：文件路径（自动推断模态）
                - PIL.Image：图像对象
        
        Returns:
            模态标记化特征字典（numpy 数组），使用 FeatureKey 访问。
        """
        features = {}
        modality_sources = {}

        if isinstance(input_data, dict):
            for modality, data in input_data.items():
                modality_str = str(modality) if isinstance(modality, Modality) else modality
                
                if modality_str == Modality.TEXT:
                    text_result = self.text_processor.process_text(data)
                    features[FeatureKey.TEXT] = text_result["input_ids"]
                    features[FeatureKey.TEXT_ATTENTION_MASK] = text_result["attention_mask"]
                    modality_sources["text_source"] = ModalitySource.TEXT.value
                elif modality_str == Modality.IMAGE:
                    if isinstance(data, str):
                        img = Image.open(data)
                    else:
                        img = data
                    features[FeatureKey.IMAGE] = self.image_processor.process_image(img)
                    modality_sources["image_source"] = ModalitySource.IMAGE.value
                elif modality_str == Modality.VIDEO:
                    # 使用公共方法处理视频
                    video_features = self._process_video_to_features(data)
                    features.update(video_features)
                    # 从video_metadata获取source信息
                    if "_video_metadata" in video_features:
                        video_meta = video_features["_video_metadata"]
                        modality_sources["image_source"] = video_meta.get("image_source", ModalitySource.VIDEO.value)
                        modality_sources["audio_source"] = video_meta.get("audio_source")
                elif modality_str == Modality.AUDIO:
                    # 🔧 修复：离线预处理模式下，如果音频无法加载，抛出异常跳过样本
                    features[FeatureKey.AUDIO] = self.audio_processor.process_audio(data)
                    modality_sources["audio_source"] = ModalitySource.AUDIO.value

        elif isinstance(input_data, str):
            is_file_path = (
                os.path.exists(input_data) or 
                os.path.sep in input_data or 
                '.' in os.path.basename(input_data) and len(os.path.splitext(input_data)[1]) > 0
            )
            
            if not is_file_path:
                text_result = self.text_processor.process_text(input_data)
                features[FeatureKey.TEXT] = text_result["input_ids"]
                features[FeatureKey.TEXT_ATTENTION_MASK] = text_result["attention_mask"]
                modality_sources["text_source"] = ModalitySource.TEXT.value
            else:
                modality = self._infer_modality(input_data)

                if modality == Modality.TEXT:
                    with open(input_data, 'r', encoding='utf-8') as f:
                        text_content = f.read()
                    text_result = self.text_processor.process_text(text_content)
                    features[FeatureKey.TEXT] = text_result["input_ids"]
                    features[FeatureKey.TEXT_ATTENTION_MASK] = text_result["attention_mask"]
                    modality_sources["text_source"] = ModalitySource.TEXT.value
                elif modality == Modality.IMAGE:
                    img_path = Path(input_data)
                    
                    # 基本验证：文件存在和大小
                    if not img_path.exists():
                        raise FileNotFoundError(f"图像文件不存在: {input_data}")
                    
                    file_size = img_path.stat().st_size
                    if file_size == 0:
                        raise ValueError(f"图像文件为空: {input_data}")
                    
                    # 只使用PIL加载，失败直接抛异常
                    try:
                        img = Image.open(input_data)
                        img.load()  # 强制读取图像数据，验证文件完整性
                    except Exception as e:
                        raise ValueError(f"无法加载图像文件: {input_data}, 错误: {e}") from e
                    
                    features[FeatureKey.IMAGE] = self.image_processor.process_image(img)
                    modality_sources["image_source"] = ModalitySource.IMAGE.value
                elif modality == Modality.VIDEO:
                    # 使用公共方法处理视频
                    video_features = self._process_video_to_features(input_data)
                    features.update(video_features)
                    # 从video_metadata获取source信息
                    if "_video_metadata" in video_features:
                        video_meta = video_features["_video_metadata"]
                        modality_sources["image_source"] = video_meta.get("image_source", ModalitySource.VIDEO.value)
                        modality_sources["audio_source"] = video_meta.get("audio_source")
                elif modality == Modality.AUDIO:
                    # 🔧 修复：离线预处理模式下，如果音频无法加载，抛出异常跳过样本
                    features[FeatureKey.AUDIO] = self.audio_processor.process_audio(input_data)
                    modality_sources["audio_source"] = ModalitySource.AUDIO.value

        elif isinstance(input_data, Image.Image):
            features[FeatureKey.IMAGE] = self.image_processor.process_image(input_data)
            modality_sources["image_source"] = ModalitySource.IMAGE.value
        else:
            raise ValueError(f"不支持的输入类型: {type(input_data)}")

        # 添加modality_sources到features
        if modality_sources:
            features["_modality_sources"] = modality_sources

        return features
    
    def _process_batch(self, batch_inputs: List[Union[Dict[str, Any], str, Image.Image]], inference_mode: bool = False) -> Dict[str, np.ndarray]:
        """Batch 预处理核心方法：处理多个样本，返回 batch 格式的特征。
        
        Args:
            batch_inputs: 输入数据列表，每个样本可能有不同的模态组合。
            inference_mode: 是否为推理模式。
        
        Returns:
            Batch 格式的特征字典，所有特征形状为 (B, ...)。
        """
        if len(batch_inputs) == 0:
            raise ValueError("batch_inputs不能为空")
        
        batch_size = len(batch_inputs)
        
        # 收集每个样本的模态数据（使用列表索引对应样本）
        modality_data = {
            Modality.TEXT.value: [None] * batch_size,
            Modality.IMAGE.value: [None] * batch_size,
            Modality.VIDEO.value: [None] * batch_size,
            Modality.AUDIO.value: [None] * batch_size,
        }
        
        # 解析每个样本的输入
        for sample_idx, input_data in enumerate(batch_inputs):
            if isinstance(input_data, dict):
                for modality, data in input_data.items():
                    modality_str = str(modality) if isinstance(modality, Modality) else modality
                    if modality_str == Modality.TEXT.value:
                        modality_data[Modality.TEXT.value][sample_idx] = data
                    elif modality_str == Modality.IMAGE.value:
                        modality_data[Modality.IMAGE.value][sample_idx] = data
                    elif modality_str == Modality.VIDEO.value:
                        modality_data[Modality.VIDEO.value][sample_idx] = data
                    elif modality_str == Modality.AUDIO.value:
                        modality_data[Modality.AUDIO.value][sample_idx] = data
            
            elif isinstance(input_data, str):
                is_file_path = (
                    os.path.exists(input_data) or 
                    os.path.sep in input_data or 
                    '.' in os.path.basename(input_data) and len(os.path.splitext(input_data)[1]) > 0
                )
                
                if not is_file_path:
                    modality_data[Modality.TEXT.value][sample_idx] = input_data
                else:
                    modality = self._infer_modality(input_data)
                    if modality == Modality.TEXT:
                        with open(input_data, 'r', encoding='utf-8') as f:
                            text_content = f.read()
                        modality_data[Modality.TEXT.value][sample_idx] = text_content
                    elif modality == Modality.IMAGE:
                        modality_data[Modality.IMAGE.value][sample_idx] = input_data
                    elif modality == Modality.VIDEO:
                        modality_data[Modality.VIDEO.value][sample_idx] = input_data
                    elif modality == Modality.AUDIO:
                        modality_data[Modality.AUDIO.value][sample_idx] = input_data
            
            elif isinstance(input_data, Image.Image):
                modality_data[Modality.IMAGE.value][sample_idx] = input_data
            else:
                raise ValueError(f"不支持的输入类型: {type(input_data)}")
        
        # 处理每个模态的batch
        features = {}
        
        # 文本batch处理
        if any(text is not None for text in modality_data[Modality.TEXT.value]):
            # 对于缺失的样本，用空字符串填充（会被padding）
            text_list = []
            for text_data in modality_data[Modality.TEXT.value]:
                if text_data is not None:
                    text_list.append(text_data)
                else:
                    text_list.append("")  # 空字符串会被padding
            
            text_result = self.text_processor.process_text(text_list)
            features[FeatureKey.TEXT] = text_result["input_ids"]
            features[FeatureKey.TEXT_ATTENTION_MASK] = text_result["attention_mask"]
        
        # 图像batch处理（只处理独立图像，视频帧已在视频处理中完成）
        if any(img is not None for img in modality_data[Modality.IMAGE.value]):
            # 检查是否已有16帧格式（来自视频）
            has_video_frames = FeatureKey.IMAGE in features and features[FeatureKey.IMAGE].ndim == 5
            
            # 只处理独立图像（不是16帧格式）
            valid_images = []
            valid_indices = []
            for idx, img_data in enumerate(modality_data[Modality.IMAGE.value]):
                if img_data is not None:
                    # 检查是否是16帧格式（来自视频）
                    if isinstance(img_data, np.ndarray) and img_data.ndim == 3 and img_data.shape[0] == 16:
                        # 这是视频帧，跳过（已在视频处理中处理）
                        continue
                    if isinstance(img_data, str):
                        valid_images.append(Image.open(img_data))
                    elif isinstance(img_data, Image.Image):
                        valid_images.append(img_data)
                    else:
                        # 可能是其他格式，尝试处理
                        valid_images.append(img_data)
                    valid_indices.append(idx)
            
            if valid_images:
                # 处理独立图像
                processed_images = self.image_processor.process_image(valid_images)
                
                if has_video_frames and not inference_mode:
                    # 如果已有16帧格式，需要将独立图像转换为16帧格式（复制第一帧）
                    for valid_idx, processed_idx in enumerate(valid_indices):
                        # 将单帧复制到16帧的第一帧
                        features[FeatureKey.IMAGE][processed_idx, 0] = processed_images[valid_idx]
                        
                        # 🔧 补充：如果不扩展（即只填第0帧），后续的np.repeat逻辑需要知道这一点。
                        # 但实际上，这里的逻辑是把单帧放入5D数组的第0帧位置。
                        # 如果不扩展，其他15帧是0（或未初始化）。
                        # 如果inference_mode=True，且我们混合了batch（不推荐），
                        # 那么这里的 features[FeatureKey.IMAGE] 是5D的。
                        # 我们只能填入第0帧。
                        # 后续模型如果只取第0帧（针对image source），那就没事。
                        # 但通常模型会对16帧做平均。
                        # 所以混合batch在inference_mode=True下是不安全的，除非模型能识别 padding。
                        # 但这里我们只负责预处理。
                else:
                    # 创建单帧格式的batch数组（只包含有图像的样本）
                    batch_images = np.stack(processed_images)
                    features[FeatureKey.IMAGE] = batch_images
        
        # 视频batch处理：分解为图像和音频
        if any(video is not None for video in modality_data[Modality.VIDEO.value]):
            video_metadata = {}  # 记录视频样本信息

            for idx, video_data in enumerate(modality_data[Modality.VIDEO.value]):
                if video_data is not None:
                    # 检查是否有独立的image/audio文件（用于决策日志）
                    has_image_file = (
                        FeatureKey.IMAGE in features and 
                        features[FeatureKey.IMAGE] is not None and
                        (features[FeatureKey.IMAGE].ndim == 4 or 
                         (features[FeatureKey.IMAGE].ndim == 5 and idx < features[FeatureKey.IMAGE].shape[0]))
                    )
                    has_audio_file = (
                        modality_data[Modality.AUDIO.value][idx] is not None
                    )
                    
                    # 使用公共方法处理视频（返回16帧）
                    video_features = self._process_video_to_features(video_data)
                    video_frames = video_features[FeatureKey.IMAGE]  # (16, 3, 224, 224)
                    
                    # 获取视频metadata和source信息
                    video_meta = video_features["_video_metadata"]
                    image_source = video_meta.get("image_source", ModalitySource.VIDEO.value)
                    audio_source = video_meta.get("audio_source")
                    
                    # 添加决策日志
                    if image_source == ModalitySource.VIDEO.value and has_image_file:
                        logger.debug(f"样本{idx}: IMAGE被视频帧覆盖")
                    if audio_source == ModalitySource.VIDEO.value and has_audio_file:
                        logger.debug(f"样本{idx}: AUDIO被视频音频覆盖")
                    
                    # 标记为视频样本（包含source信息）
                    video_metadata[idx] = video_meta

                    # 优先使用视频提取的帧（保持时序对齐）
                    modality_data[Modality.IMAGE.value][idx] = video_frames
                    
                    # 更新features[FeatureKey.IMAGE]，将16帧写入对应位置
                    if FeatureKey.IMAGE in features:
                        # 检查现有IMAGE特征的shape
                        if features[FeatureKey.IMAGE].ndim == 4:
                            if not inference_mode:
                                # 现有的是单帧格式，需要转换为16帧格式
                                # 将现有单帧扩展为16帧（复制第一帧）
                                existing_frames = features[FeatureKey.IMAGE]
                                expanded_frames = np.repeat(existing_frames[:, np.newaxis, :, :], 16, axis=1)
                                expanded_frames[idx] = video_frames
                                features[FeatureKey.IMAGE] = expanded_frames
                            else:
                                logger.warning("Inference mode: Mixed batch of images and videos detected. "
                                             "Skipping expansion of images to avoid redundancy, but this may cause shape mismatch. "
                                             "Please avoid mixing images and videos in inference mode.")
                                # 尝试直接赋值（如果形状不匹配会抛出异常）
                                try:
                                    features[FeatureKey.IMAGE][idx] = video_frames
                                except ValueError:
                                    # 如果失败，回退到扩展模式（为了Fail-Safe）
                                    logger.warning("Shape mismatch, falling back to expansion.")
                                    existing_frames = features[FeatureKey.IMAGE]
                                    expanded_frames = np.repeat(existing_frames[:, np.newaxis, :, :], 16, axis=1)
                                    expanded_frames[idx] = video_frames
                                    features[FeatureKey.IMAGE] = expanded_frames
                        elif features[FeatureKey.IMAGE].ndim == 5:
                            # 已经是16帧格式，直接更新
                            features[FeatureKey.IMAGE][idx] = video_frames
                    else:
                        # 如果IMAGE特征还不存在，创建16帧格式（只包含当前视频样本）
                        features[FeatureKey.IMAGE] = video_frames[np.newaxis, :]

                    # 优先使用视频中的音频（保持时序对齐，但valid优先于source）
                    video_meta = video_features["_video_metadata"]
                    video_audio_valid = video_meta.get("audio_valid", False)
                    independent_audio_valid = has_audio_file
                    
                    if video_audio_valid and FeatureKey.AUDIO in video_features:
                        # 视频音频有效，使用视频音频
                        video_audio = video_features[FeatureKey.AUDIO]
                        modality_data[Modality.AUDIO.value][idx] = video_audio
                    elif independent_audio_valid:
                        # 视频音频无效，但独立音频有效，使用独立音频
                        # 独立音频已在音频处理阶段处理，这里不需要额外操作
                        pass
                    else:
                        # 两者都无效，audio mask将在后续设置为False
                        pass
                else:
                    video_metadata[idx] = {"has_video": False}

            # 将视频元数据添加到features中
            features["_video_metadata"] = video_metadata

        # 音频batch处理
        # 🔧 修复：离线预处理模式下，如果音频无法加载，抛出异常跳过样本
        # 注意：process_batch主要用于在线预处理，离线预处理使用process单样本处理
        # 但为了保持一致性，batch模式也应该在失败时抛出异常
        audio_batch = []
        for audio_data in modality_data[Modality.AUDIO.value]:
            if audio_data is not None:
                # 检查是否已经是处理过的numpy数组（来自视频分解）
                if isinstance(audio_data, np.ndarray):
                    # 已经是处理过的梅尔频谱，直接使用
                    audio_batch.append(audio_data)
                else:
                    # 是文件路径或BytesIO，需要处理
                    # 🔧 修复：如果处理失败，抛出异常（离线预处理会跳过整个样本）
                    audio_result = self.audio_processor.process_audio(audio_data)
                    audio_batch.append(audio_result)
            # 缺失的音频不添加到batch中
        
        if audio_batch:
            features[FeatureKey.AUDIO] = np.stack(audio_batch)

        # 统一创建mask：基于优先级规则处理后的最终modality_data状态
        # 优先级规则已处理完成：
        # - 图片+视频：优先使用视频提取的帧（16帧）
        # - 音频+视频：优先使用视频中的音频；如果视频没有音频，使用独立音频
        # - 无视频：正常使用独立数据
        # 注意：VIDEO mask不再需要，视频已分解为IMAGE和AUDIO
        modality_masks = {
            Modality.TEXT.value: np.array(
                [text is not None for text in modality_data[Modality.TEXT.value]], 
                dtype=np.bool_
            ),
            Modality.IMAGE.value: np.array(
                [img is not None for img in modality_data[Modality.IMAGE.value]], 
                dtype=np.bool_
            ),
            Modality.AUDIO.value: np.array(
                [audio is not None for audio in modality_data[Modality.AUDIO.value]], 
                dtype=np.bool_
            ),
        }

        # 将modality_masks添加到返回的features中
        features["_modality_masks"] = modality_masks
        
        return features

    def _process_video_to_features(self, video_path: str) -> Dict[str, Any]:
        """处理视频数据，提取帧和音频特征（Fail-Safe 版本）。
        
        返回 16 帧（不是多帧平均），在 embedding 空间做 mean pool 保留时序信息。
        
        Fail-Safe 策略：
        - 视频帧提取失败 -> 返回零向量帧，frames_valid=False
        - 视频音频提取失败 -> 返回空 BytesIO，audio_valid=False
        
        Args:
            video_path: 视频文件路径。
            
        Returns:
            特征字典，包含：
            - FeatureKey.IMAGE: 视频帧 (16, 3, 224, 224), dtype=float32
            - FeatureKey.AUDIO: 音频梅尔频谱 (64, 256), dtype=float32（如果存在）
            - "_video_metadata": 视频元数据字典，包含 image_source, audio_source, valid 标志等
        """
        # 分解视频（失败会抛异常）
        video_frames, frame_mask, audio_stream = self._process_video(video_path)
        video_result = self.image_processor.process_video_frames(video_frames, frame_mask)
        
        # 返回16帧，Mean Pool在embedding空间进行
        frames = video_result["frames"]  # (16, 3, 224, 224)
        valid_mask = video_result["frame_mask"]  # (16,)
        valid_frames = frames[valid_mask == 1]  # 只取有效帧
        frame_count = len(valid_frames)
        
        # 如果有效帧不足16帧，用最后一帧填充
        if frame_count < 16:
            pad_frames = np.repeat(valid_frames[-1:] if frame_count > 0 else frames[0:1], 
                                  16 - frame_count, axis=0)
            frames = np.concatenate([valid_frames, pad_frames], axis=0) if frame_count > 0 else pad_frames
        
        # 记录数据来源和有效性
        has_video_audio = audio_stream.getvalue() != b''
        
        # 构建返回字典
        result = {
            FeatureKey.IMAGE: frames,  # (16, 3, 224, 224)
            "_video_metadata": {
                "has_video": True,
                "has_video_image": True,
                "has_video_audio": has_video_audio,
                "frame_count": frame_count,
                "is_video_frames": True,
                "image_source": ModalitySource.VIDEO.value,
                "audio_source": ModalitySource.VIDEO.value if has_video_audio else None,
                "image_valid": True,
                "audio_valid": has_video_audio
            }
        }
        
        # 处理音频（如果存在）
        if has_video_audio:
            result[FeatureKey.AUDIO] = self.audio_processor.process_audio(audio_stream)
            result["_video_metadata"]["audio_valid"] = True
        
        return result

    def _infer_modality(self, input_path: str) -> Modality:
        """模态识别双重校验：扩展名初步推断 + 内容校验。
        
        Args:
            input_path: 文件路径。
        
        Returns:
            识别出的模态枚举: Modality.IMAGE, Modality.VIDEO, Modality.AUDIO, Modality.TEXT。
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"文件不存在: {input_path}")

        ext = os.path.splitext(input_path)[-1].lower()
        ext_map = {
            ('.jpg', '.jpeg', '.png', '.bmp', '.tiff'): Modality.IMAGE,
            ('.mp4', '.mov', '.avi', '.mkv', '.wmv', '.MOV'): Modality.VIDEO,
            ('.wav', '.mp3', '.flac', '.aac', '.ogg'): Modality.AUDIO,
            ('.txt', '.md', '.csv', '.json'): Modality.TEXT
        }

        inferred_modality = None
        for exts, modality_enum in ext_map.items():
            if ext in exts:
                inferred_modality = modality_enum
                break

        if inferred_modality is None:
            raise ValueError(f"无法根据扩展名识别模态: {input_path}")

        if inferred_modality == Modality.VIDEO:
            try:
                result = subprocess.run(
                    ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                     '-show_entries', 'stream=codec_type',
                     '-of', 'default=noprint_wrappers=1:nokey=1', input_path],
                    capture_output=True, text=True, check=True, timeout=10
                )
                if result.stdout.strip() == 'video':
                    return Modality.VIDEO
                else:
                    return Modality.AUDIO
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                return Modality.AUDIO

        elif inferred_modality == Modality.IMAGE:
            try:
                with open(input_path, 'rb') as f:
                    header = f.read(10)
                    if (header.startswith(b'\xff\xd8\xff') or
                        header.startswith(b'\x89PNG\r\n\x1a\n') or
                        header.startswith(b'BM') or
                        header.startswith(b'II*\x00') or
                        header.startswith(b'MM\x00*')):
                        return Modality.IMAGE
                    else:
                        raise ValueError(f"图像文件头校验失败: {input_path}")
            except Exception as e:
                raise ValueError(f"无法验证图像文件: {str(e)}")

        return inferred_modality

    def _process_video(self, video_path: str, target_frames: int = 16) -> tuple[np.ndarray, np.ndarray, BytesIO]:
        """视频处理：提取帧和音频。
        
        Args:
            video_path: 视频文件路径。
            target_frames: 目标帧数，默认 16。
            
        Returns:
            (video_frames, frame_mask, audio_stream) 元组：
            - video_frames: 视频帧 numpy 数组，形状为 (target_frames, H, W, C)，dtype=uint8。
            - frame_mask: 帧 mask，形状为 (target_frames,)，dtype=int64，1=真实帧，0=补全帧。
            - audio_stream: 音频内存流（如果视频有音频轨道）。
            
        Raises:
            ValueError: 当视频帧提取失败时。
            FileNotFoundError: 当 ffmpeg 未找到时。
        """
        # 提取帧（失败会抛异常）
        video_frames = self._extract_frames_from_video(video_path, target_frames=target_frames)
        frame_mask = np.ones(target_frames, dtype=np.int64)
        
        # 提取音频（如果视频没有音频轨道会抛异常，需要捕获）
        try:
            audio_stream = self._extract_audio_from_video(video_path)
        except ValueError:
            # 视频没有音频轨道是正常的，返回空BytesIO
            audio_stream = BytesIO()
        
        return video_frames, frame_mask, audio_stream

    def _extract_frames_from_video(self, video_path: str, target_frames: int = 16) -> np.ndarray:
        """从视频中提取指定数量的帧。
        
        使用 ffmpeg 直接在视频中均匀采样 target_frames 帧，不提取所有帧。
        
        Args:
            video_path: 视频文件路径。
            target_frames: 目标帧数，默认 16。
        
        Returns:
            video_frames: 视频帧 numpy 数组，形状为 (target_frames, H, W, C)，dtype=uint8。
            
        Raises:
            ValueError: 当视频帧提取失败时。
            FileNotFoundError: 当 ffmpeg 未找到时。
        """
        # 尝试用ffprobe获取视频信息
        width = None
        height = None
        total_frames_in_video = None
        
        try:
            probe_result = subprocess.run(
                ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                 '-show_entries', 'stream=width,height,duration,nb_frames',
                 '-of', 'json', video_path],
                capture_output=True, text=True, check=True, timeout=10
            )
            import json
            video_info = json.loads(probe_result.stdout)
            stream = video_info['streams'][0]
            width = stream['width']
            height = stream['height']
            total_frames_in_video = stream.get('nb_frames')
            if total_frames_in_video:
                total_frames_in_video = int(total_frames_in_video)
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError, KeyError, IndexError, json.JSONDecodeError):
            # ffprobe失败不影响，继续用ffmpeg提取帧
            pass

        # 使用ffmpeg提取帧
        try:
            if total_frames_in_video and total_frames_in_video > 0:
                step = max(1, total_frames_in_video // target_frames)
                vf_filter = f"select='not(mod(n\\,{step}))',scale=256:-1"
            else:
                vf_filter = f"fps={target_frames/10.0:.1f},scale=256:-1"
            
            ffmpeg_cmd = [
                'ffmpeg', '-i', video_path,
                '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                '-vf', vf_filter,
                'pipe:1'
            ]
            result = subprocess.run(ffmpeg_cmd, capture_output=True, check=True, timeout=120)
        except subprocess.TimeoutExpired as e:
            raise ValueError(f"视频帧提取超时: {video_path}") from e
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr if isinstance(e.stderr, str) else (e.stderr.decode('utf-8', errors='ignore') if e.stderr else "未知错误")
            error_preview = error_msg.split('\n')[0] if error_msg else "未知错误"
            raise ValueError(f"ffmpeg提取帧失败: {video_path}, 错误: {error_preview}") from e
        except FileNotFoundError:
            raise FileNotFoundError(f"ffmpeg未找到，无法提取视频帧: {video_path}")

        # 解析帧数据
        try:
            frame_data = np.frombuffer(result.stdout, dtype=np.uint8)
            scale_width = 256
            
            if width is not None and height is not None and width > 0:
                scale_height = int(height * scale_width / width)
            else:
                scale_height = 256
            
            # 如果无法获取视频尺寸，从实际数据推断帧尺寸
            if width is None or height is None:
                possible_heights = [256, 240, 224, 192, 180, 144]
                extracted_frames = 0
                actual_height = None
                
                for test_height in possible_heights:
                    test_frame_size = scale_width * test_height * 3
                    test_frames = len(frame_data) // test_frame_size
                    if test_frames > 0 and len(frame_data) % test_frame_size == 0:
                        extracted_frames = test_frames
                        actual_height = test_height
                        break
                
                if extracted_frames == 0:
                    actual_height = 256
                    frame_size = scale_width * actual_height * 3
                    extracted_frames = len(frame_data) // frame_size
                    if extracted_frames == 0:
                        raise ValueError(f"无法推断视频帧尺寸: {video_path}")
                scale_height = actual_height
            else:
                frame_size = scale_width * scale_height * 3
                extracted_frames = len(frame_data) // frame_size
            
            if extracted_frames == 0:
                raise ValueError(f"未提取到任何帧: {video_path}")
            
            # 重塑为帧数组
            frame_size = scale_width * scale_height * 3
            frames = frame_data[:extracted_frames * frame_size].reshape((extracted_frames, scale_height, scale_width, 3)).copy()
            
            # 如果提取的帧数不等于目标帧数，进行采样或填充
            if extracted_frames > target_frames:
                indices = np.linspace(0, extracted_frames - 1, target_frames, dtype=np.int32)
                frames = frames[indices]
            elif extracted_frames < target_frames:
                pad_frames = np.repeat(frames[-1:], target_frames - extracted_frames, axis=0)
                frames = np.concatenate([frames, pad_frames], axis=0)
            
            return frames[:target_frames]
        except (ValueError, IndexError) as e:
            raise ValueError(f"视频帧数据解析失败: {video_path}, 错误: {e}") from e

    def _extract_audio_from_video(self, video_path: str) -> BytesIO:
        """从视频中提取音频并保存到内存流。
        
        Args:
            video_path: 视频文件路径。
        
        Returns:
            audio_stream: 包含音频数据的 BytesIO 内存流。
            
        Raises:
            ValueError: 当视频没有音频轨道或音频提取失败时。
            FileNotFoundError: 当 ffmpeg 未找到时。
        """
        # 检查是否有音频轨道
        probe_result = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'a:0',
             '-show_entries', 'stream=codec_type',
             '-of', 'default=noprint_wrappers=1:nokey=1', video_path],
            capture_output=True, text=True, timeout=10
        )
        
        if probe_result.returncode != 0 or probe_result.stdout.strip() != 'audio':
            raise ValueError(f"视频没有音频轨道: {video_path}")
        
        try:
            result = subprocess.run(
                ['ffmpeg', '-i', video_path,
                 '-vn',
                 '-acodec', 'pcm_s16le',
                 '-ar', '16000',
                 '-ac', '1',
                 '-f', 'wav',
                 'pipe:1'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                timeout=60
            )

            audio_stream = BytesIO(result.stdout)
            audio_stream.seek(0)
            return audio_stream

        except subprocess.CalledProcessError as e:
            error_msg = e.stderr if isinstance(e.stderr, str) else (e.stderr.decode('utf-8', errors='ignore') if e.stderr else "未知错误")
            error_preview = error_msg.split('\n')[0] if error_msg else "未知错误"
            raise ValueError(f"视频音频提取失败: {video_path}, 错误: {error_preview}") from e
        except subprocess.TimeoutExpired:
            raise ValueError(f"视频音频提取超时: {video_path}")
        except FileNotFoundError:
            raise FileNotFoundError(f"ffmpeg未找到，无法提取视频音频: {video_path}")


__all__ = ['Preprocessor']
