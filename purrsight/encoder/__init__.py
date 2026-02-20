"""
编码器模块：多模态特征编码器

注意：视频不再有独立的编码器，视频在预处理层被分解为图像和音频，
使用image_encoder和audio_encoder处理。
"""

from .image_encoder import _ImageEncoder
from .text_encoder import _TextEncoder
from .audio_encoder import _AudioEncoder

__all__ = [
    '_ImageEncoder',
    '_TextEncoder',
    '_AudioEncoder',
]

