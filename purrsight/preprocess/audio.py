"""
音频预处理模块：支持文件和内存流输入，返回梅尔频谱特征

使用PANNs官方的梅尔频谱提取器（从权重文件加载logmel_extractor.melW）。

包含：
- _AudioProcessor: 音频预处理器类（Fail-Safe版本）

参考仓库：
- pytorch/audio (torchaudio.pipelines.YAMNET)
"""

import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
from io import BytesIO
from typing import Union, Optional
import tempfile
import os
import subprocess
from pathlib import Path
from purrsight.config import ROOT_DIR


class _AudioProcessor:
    """
    音频预处理器：支持文件路径和内存流输入（Fail-Safe版本）
    
    使用PANNs官方的梅尔频谱提取器（从权重文件加载），输出标准梅尔频谱特征(64, 256)。
    
    Attributes:
        sample_rate: 目标采样率
        n_mels: 梅尔滤波器组数量
        n_fft: FFT窗口大小
        hop_length: 帧移
        stft: STFT变换器
        amplitude_to_db: 幅度到dB变换器
        mel_transform: 梅尔频谱变换器
        use_official_mel: 是否使用官方梅尔滤波器组
    """

    def __init__(
        self, 
        sample_rate: int = 16000, 
        n_mels: int = 64, 
        n_fft: int = 512, 
        hop_length: int = 160,
        weight_path: Optional[Union[str, Path]] = None
    ):
        """
        初始化音频预处理参数
        
        Args:
            sample_rate: 目标采样率，默认16kHz（与PANNs官方一致）
            n_mels: 梅尔滤波器组数量，默认64（与PANNs官方一致）
            n_fft: FFT窗口大小，默认512（与PANNs官方window_size一致）
            hop_length: 帧移，默认160（与PANNs官方hop_size一致）
            weight_path: PANNs权重文件路径，用于提取官方梅尔滤波器组
        """
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        self._load_panns_mel_extractor(weight_path)
        
        self.stft = T.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            power=1.0,
            normalized=False
        )
        
        self.amplitude_to_db = T.AmplitudeToDB()  # 对数变换
    
    def _load_panns_mel_extractor(self, weight_path: Optional[Union[str, Path]]):
        """
        从PANNs权重文件中加载官方的梅尔频谱提取器
        
        Args:
            weight_path: 权重文件路径，如果为None则使用默认路径
        """
        if weight_path is None:
            weight_path = Path(ROOT_DIR, "models/panns/cnn14_light_16k.pth")
        else:
            weight_path = Path(weight_path)
        
        if not weight_path.exists():
            self.use_official_mel = False
            self.mel_transform = T.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
                f_min=50,
                f_max=8000
            )
            return
        
        try:
            state_dict = torch.load(weight_path, map_location="cpu", weights_only=False)
            
            if 'model' in state_dict:
                model_state = state_dict['model']
            else:
                model_state = state_dict
            
            if 'logmel_extractor.melW' in model_state:
                self.melW = model_state['logmel_extractor.melW']
                self.use_official_mel = True
            else:
                self.use_official_mel = False
                self.mel_transform = T.MelSpectrogram(
                    sample_rate=self.sample_rate,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    n_mels=self.n_mels,
                    f_min=50,
                    f_max=8000
                )
        except Exception:
            self.use_official_mel = False
            self.mel_transform = T.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
                f_min=50,
                f_max=8000
            )
    
    def _apply_official_mel_filterbank(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        使用PANNs官方的梅尔滤波器组提取梅尔频谱
        
        Args:
            spectrogram: 幅度频谱图，形状 (B, freq_bins, time)
        
        Returns:
            梅尔频谱，形状 (B, n_mels, time)
        """
        mel_spec = torch.matmul(spectrogram.transpose(1, 2), self.melW).transpose(1, 2)
        return mel_spec

    def process_audio(self, audio_data: Union[str, BytesIO]) -> np.ndarray:
        """
        音频预处理：重采样→STFT→梅尔滤波器组→对数变换→静音裁剪
        
        🔧 修复：离线预处理模式下，如果音频无法加载，抛出异常而不是返回零向量。
        这样可以确保预处理的数据都是可用的，无法加载的样本会被跳过。
        
        Args:
            audio_data: 文件路径或内存流(BytesIO)
        
        Returns:
            mel_spec: 梅尔频谱特征，形状为(64, 256)，dtype=float32
        
        Raises:
            ValueError: 当音频文件无法加载或处理失败时
        """
        # 加载音频（如果失败会抛出异常）
        waveform, sample_rate = self._load_audio(audio_data)

        # 处理音频
        if sample_rate != self.sample_rate:
            resampler = T.Resample(sample_rate, self.sample_rate)
            waveform = resampler(waveform)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        waveform = self._trim_silence(waveform)

        if self.use_official_mel:
            spectrogram = self.stft(waveform)
            mel_spec = self._apply_official_mel_filterbank(spectrogram)
        else:
            mel_spec = self.mel_transform(waveform)

        mel_spec = self.amplitude_to_db(mel_spec)

        target_time = 256
        current_time = mel_spec.shape[2]

        if current_time > target_time:
            start = (current_time - target_time) // 2
            mel_spec = mel_spec[:, :, start:start + target_time]
        else:
            pad_width = target_time - current_time
            mel_spec = torch.nn.functional.pad(mel_spec, (0, pad_width), mode='constant', value=0)

        mel_spec = mel_spec.squeeze(0)
        return mel_spec.numpy()

    def _load_audio(self, audio_data: Union[str, BytesIO]) -> tuple[torch.Tensor, int]:
        """
        加载音频数据：支持文件路径和内存流
        
        自动处理不支持的格式（如m4a），使用ffmpeg转换为wav。
        
        Args:
            audio_data: 文件路径或BytesIO对象
        
        Returns:
            (waveform, sample_rate)元组：
            - waveform: 音频波形tensor，形状为(channels, samples)
            - sample_rate: 采样率
        
        Raises:
            ValueError: 当音频文件无法加载或转换时
        """
        if isinstance(audio_data, str):
            audio_path = Path(audio_data)
            
            try:
                waveform, sample_rate = torchaudio.load(audio_path)
                return waveform, sample_rate
            except Exception as e:
                ext = audio_path.suffix.lower()
                unsupported_formats = {'.m4a', '.aac', '.mp3', '.flac', '.ogg', '.opus'}
                error_str = str(e).lower()
                
                # 🔧 修复：如果 torchaudio 加载失败，尝试使用 ffmpeg 转换
                # 包括 .wav 文件（可能格式不标准、损坏或编码问题）
                # 检查错误信息中是否包含格式相关的关键词
                should_use_ffmpeg = (
                    ext in unsupported_formats or 
                    'format' in error_str or 
                    'not recognised' in error_str or
                    'not recognized' in error_str or
                    'unsupported' in error_str or
                    'codec' in error_str or
                    'decoder' in error_str
                )
                
                if should_use_ffmpeg:
                    temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                    temp_wav_path = temp_wav.name
                    temp_wav.close()
                    
                    try:
                        result = subprocess.run(
                            ['ffmpeg', '-i', str(audio_path), '-y', 
                             '-ar', '16000',
                             '-ac', '1',
                             '-f', 'wav', 
                             '-loglevel', 'error',
                             temp_wav_path],
                            capture_output=True,
                            check=True,
                            timeout=30
                        )
                        
                        if not os.path.exists(temp_wav_path) or os.path.getsize(temp_wav_path) == 0:
                            raise ValueError(f"ffmpeg转换后的文件为空: {audio_path}")
                        
                        waveform, sample_rate = torchaudio.load(temp_wav_path)
                        return waveform, sample_rate
                    except subprocess.CalledProcessError as ffmpeg_error:
                        error_msg = ffmpeg_error.stderr.decode('utf-8', errors='ignore') if ffmpeg_error.stderr else "未知错误"
                        error_preview = error_msg.split('\n')[0] if error_msg else "未知错误"
                        raise ValueError(f"无法加载音频文件 {audio_path}: ffmpeg转换失败: {error_preview}") from e
                    except subprocess.TimeoutExpired:
                        raise ValueError(f"音频转换超时: {audio_path}") from e
                    except FileNotFoundError:
                        raise FileNotFoundError(f"ffmpeg未找到，无法转换音频格式: {audio_path}") from e
                    finally:
                        if os.path.exists(temp_wav_path):
                            os.unlink(temp_wav_path)
                else:
                    raise ValueError(f"无法加载音频文件 {audio_path}: {e}") from e

        elif isinstance(audio_data, BytesIO):
            audio_data.seek(0)
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_file.write(audio_data.read())
                temp_path = temp_file.name

            try:
                waveform, sample_rate = torchaudio.load(temp_path)
                return waveform, sample_rate
            except Exception as e:
                raise ValueError(f"无法加载音频内存流: {str(e)}")
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
        else:
            raise ValueError(f"不支持的音频输入类型: {type(audio_data)}")

    def _trim_silence(self, waveform: torch.Tensor, threshold_db: float = -40) -> torch.Tensor:
        """
        裁剪音频首尾的静音部分
        
        Args:
            waveform: 音频波形tensor，形状为(channels, samples)
            threshold_db: 静音阈值（dB），默认-40dB
        
        Returns:
            裁剪后的音频波形tensor
        """
        """
        静音裁剪：移除前后静音段
        
        Args:
            waveform: 音频波形，形状为(1, time)
            threshold_db: 静音阈值(dB)，默认-40
        
        Returns:
            裁剪后的波形
        """
        rms = torch.sqrt(torch.mean(waveform ** 2, dim=0))
        db = 20 * torch.log10(rms + 1e-10)
        mask = db > threshold_db
        indices = torch.where(mask)[0]

        if len(indices) == 0:
            return waveform

        start_idx = indices[0]
        end_idx = indices[-1] + 1
        return waveform[:, start_idx:end_idx]