"""
Multi-Modal Dataset for Phase 2 Training

Handles multimodal data loading, preprocessing, and instruction tuning.
Includes both MultiModalDataset (general-purpose) and InstructionDataset (instruction tuning).
"""

import json
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, Any, List, Optional, Callable, Union
from pathlib import Path
from transformers import PreTrainedTokenizer

from purrsight.preprocess import Preprocessor
from purrsight.config import FeatureKey, Modality
from purrsight.utils.logging import logger
from purrsight.LLM.prompts import SYSTEM_PROMPT_V3


class MultiModalDataset(Dataset):
    """
    Multi-modal dataset for Phase 2 training.
    
    Supports loading image, audio, and text data with preprocessing.
    Handles missing modalities gracefully using zero tensors.
    Implements proper error handling for corrupted files.
    
    This dataset is designed for the Phase 2 training pipeline where:
    - Image, audio, and text inputs are processed through their respective encoders
    - Aligned features are projected into LLM input space
    - The LLM processes multi-modal inputs to generate text outputs
    
    Attributes:
        data_dir: Root directory containing data files
        split: Dataset split ('train', 'val', 'test')
        tokenizer: Text tokenizer for processing text inputs
        max_length: Maximum sequence length for text
        preprocessor: Multi-modal preprocessor instance
        samples: List of sample metadata loaded from index file
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str,
        tokenizer: PreTrainedTokenizer,
        image_transform: Optional[Callable] = None,
        audio_transform: Optional[Callable] = None,
        max_length: int = 512
    ):
        """
        Initialize multi-modal dataset.
        
        Args:
            data_dir: Root directory containing data files
            split: Dataset split ('train', 'val', 'test')
            tokenizer: Text tokenizer instance
            image_transform: Optional image preprocessing transform (deprecated, using Preprocessor)
            audio_transform: Optional audio preprocessing transform (deprecated, using Preprocessor)
            max_length: Maximum sequence length for text
            
        Raises:
            FileNotFoundError: If data directory doesn't exist
            ValueError: If split is invalid or no data files found
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Validate data directory
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
        
        if not self.data_dir.is_dir():
            raise ValueError(f"Data path is not a directory: {data_dir}")
        
        # Validate split
        valid_splits = {'train', 'val', 'test', 'validation'}
        if split not in valid_splits:
            raise ValueError(f"Invalid split '{split}'. Must be one of: {valid_splits}")
        
        # Initialize preprocessor for image/audio/text processing
        try:
            self.preprocessor = Preprocessor()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize preprocessor: {e}")
        
        # Load dataset samples
        self.samples = self._load_samples()
        
        if len(self.samples) == 0:
            logger.warning(f"No samples found for split '{split}' in directory '{data_dir}'")
    
    def _load_samples(self) -> List[Dict[str, Any]]:
        """
        Load sample metadata from index files.
        
        Looks for index files in the following order:
        1. {split}.jsonl (e.g., train.jsonl)
        2. index.jsonl (general index file)
        3. Direct file enumeration in data directory
        
        Returns:
            List of sample dictionaries containing file paths and metadata
            
        Raises:
            ValueError: If no valid samples found
        """
        samples = []
        
        # Try split-specific index file first
        split_index_file = self.data_dir / f"{self.split}.jsonl"
        if split_index_file.exists():
            try:
                samples = self._load_from_jsonl(split_index_file)
                logger.info(f"Loaded {len(samples)} samples from {split_index_file}")
                return samples
            except Exception as e:
                logger.warning(f"Failed to load from {split_index_file}: {e}")
        
        # Try general index file
        general_index_file = self.data_dir / "index.jsonl"
        if general_index_file.exists():
            try:
                all_samples = self._load_from_jsonl(general_index_file)
                # Filter by split if split information is available
                samples = [s for s in all_samples if s.get('split', self.split) == self.split]
                if not samples:
                    # If no split info, use all samples (assume single split)
                    samples = all_samples
                logger.info(f"Loaded {len(samples)} samples from {general_index_file}")
                return samples
            except Exception as e:
                logger.warning(f"Failed to load from {general_index_file}: {e}")
        
        # Fallback: enumerate files directly
        try:
            samples = self._enumerate_files()
            logger.info(f"Enumerated {len(samples)} samples from directory")
            return samples
        except Exception as e:
            logger.error(f"Failed to enumerate files: {e}")
            return []
    
    def _load_from_jsonl(self, index_file: Path) -> List[Dict[str, Any]]:
        """
        Load samples from JSONL index file.
        
        Args:
            index_file: Path to JSONL index file
            
        Returns:
            List of sample dictionaries
            
        Raises:
            ValueError: If file is corrupted or invalid format
        """
        samples = []
        
        try:
            with open(index_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        sample = json.loads(line)
                        # Validate required fields
                        if not isinstance(sample, dict):
                            logger.warning(f"Line {line_num}: Sample is not a dictionary, skipping")
                            continue
                        
                        # Convert relative paths to absolute paths
                        for key in ['image', 'audio', 'video', 'text']:
                            if key in sample and isinstance(sample[key], str):
                                if not Path(sample[key]).is_absolute():
                                    sample[key] = str(self.data_dir / sample[key])
                        
                        samples.append(sample)
                        
                    except json.JSONDecodeError as e:
                        logger.warning(f"Line {line_num}: Invalid JSON, skipping: {e}")
                        continue
                    except Exception as e:
                        logger.warning(f"Line {line_num}: Error processing sample: {e}")
                        continue
                        
        except Exception as e:
            raise ValueError(f"Failed to read index file {index_file}: {e}")
        
        return samples
    
    def _enumerate_files(self) -> List[Dict[str, Any]]:
        """
        Enumerate files directly from data directory.
        
        Creates samples by finding matching image/audio/text files.
        This is a fallback when no index file is available.
        
        Returns:
            List of sample dictionaries
        """
        samples = []
        
        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        audio_extensions = {'.wav', '.mp3', '.flac', '.aac', '.ogg', '.m4a'}
        video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.wmv'}
        text_extensions = {'.txt', '.md'}
        
        # Group files by base name (without extension)
        file_groups = {}
        
        for file_path in self.data_dir.rglob('*'):
            if file_path.is_file():
                base_name = file_path.stem
                ext = file_path.suffix.lower()
                
                if base_name not in file_groups:
                    file_groups[base_name] = {}
                
                if ext in image_extensions:
                    file_groups[base_name]['image'] = str(file_path)
                elif ext in audio_extensions:
                    file_groups[base_name]['audio'] = str(file_path)
                elif ext in video_extensions:
                    file_groups[base_name]['video'] = str(file_path)
                elif ext in text_extensions:
                    file_groups[base_name]['text'] = str(file_path)
        
        # Create samples from file groups
        for base_name, files in file_groups.items():
            if files:  # Only create sample if at least one modality exists
                sample = {
                    'sample_id': base_name,
                    'split': self.split,
                    **files
                }
                samples.append(sample)
        
        return samples
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get single sample with preprocessing.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing:
                - image: Preprocessed image tensor (3, 224, 224) or (16, 3, 224, 224) for video
                - audio: Preprocessed audio tensor (64, 256)
                - text_tokens: Tokenized text (max_length,)
                - attention_mask: Attention mask (max_length,)
                - labels: Target labels (same as text_tokens for language modeling)
                - sample_id: Sample identifier for debugging
                
        Raises:
            IndexError: If idx is out of range
            ValueError: If sample processing fails critically
        """
        if idx < 0 or idx >= len(self.samples):
            raise IndexError(f"Sample index {idx} out of range [0, {len(self.samples)})")
        
        sample = self.samples[idx]
        sample_id = sample.get('sample_id', f'sample_{idx}')
        
        result = {
            'sample_id': sample_id
        }
        
        # Process multi-modal inputs
        mm_inputs = {}
        for modality in ['image', 'audio', 'video', 'text']:
            if modality in sample:
                mm_inputs[modality] = sample[modality]
        
        # Use preprocessor to extract features
        try:
            features = self.preprocessor.process(mm_inputs)
        except Exception as e:
            # Handle corrupted files gracefully
            logger.warning(f"Failed to process sample {sample_id} (idx={idx}): {e}")
            features = {}
        
        # Convert features to tensors and handle missing modalities
        
        # Image processing
        if FeatureKey.IMAGE in features:
            try:
                image_array = features[FeatureKey.IMAGE]
                result['image'] = torch.from_numpy(image_array).float()
            except Exception as e:
                logger.warning(f"Failed to convert image for sample {sample_id}: {e}")
                result['image'] = self._get_zero_image_tensor()
        else:
            result['image'] = self._get_zero_image_tensor()
        
        # Audio processing
        if FeatureKey.AUDIO in features:
            try:
                audio_array = features[FeatureKey.AUDIO]
                result['audio'] = torch.from_numpy(audio_array).float()
            except Exception as e:
                logger.warning(f"Failed to convert audio for sample {sample_id}: {e}")
                result['audio'] = self._get_zero_audio_tensor()
        else:
            result['audio'] = self._get_zero_audio_tensor()
        
        # Text processing
        text_content = ""
        if FeatureKey.TEXT in features:
            try:
                # Extract text tokens from preprocessor
                text_tokens = features[FeatureKey.TEXT]
                attention_mask = features[FeatureKey.TEXT_ATTENTION_MASK]
                
                result['text_tokens'] = torch.from_numpy(text_tokens).long()
                result['attention_mask'] = torch.from_numpy(attention_mask).long()
                
                # For language modeling, labels are the same as input tokens
                result['labels'] = result['text_tokens'].clone()
                
            except Exception as e:
                logger.warning(f"Failed to process text tokens for sample {sample_id}: {e}")
                # Fallback to empty text
                result.update(self._get_zero_text_tensors())
        elif 'text' in sample:
            # Text file path provided, read and tokenize
            try:
                text_path = Path(sample['text'])
                if text_path.exists():
                    with open(text_path, 'r', encoding='utf-8') as f:
                        text_content = f.read().strip()
                else:
                    logger.warning(f"Text file not found for sample {sample_id}: {text_path}")
                    text_content = ""
            except Exception as e:
                logger.warning(f"Failed to read text file for sample {sample_id}: {e}")
                text_content = ""
            
            # Tokenize text content
            try:
                if text_content:
                    encoded = self.tokenizer(
                        text_content,
                        max_length=self.max_length,
                        padding='max_length',
                        truncation=True,
                        return_tensors='pt'
                    )
                    result['text_tokens'] = encoded['input_ids'].squeeze(0)
                    result['attention_mask'] = encoded['attention_mask'].squeeze(0)
                    result['labels'] = result['text_tokens'].clone()
                else:
                    result.update(self._get_zero_text_tensors())
            except Exception as e:
                logger.warning(f"Failed to tokenize text for sample {sample_id}: {e}")
                result.update(self._get_zero_text_tensors())
        else:
            # No text data
            result.update(self._get_zero_text_tensors())
        
        return result
    
    def _get_zero_image_tensor(self) -> torch.Tensor:
        """
        Get zero tensor for missing image data.
        
        Returns:
            Zero tensor with shape (3, 224, 224) for single image
        """
        return torch.zeros(3, 224, 224, dtype=torch.float32)
    
    def _get_zero_audio_tensor(self) -> torch.Tensor:
        """
        Get zero tensor for missing audio data.
        
        Returns:
            Zero tensor with shape (64, 256) for mel spectrogram
        """
        return torch.zeros(64, 256, dtype=torch.float32)
    
    def _get_zero_text_tensors(self) -> Dict[str, torch.Tensor]:
        """
        Get zero tensors for missing text data.
        
        Returns:
            Dictionary with text_tokens, attention_mask, and labels
        """
        return {
            'text_tokens': torch.zeros(self.max_length, dtype=torch.long),
            'attention_mask': torch.zeros(self.max_length, dtype=torch.long),
            'labels': torch.full((self.max_length,), -100, dtype=torch.long)  # -100 is ignore index
        }


class InstructionDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer: Any,
        max_length: int = 2048,
        ignore_index: int = -100
    ):
        """
        Args:
            data_path: Path to JSONL file
            tokenizer: HF Tokenizer
            max_length: Max sequence length
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.ignore_index = ignore_index
        
        # Load Data
        self.data_indices = []
        if self.data_path.exists():
            # Memory Optimization: Lazy Loading using file offsets.
            # Instead of loading all JSON objects into memory, we store file offsets.
            # This reduces RAM usage from GBs to MBs for large datasets.
            with open(self.data_path, 'r', encoding='utf-8') as f:
                while True:
                    offset = f.tell()
                    line = f.readline()
                    if not line:
                        break
                    if line.strip():
                        self.data_indices.append(offset)
        else:
            logger.warning(f"Data file not found: {data_path}")
            
        # Initialize Preprocessor (for Image/Audio/Video)
        self.preprocessor = Preprocessor()
        
    def __len__(self):
        return len(self.data_indices)
    
    def __getitem__(self, idx):
        # Memory Optimization: Lazy Load.
        offset = self.data_indices[idx]
        with open(self.data_path, 'r', encoding='utf-8') as f:
            f.seek(offset)
            line = f.readline()
            try:
                item = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                # File Modification Safety: Handle cases where file changed during training.
                logger.warning(f"Failed to decode JSON at offset {offset}: {e}. Skipping sample.")
                item = {}
            except Exception as e:
                logger.error(f"Unexpected error reading data at offset {offset}: {e}")
                item = {}

        # 1. Process Multimodal Inputs
        mm_input = {}
        data_dir = Path(self.data_path).parent  # Get directory containing the JSONL file
        
        for k in ['image', 'audio', 'video']:
            if k in item:
                # Convert relative paths to absolute paths
                file_path = item[k]
                if not Path(file_path).is_absolute():
                    # Resolve relative path from data directory
                    file_path = str((data_dir / file_path).resolve())
                mm_input[k] = file_path
        
        # Use Preprocessor to get Image/Audio features (numpy)
        try:
            # Debug: Print paths being processed
            if mm_input:
                logger.info(f"Processing multimodal inputs for sample {idx}: {mm_input}")
            
            # Preprocessor.process handles file loading and processing
            features = self.preprocessor.process(mm_input)
        except Exception as e:
            logger.warning(f"Failed to process multimodal inputs for sample {idx}: {e}")
            features = {}
            
        # 2. Process Text (Instruction + Response)
        instruction = item.get('instruction', '')
        response = item.get('response', '')
        
        # Format Prompt
        # Inject System Prompt V3.0
        # Format: <System Prompt>\n\nHuman: <Instruction>\nAssistant: 
        prompt = f"{SYSTEM_PROMPT_V3}\n\nHuman: {instruction}\nAssistant: "
        
        # Tokenize Prompt and Response separately to create labels
        # Note: We need to handle BOS/EOS manually or rely on tokenizer
        
        # A robust way:
        # 1. Tokenize Prompt (add BOS)
        prompt_tokens = self.tokenizer(
            prompt, 
            add_special_tokens=True, 
            truncation=True, 
            max_length=self.max_length - 512 # Safety Buffer: Reserve tokens for Response.
        )
        
        # 2. Tokenize Response (add EOS)
        response_tokens = self.tokenizer(
            response + self.tokenizer.eos_token,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length - len(prompt_tokens.input_ids)
        )
        
        input_ids = prompt_tokens.input_ids + response_tokens.input_ids
        attention_mask = prompt_tokens.attention_mask + response_tokens.attention_mask
        
        # Create Labels: Mask prompt with ignore_index
        labels = [self.ignore_index] * len(prompt_tokens.input_ids) + response_tokens.input_ids
        
        # Padding
        pad_len = self.max_length - len(input_ids)
        if pad_len > 0:
            input_ids += [self.tokenizer.pad_token_id] * pad_len
            attention_mask += [0] * pad_len
            labels += [self.ignore_index] * pad_len
        else:
            # Truncate if too long (should be handled by max_length above but double check)
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
            labels = labels[:self.max_length]
            
        # Convert to Tensor
        result = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        
        # Add Multimodal Features (as Tensors)
        if FeatureKey.IMAGE in features:
            # Preprocessor returns numpy, convert to tensor
            img = features[FeatureKey.IMAGE]
            result[FeatureKey.IMAGE] = torch.from_numpy(img).float()
            
        if FeatureKey.AUDIO in features:
            aud = features[FeatureKey.AUDIO]
            result[FeatureKey.AUDIO] = torch.from_numpy(aud).float()
            
        return result

def multimodal_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for MultiModalDataset.
    
    Handles batching of multi-modal data with proper padding and shape consistency.
    Supports mixed batches with different modality combinations.
    
    Args:
        batch: List of sample dictionaries from MultiModalDataset
        
    Returns:
        Batched dictionary with consistent tensor shapes
    """
    if not batch:
        return {}
    
    batch_size = len(batch)
    result = {}
    
    # Collect sample IDs for debugging
    result['sample_ids'] = [item.get('sample_id', f'sample_{i}') for i, item in enumerate(batch)]
    
    # Handle text tokens
    if 'text_tokens' in batch[0]:
        result['text_tokens'] = torch.stack([item['text_tokens'] for item in batch])
        result['attention_mask'] = torch.stack([item['attention_mask'] for item in batch])
        result['labels'] = torch.stack([item['labels'] for item in batch])
    
    # Handle images with shape consistency
    if 'image' in batch[0]:
        images = []
        target_shape = None
        
        # Determine target shape (prefer video format if present)
        for item in batch:
            img = item['image']
            if img.dim() == 4:  # Video format (T, C, H, W)
                target_shape = img.shape
                break
        
        if target_shape is None:
            # All images are single frames (C, H, W)
            target_shape = batch[0]['image'].shape
        
        # Stack images with shape consistency
        for item in batch:
            img = item['image']
            
            # Handle shape mismatches
            if img.shape != target_shape:
                if img.dim() == 3 and len(target_shape) == 4:
                    # Expand single image to video format
                    img = img.unsqueeze(0).repeat(target_shape[0], 1, 1, 1)
                elif img.dim() == 4 and len(target_shape) == 3:
                    # Take first frame of video
                    img = img[0]
                else:
                    # Shape mismatch, use zero tensor
                    logger.warning(f"Image shape mismatch: {img.shape} vs {target_shape}, using zero tensor")
                    img = torch.zeros(target_shape, dtype=torch.float32)
            
            images.append(img)
        
        result['image'] = torch.stack(images)
    
    # Handle audio
    if 'audio' in batch[0]:
        audios = []
        target_shape = batch[0]['audio'].shape
        
        for item in batch:
            audio = item['audio']
            if audio.shape != target_shape:
                logger.warning(f"Audio shape mismatch: {audio.shape} vs {target_shape}, using zero tensor")
                audio = torch.zeros(target_shape, dtype=torch.float32)
            audios.append(audio)
        
        result['audio'] = torch.stack(audios)
    
    return result


def collate_fn(batch):
    """
    Custom collate function to handle multimodal batching.
    Ensures zero-padding for missing modalities to maintain batch alignment.
    """
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    
    result = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }
    
    batch_size = len(batch)

    # Critical Fix: Handle Image padding and Mixed Batch (Image + Video).
    # We must ensure that result[FeatureKey.IMAGE] has batch dimension equal to batch_size.
    # If a sample is missing image, we pad with a zero tensor.
    # If batch contains both Images (3D) and Videos (4D), we promote Images to Video format.
    
    target_img_shape = None
    img_dtype = torch.float32
    has_video = False
    
    # 1. First pass: Check for video to determine target shape
    for item in batch:
        if FeatureKey.IMAGE in item:
            shape = item[FeatureKey.IMAGE].shape
            img_dtype = item[FeatureKey.IMAGE].dtype
            if len(shape) == 4: # (T, C, H, W) -> Video
                has_video = True
                target_img_shape = shape
                break
    
    # 2. Second pass: If no video found, pick first image shape
    if target_img_shape is None:
        for item in batch:
            if FeatureKey.IMAGE in item:
                target_img_shape = item[FeatureKey.IMAGE].shape
                img_dtype = item[FeatureKey.IMAGE].dtype
                break
            
    if target_img_shape is None:
        # No images in the entire batch, skip (Model handles missing key)
        pass
    else:
        # 3. Stack images, handling expansion and padding
        images_batch = []
        
        for item in batch:
            if FeatureKey.IMAGE in item:
                img = item[FeatureKey.IMAGE]
                
                # Handle Mixed Batch: Promote Image to Video
                if has_video and img.dim() == 3: # Image (C, H, W) -> Video (T, C, H, W)
                    # Expand by repeating to match temporal dimension
                    T = target_img_shape[0]
                    img = img.unsqueeze(0).repeat(T, 1, 1, 1)
                
                # Check consistency
                if img.shape != target_img_shape:
                     logger.warning(f"Batch image shape mismatch: {img.shape} vs {target_img_shape}. Replacing with zeros to avoid crash.")
                     images_batch.append(torch.zeros(target_img_shape, dtype=img_dtype))
                else:
                     images_batch.append(img)
            else:
                # Create zero tensor matching the target shape
                images_batch.append(torch.zeros(target_img_shape, dtype=img_dtype))
        
        result[FeatureKey.IMAGE] = torch.stack(images_batch)
    
    # Critical Fix: Handle Audio padding.
    aud_shape = None
    aud_dtype = torch.float32
    for item in batch:
        if FeatureKey.AUDIO in item:
            aud_shape = item[FeatureKey.AUDIO].shape
            aud_dtype = item[FeatureKey.AUDIO].dtype
            break
            
    if aud_shape is None:
        pass
    else:
        audios_batch = []
        for item in batch:
            if FeatureKey.AUDIO in item:
                audios_batch.append(item[FeatureKey.AUDIO])
            else:
                audios_batch.append(torch.zeros(aud_shape, dtype=aud_dtype))
        
        result[FeatureKey.AUDIO] = torch.stack(audios_batch)
                
    return result
