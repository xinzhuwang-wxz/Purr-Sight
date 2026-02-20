"""
文本预处理模块：分词、padding、token转换

包含：
- _TextProcessor: 文本预处理器类
"""

import numpy as np
from typing import Union, List
from transformers import BertTokenizer
from purrsight.config import ROOT_DIR
from pathlib import Path


class _TextProcessor:
    """
    文本预处理器：分词→padding→token id转换
    
    将文本字符串转换为标准token序列，形状为(seq_len,)或(B, seq_len)。
    
    Attributes:
        max_length: 最大序列长度
        tokenizer: BertTokenizer实例
    """
    
    def __init__(self, tokenizer_name: str = "mini-lm-l6-h384-uncased", max_length: int = 32):
        """
        初始化文本预处理器
        
        Args:
            tokenizer_name: Tokenizer文件夹名，位于./models/下
            max_length: 最大序列长度，默认32
        """
        self.max_length = max_length
        
        tokenizer_path = Path(ROOT_DIR, "models", tokenizer_name)
        self.tokenizer = BertTokenizer.from_pretrained(
            str(tokenizer_path),
            local_files_only=True,
            trust_remote_code=False
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def process_text(self, text: Union[str, List[str]]) -> dict[str, np.ndarray]:
        """
        文本预处理：分词→padding→token id转换
        
        Args:
            text: 输入文本字符串或字符串列表（batch）
        
        Returns:
            包含"input_ids"和"attention_mask"的字典：
            - 单个文本：input_ids形状为(seq_len,)，attention_mask形状为(seq_len,)
            - Batch文本：input_ids形状为(B, seq_len)，attention_mask形状为(B, seq_len)
            - dtype=int64，token id序列
        """
        is_batch = isinstance(text, list)
        if not is_batch:
            text = [text]
        
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].numpy().astype(np.int64)
        attention_mask = encoded['attention_mask'].numpy().astype(np.int64)
        
        # 如果不是batch，去掉batch维度
        if not is_batch:
            input_ids = input_ids.squeeze(0)
            attention_mask = attention_mask.squeeze(0)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }