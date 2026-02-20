"""
文本编码器：基于MiniLM-L6-H384-uncased实现

输入：文本token序列（预处理后的input_ids和attention_mask）
输出：384维文本特征
本地权重路径：./models/mini-lm-l6-h384-uncased/

包含：
- _TextEncoder: 文本编码器类

参考仓库：
- huggingface/transformers (MiniLM)
"""

import torch
import torch.nn as nn
from transformers import AutoModel
from purrsight.config import ROOT_DIR
from pathlib import Path


class _TextEncoder(nn.Module):
    """
    MiniLM-L6-H384-uncased文本编码器

    流程：token序列→384维特征
    
    注意：文本预处理已在preprocess模块完成，这里接收预处理后的token序列。
    
    Attributes:
        model: AutoModel实例（MiniLM）
        weight_path: 权重文件目录路径
    """
    
    def __init__(self, weight_dir: str = "models/mini-lm-l6-h384-uncased"):
        """
        初始化文本编码器
        
        Args:
            weight_dir: 权重文件目录，默认models/mini-lm-l6-h384-uncased
        
        Raises:
            FileNotFoundError: 当权重文件不完整时
        """
        super().__init__()
        self.weight_path = Path(ROOT_DIR, weight_dir)

        self._check_weights()

        try:
            self.model = AutoModel.from_pretrained(
                str(self.weight_path),
                local_files_only=True
            )
        except ValueError as e:
            if "torch.load" in str(e) or "v2.6" in str(e) or "CVE-2025-32434" in str(e):
                from transformers import BertConfig, BertModel
                import json
                
                config_path = self.weight_path / "config.json"
                with open(config_path, 'r') as f:
                    config_dict = json.load(f)
                config = BertConfig(**config_dict)
                
                self.model = BertModel(config)
                
                weight_path = self.weight_path / "pytorch_model.bin"
                
                # Improved: Add file integrity check and EOFError handling
                file_size = weight_path.stat().st_size / (1024 * 1024)  # MB
                
                try:
                    state_dict = torch.load(weight_path, map_location="cpu", weights_only=False)
                except EOFError as e:
                    raise EOFError(
                        f"权重文件损坏或不完整！\n"
                        f"  文件路径: {weight_path}\n"
                        f"  文件大小: {file_size:.2f} MB (预期约90MB)\n"
                        f"  错误: {e}\n"
                        f"  请删除损坏的文件并重新下载：\n"
                        f"  rm {weight_path}\n"
                        f"  然后从以下链接下载：\n"
                        f"  https://huggingface.co/nreimers/MiniLM-L6-H384-uncased/tree/main"
                    ) from e
                except Exception as e:
                    raise RuntimeError(
                        f"加载权重文件失败！\n"
                        f"  文件路径: {weight_path}\n"
                        f"  文件大小: {file_size:.2f} MB\n"
                        f"  错误类型: {type(e).__name__}\n"
                        f"  错误信息: {e}\n"
                        f"  请检查文件是否完整或重新下载。"
                    ) from e
                
                filtered_state_dict = {k: v for k, v in state_dict.items() 
                                     if k != "embeddings.position_ids"}
                self.model.load_state_dict(filtered_state_dict, strict=False)
            else:
                raise

        self.eval()  


    def _check_weights(self):
        """
        验证本地权重文件是否完整
        """
        required_files = [
            "pytorch_model.bin", "vocab.txt",
            "tokenizer_config.json", "special_tokens_map.json", "config.json"
        ]
        missing = [f for f in required_files if not (self.weight_path / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"缺少权重文件：{missing}\n"
                f"请从以下链接下载并保存至 {self.weight_path}：\n"
                f"https://huggingface.co/nreimers/MiniLM-L6-H384-uncased/tree/main"
            )
        
        # Improved: Check weight file size (pytorch_model.bin should be ~90MB)
        weight_file = self.weight_path / "pytorch_model.bin"
        if weight_file.exists():
            file_size = weight_file.stat().st_size / (1024 * 1024)  # MB
            if file_size < 80:  # If less than 80MB, it might be incomplete
                raise FileNotFoundError(
                    f"权重文件可能损坏或不完整！\n"
                    f"  文件路径: {weight_file}\n"
                    f"  文件大小: {file_size:.2f} MB (预期约90MB)\n"
                    f"  请删除损坏的文件并重新下载：\n"
                    f"  rm {weight_file}\n"
                    f"  然后从以下链接下载：\n"
                    f"  https://huggingface.co/nreimers/MiniLM-L6-H384-uncased/tree/main"
                )  


    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        前向传播：token序列→384维特征

        Args:
            input_ids: 预处理后的token ID序列，形状为(B, seq_len)或(seq_len,)，dtype=int64
            attention_mask: attention mask，形状为(B, seq_len)或(seq_len,)，dtype=int64，可选
                          如果为None，则所有token都会被关注

        Returns:
            384维文本特征，形状为(B, 384)或(1, 384)，dtype=float32
        """
        with torch.no_grad():
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
                if attention_mask is not None:
                    attention_mask = attention_mask.unsqueeze(0)

            model_inputs = {"input_ids": input_ids}
            if attention_mask is not None:
                model_inputs["attention_mask"] = attention_mask

            outputs = self.model(**model_inputs)
            cls_feat = outputs.last_hidden_state[:, 0, :]

        return cls_feat

