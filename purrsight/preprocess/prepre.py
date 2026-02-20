"""
离线预处理脚本：一次性预处理所有数据，训练时只load tensor

功能：
1. 读取原始数据（video/image/audio/text）
2. 执行完整预处理：
   - video → 16帧tensor (16, 3, 224, 224) + audio tensor (64, 256)
   - image → tensor (3, 224, 224)
   - audio → tensor (64, 256)
   - text → token ids (seq_len,)
3. 保存为.pt文件
4. 生成索引文件（JSONL），指向预处理后的文件

包含：
- compute_file_hash: 计算文件MD5哈希值
- preprocess_sample: 预处理单个样本并保存
- load_data_from_jsonl: 从JSONL文件加载数据
- main: 主函数

使用方法：
    python -m purrsight.preprocess.prepre --input_file data/train.jsonl --output_dir data/preprocessed
"""

import argparse
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional
import torch
import numpy as np
from tqdm import tqdm

from purrsight.preprocess import Preprocessor
from purrsight.config import FeatureKey, Modality, ROOT_DIR


def compute_file_hash(file_path: Path) -> str:
    """
    计算文件MD5哈希值
    
    Args:
        file_path: 文件路径
    
    Returns:
        MD5哈希值的十六进制字符串
    """
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def _save_tensor(tensor_or_dict, file_path: Path) -> None:
    """
    保存tensor
    
    Args:
        tensor_or_dict: 要保存的tensor或字典
        file_path: 保存路径
        
    Raises:
        Exception: 当保存失败时
    """
    torch.save(tensor_or_dict, file_path)
    
    # 基本验证：文件存在且大小>0
    if not file_path.exists() or file_path.stat().st_size == 0:
        if file_path.exists():
            file_path.unlink()
        raise ValueError(f"保存的文件为空: {file_path}")


def _check_file_exists(file_path: Path) -> bool:
    """
    检查文件是否存在且大小>0
    
    Args:
        file_path: 文件路径
        
    Returns:
        文件是否存在且大小>0
    """
    return file_path.exists() and file_path.stat().st_size > 0


def preprocess_sample(
    sample: Dict[str, Any],
    preprocessor: Preprocessor,
    output_dir: Path,
    sample_idx: int,
    skip_existing: bool = True
) -> Optional[Dict[str, Any]]:
    """
    预处理单个样本并保存
    
    Args:
        sample: 原始样本字典
        preprocessor: 预处理器实例
        output_dir: 输出目录
        sample_idx: 样本索引
        skip_existing: 如果为True，跳过已存在的预处理文件（默认：True）
        
    Returns:
        预处理后的索引条目，如果失败返回None
    """
    try:
        # 构建输出文件名（使用索引和哈希）
        sample_hash = hashlib.md5(str(sample).encode()).hexdigest()[:8]
        base_name = f"sample_{sample_idx:06d}_{sample_hash}"
        
        # 🔧 改进：检查文件是否已存在
        text_file = output_dir / f"{base_name}_text.pt"
        image_file = output_dir / f"{base_name}_image.pt"
        audio_file = output_dir / f"{base_name}_audio.pt"
        metadata_file = output_dir / f"{base_name}_metadata.pt"
        
        # 如果skip_existing=True，检查已存在的文件，如果所有应该存在的文件都存在，则跳过预处理
        if skip_existing:
            saved_files = {}
            expected_files = set()
            
            # 检查文本
            if "text" in sample and sample.get("text"):
                expected_files.add("text")
                if _check_file_exists(text_file):
                    saved_files["text"] = str(text_file.relative_to(output_dir))
            
            # 检查图像（如果原始样本有image，且没有video）
            if "image" in sample and sample.get("image") and "video" not in sample:
                expected_files.add("image")
                if _check_file_exists(image_file):
                    saved_files["image"] = str(image_file.relative_to(output_dir))
            
            # 检查视频（如果原始样本有video，会生成image和metadata）
            if "video" in sample and sample.get("video"):
                expected_files.add("image")
                expected_files.add("video_metadata")
                if _check_file_exists(image_file):
                    saved_files["image"] = str(image_file.relative_to(output_dir))
                if _check_file_exists(metadata_file):
                    saved_files["video_metadata"] = str(metadata_file.relative_to(output_dir))
            
            # 检查音频
            if "audio" in sample and sample.get("audio"):
                expected_files.add("audio")
                if _check_file_exists(audio_file):
                    saved_files["audio"] = str(audio_file.relative_to(output_dir))
            
            # 如果所有期望的文件都存在，跳过预处理
            if expected_files and len(saved_files) == len(expected_files):
                return {
                    "sample_idx": sample_idx,
                    "original_sample": sample,
                    "preprocessed_files": saved_files
                }
        
        # 预处理样本（失败会抛异常）
        features = preprocessor.process(sample)
        
        # 保存各个模态的特征
        saved_files = {}
        
        # 保存文本特征
        if FeatureKey.TEXT in features:
            _save_tensor({
                "input_ids": torch.from_numpy(features[FeatureKey.TEXT]),
                "attention_mask": torch.from_numpy(features[FeatureKey.TEXT_ATTENTION_MASK])
            }, text_file)
            saved_files["text"] = str(text_file.relative_to(output_dir))
        
        # 保存图像特征
        if FeatureKey.IMAGE in features:
            image_tensor = torch.from_numpy(features[FeatureKey.IMAGE])
            _save_tensor(image_tensor, image_file)
            saved_files["image"] = str(image_file.relative_to(output_dir))
            
            # 记录图像形状信息
            if image_tensor.dim() == 3:
                saved_files["image_shape"] = list(image_tensor.shape)
        
        # 保存音频特征
        if FeatureKey.AUDIO in features:
            audio_tensor = torch.from_numpy(features[FeatureKey.AUDIO])
            _save_tensor(audio_tensor, audio_file)
            saved_files["audio"] = str(audio_file.relative_to(output_dir))
        
        # 保存视频元数据
        if "_video_metadata" in features:
            _save_tensor(features["_video_metadata"], metadata_file)
            saved_files["video_metadata"] = str(metadata_file.relative_to(output_dir))
        
        # 构建索引条目
        index_entry = {
            "sample_idx": sample_idx,
            "original_sample": sample,  # 保留原始样本信息
            "preprocessed_files": saved_files
        }
        
        return index_entry
        
    except Exception as e:
        print(f"预处理样本 {sample_idx} 失败: {e}")
        return None


def load_data_from_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """
    从JSONL文件加载数据
    
    Args:
        file_path: JSONL文件路径
    
    Returns:
        数据列表，每个元素是一个样本字典
    """
    data_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data_list.append(json.loads(line))
    return data_list


def main():
    """
    主函数：离线预处理数据
    
    从JSONL文件读取原始数据，预处理后保存为.pt文件，并生成索引文件。
    """
    parser = argparse.ArgumentParser(description="离线预处理数据")
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="输入JSONL文件路径"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/preprocessed",
        help="输出目录（默认：data/preprocessed）"
    )
    parser.add_argument(
        "--index_file",
        type=str,
        default=None,
        help="索引文件路径（默认：output_dir/index.jsonl）"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="并行处理worker数（默认：1，单进程）"
    )
    parser.add_argument(
        "--force_reprocess",
        action="store_true",
        help="强制重新预处理所有样本（即使文件已存在）"
    )
    parser.add_argument(
        "--cleanup_corrupted",
        action="store_true",
        help="清理损坏的文件并重新预处理"
    )
    
    args = parser.parse_args()
    
    # 解析路径
    input_file = Path(args.input_file)
    if not input_file.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_file}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    index_file = Path(args.index_file) if args.index_file else output_dir / "index.jsonl"
    
    # 加载数据
    print(f"加载数据从: {input_file}")
    data_list = load_data_from_jsonl(input_file)
    print(f"共 {len(data_list)} 个样本")
    
    # 初始化预处理器
    preprocessor = Preprocessor()
    
    # 预处理所有样本
    print(f"开始预处理，输出目录: {output_dir}")
    index_entries = []
    
    # 🔧 改进：添加跳过已存在文件的选项（默认启用）
    skip_existing = not args.force_reprocess  # 如果force_reprocess=True，则skip_existing=False
    skipped_count = 0
    reprocessed_count = 0
    
    # 🔧 修复：如果cleanup_corrupted=True，先扫描并删除损坏的文件
    if args.cleanup_corrupted:
        print("清理损坏的文件...")
        corrupted_count = 0
        for pt_file in output_dir.glob("*.pt"):
            # 🔧 修复：使用正确的函数名
            if not _check_file_exists(pt_file):
                corrupted_count += 1
                pt_file.unlink(missing_ok=True)
        print(f"  发现并删除 {corrupted_count} 个损坏的文件")
    
    # 🔧 改进：并行处理
    if args.num_workers > 1:
        print(f"使用 {args.num_workers} 个worker并行处理...")
        from concurrent.futures import ProcessPoolExecutor, as_completed
        
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = []
            for idx, sample in enumerate(data_list):
                futures.append(executor.submit(preprocess_sample, sample, preprocessor, output_dir, idx, skip_existing))
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="预处理中 (并行)"):
                try:
                    index_entry = future.result()
                    if index_entry:
                        index_entries.append(index_entry)
                except Exception as e:
                    print(f"Worker exception: {e}")
    else:
        for idx, sample in enumerate(tqdm(data_list, desc="预处理中")):
            index_entry = preprocess_sample(sample, preprocessor, output_dir, idx, skip_existing=skip_existing)
            if index_entry:
                index_entries.append(index_entry)
    
    # 保存索引文件
    print(f"保存索引文件: {index_file}")
    with open(index_file, 'w', encoding='utf-8') as f:
        for entry in index_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"\n预处理完成！")
    print(f"  - 成功处理: {len(index_entries)}/{len(data_list)} 个样本")
    if skip_existing:
        print(f"  - 跳过已存在的有效文件")
    if args.force_reprocess:
        print(f"  - 强制重新预处理模式（所有文件已重新生成）")
    if args.cleanup_corrupted:
        print(f"  - 已清理损坏的文件")
    print(f"  - 输出目录: {output_dir}")
    print(f"  - 索引文件: {index_file}")


if __name__ == "__main__":
    main()
