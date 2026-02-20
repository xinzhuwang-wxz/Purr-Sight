"""
继续预处理脚本：跳过已存在的.pt文件，继续预处理剩余样本

功能：
1. 读取原始数据文件（align_v0.jsonl）
2. 检查每个样本是否已有对应的.pt文件（通过sample_idx和hash匹配）
3. 如果已存在，跳过；如果不存在，预处理并保存
4. 可以随时停止，下次运行会继续处理剩余样本

使用方法:
    python scripts/update_preprocessed.py \
        --input_file data_formal_alin/align_v0.jsonl \
        --preprocessed_dir data_formal_alin/preprocessed
"""

import argparse
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional
import torch
from tqdm import tqdm

from purrsight.preprocess import Preprocessor
from purrsight.config import FeatureKey


def _save_tensor(tensor_or_dict, file_path: Path) -> bool:
    """
    保存tensor或字典到文件
    
    Args:
        tensor_or_dict: 要保存的tensor或字典
        file_path: 保存路径
        
    Returns:
        是否保存成功
    """
    try:
        torch.save(tensor_or_dict, file_path)
        return file_path.exists() and file_path.stat().st_size > 0
    except Exception as e:
        if file_path.exists():
            file_path.unlink()
        print(f"  警告: 保存文件失败 {file_path.name}: {e}")
        return False


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
    预处理单个样本并保存（跳过已存在的文件）
    
    Args:
        sample: 原始样本字典
        preprocessor: 预处理器实例
        output_dir: 输出目录
        sample_idx: 样本索引
        skip_existing: 如果为True，跳过已存在的预处理文件（默认：True）
        
    Returns:
        预处理后的索引条目，如果失败或已存在返回None
    """
    try:
        # 构建输出文件名（使用索引和哈希）
        sample_hash = hashlib.md5(str(sample).encode()).hexdigest()[:8]
        base_name = f"sample_{sample_idx:06d}_{sample_hash}"
        
        # 检查文件是否已存在
        text_file = output_dir / f"{base_name}_text.pt"
        image_file = output_dir / f"{base_name}_image.pt"
        audio_file = output_dir / f"{base_name}_audio.pt"
        metadata_file = output_dir / f"{base_name}_metadata.pt"
        
        # 如果skip_existing=True，检查已存在的文件
        if skip_existing:
            saved_files = {}
            expected_files = set()
            
            # 检查文本（所有样本都应该有text）
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
                return None  # 已存在，跳过
        
        # 预处理样本（失败会抛异常）
        features = preprocessor.process(sample)
        
        # 保存各个模态的特征
        saved_files = {}
        
        # 保存文本特征
        if FeatureKey.TEXT in features:
            if _save_tensor({
                "input_ids": torch.from_numpy(features[FeatureKey.TEXT]),
                "attention_mask": torch.from_numpy(features[FeatureKey.TEXT_ATTENTION_MASK])
            }, text_file):
                saved_files["text"] = str(text_file.relative_to(output_dir))
        
        # 保存图像特征
        if FeatureKey.IMAGE in features:
            image_tensor = torch.from_numpy(features[FeatureKey.IMAGE])
            if _save_tensor(image_tensor, image_file):
                saved_files["image"] = str(image_file.relative_to(output_dir))
                
                # 记录图像形状信息
                if image_tensor.dim() == 3:
                    saved_files["image_shape"] = list(image_tensor.shape)
        
        # 保存音频特征
        if FeatureKey.AUDIO in features:
            audio_tensor = torch.from_numpy(features[FeatureKey.AUDIO])
            if _save_tensor(audio_tensor, audio_file):
                saved_files["audio"] = str(audio_file.relative_to(output_dir))
        
        # 保存视频元数据
        if "_video_metadata" in features:
            if _save_tensor(features["_video_metadata"], metadata_file):
                saved_files["video_metadata"] = str(metadata_file.relative_to(output_dir))
        
        # 构建索引条目
        index_entry = {
            "sample_idx": sample_idx,
            "original_sample": sample,
            "preprocessed_files": saved_files
        }
        
        return index_entry
        
    except Exception as e:
        print(f"预处理样本 {sample_idx} 失败: {e}")
        return None


def load_data_from_jsonl(file_path: Path) -> list:
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
    主函数：继续预处理数据
    """
    parser = argparse.ArgumentParser(
        description="继续预处理数据，跳过已存在的.pt文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python scripts/update_preprocessed.py \\
      --input_file data_formal_alin/align_v0.jsonl \\
      --preprocessed_dir data_formal_alin/preprocessed
        """
    )
    
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="输入JSONL文件路径（align_v0.jsonl）"
    )
    parser.add_argument(
        "--preprocessed_dir",
        type=str,
        required=True,
        help="预处理文件输出目录"
    )
    parser.add_argument(
        "--force_reprocess",
        action="store_true",
        help="强制重新预处理所有样本（即使文件已存在）"
    )
    
    args = parser.parse_args()
    
    # 解析路径
    input_file = Path(args.input_file)
    if not input_file.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_file}")
    
    preprocessed_dir = Path(args.preprocessed_dir)
    preprocessed_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    print("=" * 80)
    print("继续预处理数据")
    print("=" * 80)
    print(f"输入文件: {input_file}")
    print(f"输出目录: {preprocessed_dir}")
    
    data_list = load_data_from_jsonl(input_file)
    print(f"总样本数: {len(data_list):,}")
    
    # 初始化预处理器
    preprocessor = Preprocessor()
    
    # 统计信息
    skip_existing = not args.force_reprocess
    skipped_count = 0
    processed_count = 0
    failed_count = 0
    
    print(f"\n开始预处理（跳过已存在文件: {skip_existing}）...")
    
    # 预处理所有样本
    for idx, sample in enumerate(tqdm(data_list, desc="预处理中")):
        result = preprocess_sample(
            sample=sample,
            preprocessor=preprocessor,
            output_dir=preprocessed_dir,
            sample_idx=idx,
            skip_existing=skip_existing
        )
        
        if result is None:
            if skip_existing:
                skipped_count += 1
            else:
                failed_count += 1
        else:
            processed_count += 1
    
    # 输出统计信息
    print("\n" + "=" * 80)
    print("预处理完成")
    print("=" * 80)
    print(f"总样本数: {len(data_list):,}")
    print(f"跳过（已存在）: {skipped_count:,}")
    print(f"新处理: {processed_count:,}")
    print(f"失败: {failed_count:,}")
    print(f"\n提示: 运行 update_index.py 来更新索引文件")


if __name__ == "__main__":
    main()
