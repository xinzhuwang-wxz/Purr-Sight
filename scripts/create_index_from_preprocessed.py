"""
为现有的预处理文件创建索引文件（index.jsonl）

功能：
1. 扫描预处理目录，找出所有预处理文件
2. 根据文件名解析出样本信息
3. 读取原始数据文件，匹配样本
4. 生成 index.jsonl 文件

使用方法:
    python scripts/create_index_from_preprocessed.py \
        --preprocessed_dir data_formal_alin/preprocessed \
        --data_file data_formal_alin/align_v0.jsonl \
        --output_file data_formal_alin/preprocessed/index.jsonl
"""

import argparse
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Set
from collections import defaultdict


def parse_preprocessed_files(preprocessed_dir: Path) -> Dict[int, Dict[str, str]]:
    """
    扫描预处理目录，解析所有预处理文件
    
    Args:
        preprocessed_dir: 预处理目录路径
        
    Returns:
        字典，键为样本索引，值为预处理文件路径字典
    """
    samples = defaultdict(dict)
    
    # 扫描所有 .pt 文件
    for pt_file in preprocessed_dir.glob("*.pt"):
        # 文件名格式: sample_{sample_idx:06d}_{sample_hash}_{modality}.pt
        # 例如: sample_000000_19f46955_text.pt
        name = pt_file.stem  # 去掉 .pt 扩展名
        
        if not name.startswith("sample_"):
            continue
        
        # 解析文件名
        parts = name.split("_")
        if len(parts) < 4:
            continue
        
        try:
            sample_idx = int(parts[1])
            sample_hash = parts[2]
            modality = "_".join(parts[3:])  # 可能是 text, image, audio, video_metadata
            
            # 验证文件存在且大小>0
            if pt_file.exists() and pt_file.stat().st_size > 0:
                samples[sample_idx][modality] = str(pt_file.relative_to(preprocessed_dir))
        except (ValueError, IndexError):
            continue
    
    return dict(samples)


def match_samples_with_data(
    preprocessed_samples: Dict[int, Dict[str, str]],
    data_file: Path
) -> List[Dict]:
    """
    匹配预处理文件和原始数据，生成索引条目
    
    Args:
        preprocessed_samples: 预处理文件字典（样本索引 -> 预处理文件路径）
        data_file: 原始数据文件路径
        
    Returns:
        索引条目列表
    """
    index_entries = []
    
    # 读取原始数据文件
    with open(data_file, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            
            try:
                sample = json.loads(line)
                
                # 计算样本哈希（与prepre.py中的逻辑一致）
                sample_hash = hashlib.md5(str(sample).encode()).hexdigest()[:8]
                
                # 查找匹配的预处理文件
                # 方法1: 使用行索引（如果预处理时使用的是顺序索引）
                if line_idx in preprocessed_samples:
                    preprocessed_files = preprocessed_samples[line_idx]
                    # 验证哈希是否匹配（如果文件名中包含哈希）
                    # 这里我们假设索引匹配就是正确的，因为预处理时是按顺序处理的
                    index_entry = {
                        "sample_idx": line_idx,
                        "original_sample": sample,
                        "preprocessed_files": preprocessed_files
                    }
                    index_entries.append(index_entry)
                    continue
                
                # 方法2: 通过哈希匹配（更可靠）
                # 扫描所有预处理文件，查找哈希匹配的
                for sample_idx, preprocessed_files in preprocessed_samples.items():
                    # 检查文件名中是否包含匹配的哈希
                    # 文件名格式: sample_{idx}_{hash}_{modality}.pt
                    # 我们需要从文件名中提取哈希
                    first_file = list(preprocessed_files.values())[0]
                    file_name = Path(first_file).stem
                    parts = file_name.split("_")
                    if len(parts) >= 3:
                        file_hash = parts[2]
                        if file_hash == sample_hash:
                            index_entry = {
                                "sample_idx": sample_idx,
                                "original_sample": sample,
                                "preprocessed_files": preprocessed_files
                            }
                            index_entries.append(index_entry)
                            break
                
            except json.JSONDecodeError:
                continue
    
    return index_entries


def main():
    parser = argparse.ArgumentParser(
        description="为现有的预处理文件创建索引文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python scripts/create_index_from_preprocessed.py \\
      --preprocessed_dir data_formal_alin/preprocessed \\
      --data_file data_formal_alin/align_v0.jsonl \\
      --output_file data_formal_alin/preprocessed/index.jsonl
        """
    )
    
    parser.add_argument(
        "--preprocessed_dir",
        type=str,
        required=True,
        help="预处理文件目录路径"
    )
    
    parser.add_argument(
        "--data_file",
        type=str,
        required=True,
        help="原始数据文件路径（align_v0.jsonl）"
    )
    
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="输出索引文件路径（默认: preprocessed_dir/index.jsonl）"
    )
    
    args = parser.parse_args()
    
    preprocessed_dir = Path(args.preprocessed_dir)
    data_file = Path(args.data_file)
    output_file = Path(args.output_file) if args.output_file else preprocessed_dir / "index.jsonl"
    
    # 验证输入
    if not preprocessed_dir.exists():
        raise FileNotFoundError(f"预处理目录不存在: {preprocessed_dir}")
    
    if not data_file.exists():
        raise FileNotFoundError(f"数据文件不存在: {data_file}")
    
    print("=" * 80)
    print("为预处理文件创建索引")
    print("=" * 80)
    print(f"预处理目录: {preprocessed_dir}")
    print(f"数据文件: {data_file}")
    print(f"输出文件: {output_file}")
    
    # 扫描预处理文件
    print("\n扫描预处理文件...")
    preprocessed_samples = parse_preprocessed_files(preprocessed_dir)
    print(f"找到 {len(preprocessed_samples)} 个样本的预处理文件")
    
    # 统计各模态的文件数量
    modality_counts = defaultdict(int)
    for files in preprocessed_samples.values():
        for modality in files.keys():
            modality_counts[modality] += 1
    
    print("\n预处理文件统计:")
    for modality, count in modality_counts.items():
        print(f"  {modality}: {count} 个文件")
    
    # 匹配样本
    print("\n匹配预处理文件和原始数据...")
    index_entries = match_samples_with_data(preprocessed_samples, data_file)
    print(f"匹配到 {len(index_entries)} 个样本")
    
    # 保存索引文件
    print(f"\n保存索引文件: {output_file}")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in index_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"\n✓ 索引文件创建完成！")
    print(f"  总样本数: {len(index_entries)}")
    print(f"  文件大小: {output_file.stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
