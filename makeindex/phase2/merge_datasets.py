"""
合并 ESC-50 和 LAION Subset 数据集为统一的 align_v0.jsonl 文件

功能:
1. 读取 esc50_train.jsonl (audio-text pairs)
2. 读取 laion_subset_train.jsonl (image-text pairs)
3. 合并为一个 align_v0.jsonl 文件
"""

import json
from pathlib import Path
from typing import Dict, List


def merge_datasets(
    esc50_file: str,
    laion_file: str,
    output_file: str
) -> None:
    """
    合并两个数据集文件
    
    Args:
        esc50_file: ESC-50 数据集文件路径
        laion_file: LAION Subset 数据集文件路径
        output_file: 输出文件路径
    """
    esc50_path = Path(esc50_file)
    laion_path = Path(laion_file)
    output_path = Path(output_file)
    
    # 检查输入文件是否存在
    if not esc50_path.exists():
        raise FileNotFoundError(f"ESC-50文件不存在: {esc50_file}")
    
    if not laion_path.exists():
        raise FileNotFoundError(f"LAION文件不存在: {laion_file}")
    
    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    esc50_count = 0
    laion_count = 0
    total_count = 0
    
    print("=" * 80)
    print("Merging Datasets")
    print("=" * 80)
    print(f"\nInput files:")
    print(f"  ESC-50: {esc50_path}")
    print(f"  LAION Subset: {laion_path}")
    print(f"\nOutput file: {output_path}")
    
    # 合并文件
    with open(output_path, 'w', encoding='utf-8') as out_f:
        # 读取并写入 ESC-50 数据
        print(f"\nReading ESC-50 dataset...")
        with open(esc50_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        # 验证记录格式
                        if 'text' in record and 'audio' in record:
                            out_f.write(json.dumps(record, ensure_ascii=False) + '\n')
                            esc50_count += 1
                            total_count += 1
                        # 格式不正确的记录直接跳过，不打印警告
                    except json.JSONDecodeError:
                        # JSON解析失败直接跳过，不打印警告
                        pass
        
        print(f"  Processed {esc50_count:,} ESC-50 samples")
        
        # 读取并写入 LAION Subset 数据
        print(f"\nReading LAION Subset dataset...")
        with open(laion_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        # 验证记录格式
                        if 'text' in record and 'image' in record:
                            out_f.write(json.dumps(record, ensure_ascii=False) + '\n')
                            laion_count += 1
                            total_count += 1
                        # 格式不正确的记录直接跳过，不打印警告
                    except json.JSONDecodeError:
                        # JSON解析失败直接跳过，不打印警告
                        pass
        
        print(f"  Processed {laion_count:,} LAION Subset samples")
    
    print("\n" + "=" * 80)
    print("Merge Complete!")
    print("=" * 80)
    print(f"\nSummary:")
    print(f"  ESC-50 samples: {esc50_count:,}")
    print(f"  LAION Subset samples: {laion_count:,}")
    print(f"  Total samples: {total_count:,}")
    print(f"\nOutput file: {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="合并 ESC-50 和 LAION Subset 数据集为统一的 align_v0.jsonl 文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python merge_datasets.py  # 使用默认路径
  python merge_datasets.py --esc50 custom_esc50.jsonl --laion custom_laion.jsonl --output custom_output.jsonl
        """
    )
    
    parser.add_argument(
        "--esc50",
        type=str,
        default="data_formal_alin/esc50_train.jsonl",
        help="ESC-50 数据集文件路径（默认: data_formal_alin/esc50_train.jsonl）"
    )
    
    parser.add_argument(
        "--laion",
        type=str,
        default="data_formal_alin/laion_subset_train.jsonl",
        help="LAION Subset 数据集文件路径（默认: data_formal_alin/laion_subset_train.jsonl）"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="data_formal_alin/align_v0.jsonl",
        help="输出文件路径（默认: data_formal_alin/align_v0.jsonl）"
    )
    
    args = parser.parse_args()
    
    merge_datasets(
        esc50_file=args.esc50,
        laion_file=args.laion,
        output_file=args.output
    )
