"""
更新索引文件脚本：将新的.pt文件补充到现有的index.jsonl中

功能：
1. 扫描预处理目录，找出所有.pt文件
2. 读取现有的index.jsonl，获取已有的sample_idx集合
3. 找出新的.pt文件（不在现有index中的）
4. 匹配原始数据文件，生成新的索引条目
5. 追加到现有的index.jsonl中

使用方法:
    python scripts/update_index.py \
        --preprocessed_dir data_formal_alin/preprocessed \
        --data_file data_formal_alin/align_v0.jsonl \
        --index_file data_formal_alin/preprocessed/index.jsonl
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
        # 跳过index.jsonl（如果有同名文件）
        if pt_file.name == "index.jsonl":
            continue
            
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


def load_existing_index(index_file: Path) -> Dict[int, Dict]:
    """
    加载现有的索引文件
    
    Args:
        index_file: 索引文件路径
        
    Returns:
        字典，键为sample_idx，值为索引条目
    """
    existing_entries = {}
    
    if not index_file.exists():
        return existing_entries
    
    with open(index_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            try:
                entry = json.loads(line)
                sample_idx = entry.get('sample_idx')
                if sample_idx is not None:
                    existing_entries[sample_idx] = entry
            except json.JSONDecodeError:
                continue
    
    return existing_entries


def find_new_samples(
    all_samples: Dict[int, Dict[str, str]],
    existing_indices: Set[int]
) -> Dict[int, Dict[str, str]]:
    """
    找出新的样本（不在现有索引中的）
    
    Args:
        all_samples: 所有预处理文件字典
        existing_indices: 现有索引中的sample_idx集合
        
    Returns:
        新的样本字典
    """
    new_samples = {}
    for sample_idx, files in all_samples.items():
        if sample_idx not in existing_indices:
            new_samples[sample_idx] = files
    return new_samples


def match_samples_with_data(
    new_samples: Dict[int, Dict[str, str]],
    data_file: Path
) -> List[Dict]:
    """
    匹配新的预处理文件和原始数据，生成索引条目
    
    Args:
        new_samples: 新的预处理文件字典（样本索引 -> 预处理文件路径）
        data_file: 原始数据文件路径
        
    Returns:
        新的索引条目列表
    """
    index_entries = []
    
    # 读取原始数据文件
    with open(data_file, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            
            # 只处理新的样本索引
            if line_idx not in new_samples:
                continue
            
            try:
                sample = json.loads(line)
                
                # 计算样本哈希（与prepre.py中的逻辑一致）
                sample_hash = hashlib.md5(str(sample).encode()).hexdigest()[:8]
                
                # 获取预处理文件
                preprocessed_files = new_samples[line_idx]
                
                # 验证哈希是否匹配（检查文件名中的哈希）
                # 文件名格式: sample_{idx}_{hash}_{modality}.pt
                first_file = list(preprocessed_files.values())[0]
                file_name = Path(first_file).stem
                parts = file_name.split("_")
                if len(parts) >= 3:
                    file_hash = parts[2]
                    if file_hash == sample_hash:
                        # 哈希匹配，创建索引条目
                        index_entry = {
                            "sample_idx": line_idx,
                            "original_sample": sample,
                            "preprocessed_files": preprocessed_files
                        }
                        index_entries.append(index_entry)
                    else:
                        print(f"警告: 样本 {line_idx} 哈希不匹配: 文件={file_hash}, 数据={sample_hash}")
                
            except json.JSONDecodeError:
                continue
    
    return index_entries


def main():
    """
    主函数：更新索引文件
    """
    parser = argparse.ArgumentParser(
        description="将新的.pt文件补充到现有的index.jsonl中",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python scripts/update_index.py \\
      --preprocessed_dir data_formal_alin/preprocessed \\
      --data_file data_formal_alin/align_v0.jsonl \\
      --index_file data_formal_alin/preprocessed/index.jsonl
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
        "--index_file",
        type=str,
        default=None,
        help="索引文件路径（默认: preprocessed_dir/index.jsonl）"
    )
    
    args = parser.parse_args()
    
    preprocessed_dir = Path(args.preprocessed_dir)
    data_file = Path(args.data_file)
    index_file = Path(args.index_file) if args.index_file else preprocessed_dir / "index.jsonl"
    
    # 验证输入
    if not preprocessed_dir.exists():
        raise FileNotFoundError(f"预处理目录不存在: {preprocessed_dir}")
    
    if not data_file.exists():
        raise FileNotFoundError(f"数据文件不存在: {data_file}")
    
    print("=" * 80)
    print("更新索引文件")
    print("=" * 80)
    print(f"预处理目录: {preprocessed_dir}")
    print(f"数据文件: {data_file}")
    print(f"索引文件: {index_file}")
    
    # 1. 扫描所有预处理文件
    print("\n扫描预处理文件...")
    all_samples = parse_preprocessed_files(preprocessed_dir)
    print(f"找到 {len(all_samples)} 个样本的预处理文件")
    
    # 2. 加载现有索引
    print("\n加载现有索引文件...")
    existing_entries = load_existing_index(index_file)
    existing_indices = set(existing_entries.keys())
    print(f"现有索引包含 {len(existing_indices)} 个样本")
    
    # 3. 找出新的样本
    print("\n查找新的样本...")
    new_samples = find_new_samples(all_samples, existing_indices)
    print(f"发现 {len(new_samples)} 个新样本")
    
    if len(new_samples) == 0:
        print("\n✓ 没有新样本需要添加到索引")
        return
    
    # 4. 匹配新样本和原始数据
    print("\n匹配新样本和原始数据...")
    new_entries = match_samples_with_data(new_samples, data_file)
    print(f"匹配到 {len(new_entries)} 个新索引条目")
    
    if len(new_entries) == 0:
        print("\n⚠️ 没有匹配到新的索引条目")
        return
    
    # 5. 追加到索引文件
    print(f"\n追加 {len(new_entries)} 个新条目到索引文件...")
    
    # 按sample_idx排序
    new_entries.sort(key=lambda x: x['sample_idx'])
    
    # 追加到文件
    with open(index_file, 'a', encoding='utf-8') as f:
        for entry in new_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    # 6. 验证结果
    print("\n验证更新后的索引文件...")
    updated_entries = load_existing_index(index_file)
    print(f"更新后索引包含 {len(updated_entries)} 个样本")
    print(f"新增: {len(new_entries)} 个样本")
    
    print("\n" + "=" * 80)
    print("✓ 索引文件更新完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
