"""
清理损坏的预处理文件脚本

功能：
1. 扫描预处理目录，检测损坏的.pt文件
2. 从索引文件中移除损坏文件对应的条目
3. 可选：删除损坏的文件
4. 可选：重新预处理损坏的样本

使用方法:
    python scripts/cleanup_corrupted_preprocessed.py --preprocessed_dir data_formal_alin/preprocessed --cleanup
    python scripts/cleanup_corrupted_preprocessed.py --preprocessed_dir data_formal_alin/preprocessed --reprocess
"""

import argparse
import json
import torch
from pathlib import Path
from typing import Dict, Any, List, Set
from tqdm import tqdm

from purrsight.utils.logging import logger


def verify_pt_file(file_path: Path) -> bool:
    """
    验证.pt文件是否完整
    
    Args:
        file_path: 文件路径
        
    Returns:
        文件是否完整可用
    """
    if not file_path.exists():
        return False
    
    try:
        # 检查文件大小
        if file_path.stat().st_size == 0:
            return False
        
        # 尝试加载验证
        torch.load(file_path, map_location="cpu", weights_only=False)
        return True
    except Exception:
        return False


def scan_corrupted_files(preprocessed_dir: Path) -> Dict[str, List[Path]]:
    """
    扫描预处理目录，找出损坏的文件
    
    Args:
        preprocessed_dir: 预处理目录
        
    Returns:
        字典，包含损坏的文件列表，按类型分类
    """
    corrupted = {
        "text": [],
        "image": [],
        "audio": [],
        "metadata": []
    }
    
    logger.info(f"扫描预处理目录: {preprocessed_dir}")
    
    # 扫描所有.pt文件
    pt_files = list(preprocessed_dir.glob("*.pt"))
    logger.info(f"找到 {len(pt_files)} 个.pt文件")
    
    for pt_file in tqdm(pt_files, desc="验证文件"):
        if not verify_pt_file(pt_file):
            # 根据文件名判断类型
            name = pt_file.name
            if "_text.pt" in name:
                corrupted["text"].append(pt_file)
            elif "_image.pt" in name:
                corrupted["image"].append(pt_file)
            elif "_audio.pt" in name:
                corrupted["audio"].append(pt_file)
            elif "_metadata.pt" in name:
                corrupted["metadata"].append(pt_file)
    
    total_corrupted = sum(len(files) for files in corrupted.values())
    logger.info(f"\n发现损坏文件:")
    logger.info(f"  文本文件: {len(corrupted['text'])}")
    logger.info(f"  图像文件: {len(corrupted['image'])}")
    logger.info(f"  音频文件: {len(corrupted['audio'])}")
    logger.info(f"  元数据文件: {len(corrupted['metadata'])}")
    logger.info(f"  总计: {total_corrupted}")
    
    return corrupted


def cleanup_index_file(index_file: Path, corrupted_files: Dict[str, List[Path]], cleanup_files: bool = False):
    """
    清理索引文件，移除损坏文件对应的条目
    
    Args:
        index_file: 索引文件路径
        corrupted_files: 损坏文件字典
        cleanup_files: 是否删除损坏的文件
    """
    if not index_file.exists():
        logger.warning(f"索引文件不存在: {index_file}")
        return
    
    # 构建损坏文件的集合（用于快速查找）
    corrupted_set: Set[str] = set()
    for file_list in corrupted_files.values():
        for file_path in file_list:
            corrupted_set.add(file_path.name)
    
    logger.info(f"\n清理索引文件: {index_file}")
    
    # 读取索引文件
    valid_entries = []
    removed_count = 0
    
    with open(index_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            
            try:
                entry = json.loads(line)
                preprocessed_files = entry.get("preprocessed_files", {})
                
                # 检查是否有损坏的文件
                has_corrupted = False
                files_to_remove = []
                
                for file_type, relative_path in preprocessed_files.items():
                    if file_type in ["text", "image", "audio", "video_metadata"]:
                        file_path = index_file.parent / relative_path
                        if file_path.name in corrupted_set:
                            has_corrupted = True
                            files_to_remove.append(file_type)
                            
                            # 如果cleanup_files=True，删除损坏的文件
                            if cleanup_files and file_path.exists():
                                file_path.unlink()
                                logger.debug(f"删除损坏文件: {file_path.name}")
                
                # 如果有损坏的文件，移除对应的条目
                if has_corrupted:
                    removed_count += 1
                    # 从preprocessed_files中移除损坏的文件引用
                    for file_type in files_to_remove:
                        preprocessed_files.pop(file_type, None)
                    
                    # 如果preprocessed_files为空，跳过这个条目
                    if not preprocessed_files:
                        continue
                    
                    # 更新entry
                    entry["preprocessed_files"] = preprocessed_files
                
                valid_entries.append(entry)
                
            except json.JSONDecodeError as e:
                logger.warning(f"索引文件第 {line_num} 行JSON解析失败: {e}")
                continue
    
    # 写回清理后的索引文件
    if removed_count > 0 or cleanup_files:
        backup_file = index_file.with_suffix('.jsonl.backup')
        logger.info(f"备份原索引文件到: {backup_file}")
        index_file.rename(backup_file)
        
        logger.info(f"写入清理后的索引文件...")
        with open(index_file, 'w', encoding='utf-8') as f:
            for entry in valid_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        logger.info(f"✓ 索引文件清理完成:")
        logger.info(f"  原始条目数: {len(valid_entries) + removed_count}")
        logger.info(f"  移除条目数: {removed_count}")
        logger.info(f"  有效条目数: {len(valid_entries)}")
    else:
        logger.info("✓ 索引文件无需清理（没有损坏的文件引用）")


def main():
    parser = argparse.ArgumentParser(description="清理损坏的预处理文件")
    parser.add_argument(
        "--preprocessed_dir",
        type=str,
        required=True,
        help="预处理目录路径"
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="删除损坏的文件"
    )
    parser.add_argument(
        "--reprocess",
        action="store_true",
        help="重新预处理损坏的样本（需要原始数据文件）"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        help="原始数据文件路径（reprocess模式需要）"
    )
    
    args = parser.parse_args()
    
    preprocessed_dir = Path(args.preprocessed_dir)
    if not preprocessed_dir.exists():
        logger.error(f"预处理目录不存在: {preprocessed_dir}")
        return 1
    
    index_file = preprocessed_dir / "index.jsonl"
    
    logger.info("=" * 80)
    logger.info("清理损坏的预处理文件")
    logger.info("=" * 80)
    
    # 1. 扫描损坏的文件
    corrupted_files = scan_corrupted_files(preprocessed_dir)
    total_corrupted = sum(len(files) for files in corrupted_files.values())
    
    if total_corrupted == 0:
        logger.info("\n✓ 没有发现损坏的文件")
        return 0
    
    # 2. 清理索引文件
    cleanup_index_file(index_file, corrupted_files, cleanup_files=args.cleanup)
    
    # 3. 如果需要重新预处理
    if args.reprocess:
        if not args.input_file:
            logger.error("重新预处理需要指定 --input_file")
            return 1
        
        logger.info("\n" + "=" * 80)
        logger.info("重新预处理损坏的样本")
        logger.info("=" * 80)
        
        # 这里可以调用预处理脚本重新处理
        # 由于需要知道哪些样本损坏了，需要从索引文件中找出对应的原始样本
        logger.info("提示: 请运行预处理脚本重新处理:")
        logger.info(f"  python -m purrsight.preprocess.prepre \\")
        logger.info(f"    --input_file {args.input_file} \\")
        logger.info(f"    --output_dir {preprocessed_dir} \\")
        logger.info(f"    --index_file {index_file}")
        logger.info("\n注意: 预处理脚本会自动跳过已存在的有效文件")
    
    logger.info("\n" + "=" * 80)
    logger.info("✓ 清理完成")
    logger.info("=" * 80)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
