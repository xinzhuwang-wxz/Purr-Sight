"""
Quick Training Diagnosis Script

快速诊断训练性能问题，测量 DataLoader 迭代时间、collate 函数时间、GPU 利用率等。

使用方法:
    python scripts/quick_diagnose.py
"""

import sys
import time
import statistics
from pathlib import Path
from typing import List, Dict

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from torch.utils.data import DataLoader

from purrsight.utils.logging import logger
from train.train_alignment.train_align_conf import AlignmentConfig
from train.train_alignment.train import collate_batch, load_data_from_jsonl
from train.train_alignment.dataset import AlignmentDataset


def measure_dataloader_time(
    dataset: AlignmentDataset,
    batch_size: int,
    num_workers: int,
    num_iterations: int = 20
) -> Dict[str, float]:
    """
    测量DataLoader迭代时间
    
    Returns:
        包含平均时间、中位数时间、最小/最大时间的字典
    """
    logger.info(f"Measuring DataLoader time (num_workers={num_workers}, batch_size={batch_size})...")
    
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_batch,
        pin_memory=False,
        persistent_workers=(num_workers > 0)
    )
    
    times = []
    
    # 预热（跳过第一个batch，因为可能包含初始化开销）
    for batch in train_loader:
        break
    
    # 测量迭代时间
    for i in range(num_iterations):
        start = time.time()
        try:
            batch = next(iter(train_loader))
        except StopIteration:
            # 重新创建DataLoader
            train_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=collate_batch,
                pin_memory=False,
                persistent_workers=(num_workers > 0)
            )
            batch = next(iter(train_loader))
        end = time.time()
        times.append(end - start)
        
        if (i + 1) % 5 == 0:
            logger.info(f"  Processed {i + 1}/{num_iterations} iterations")
    
    return {
        "mean": statistics.mean(times),
        "median": statistics.median(times),
        "min": min(times),
        "max": max(times),
        "std": statistics.stdev(times) if len(times) > 1 else 0.0,
        "all_times": times
    }


def measure_collate_time(
    dataset: AlignmentDataset,
    batch_size: int,
    num_iterations: int = 50
) -> Dict[str, float]:
    """
    测量collate函数时间
    
    Returns:
        包含平均时间、中位数时间、最小/最大时间的字典
    """
    logger.info(f"Measuring collate_batch time (batch_size={batch_size})...")
    
    times = []
    
    for i in range(num_iterations):
        # 获取一个batch的原始数据
        indices = list(range(i * batch_size, min((i + 1) * batch_size, len(dataset))))
        batch_samples = [dataset[idx] for idx in indices]
        
        start = time.time()
        collate_batch(batch_samples)
        end = time.time()
        times.append(end - start)
        
        if (i + 1) % 10 == 0:
            logger.info(f"  Processed {i + 1}/{num_iterations} iterations")
    
    return {
        "mean": statistics.mean(times),
        "median": statistics.median(times),
        "min": min(times),
        "max": max(times),
        "std": statistics.stdev(times) if len(times) > 1 else 0.0,
        "all_times": times
    }


def measure_numpy_to_tensor_time(
    batch_features: Dict,
    modality_masks: Dict,
    device: str,
    num_iterations: int = 100
) -> Dict[str, float]:
    """
    测量numpy到tensor转换时间
    
    Returns:
        包含平均时间、中位数时间、最小/最大时间的字典
    """
    logger.info(f"Measuring numpy→tensor conversion time (device={device})...")
    
    times = []
    
    for i in range(num_iterations):
        start = time.time()
        
        # 模拟training_step中的转换逻辑
        tensor_features = {}
        for key, feat in batch_features.items():
            if key in ["_video_metadata", "_modality_sources"]:
                tensor_features[key] = feat
                continue
            if feat is not None:
                if isinstance(feat, np.ndarray):
                    if feat.dtype == np.int64:
                        tensor_feat = torch.from_numpy(feat).long().to(device)
                    else:
                        tensor_feat = torch.from_numpy(feat.astype(np.float32)).float().to(device)
                    tensor_features[key] = tensor_feat
                elif isinstance(feat, torch.Tensor):
                    tensor_features[key] = feat.to(device)
        
        modality_masks_gpu = {}
        for k, v in modality_masks.items():
            if isinstance(v, np.ndarray):
                modality_masks_gpu[k] = torch.from_numpy(v).bool().to(device)
            elif isinstance(v, torch.Tensor):
                modality_masks_gpu[k] = v.to(device)
        
        end = time.time()
        times.append(end - start)
    
    return {
        "mean": statistics.mean(times),
        "median": statistics.median(times),
        "min": min(times),
        "max": max(times),
        "std": statistics.stdev(times) if len(times) > 1 else 0.0,
        "all_times": times
    }


def check_gpu_utilization():
    """检查GPU利用率（仅信息性，需要外部工具）"""
    logger.info("\nGPU Utilization Check:")
    
    if torch.backends.mps.is_available():
        logger.info("  Device: MPS (Apple Silicon)")
        logger.info("  Note: Use Activity Monitor > Window > GPU History to monitor GPU usage")
    elif torch.cuda.is_available():
        logger.info("  Device: CUDA")
        logger.info("  Note: Run 'nvidia-smi -l 1' in another terminal to monitor GPU usage")
    else:
        logger.info("  Device: CPU")
        logger.info("  Note: No GPU available")


def print_summary(stats: Dict[str, Dict[str, float]]):
    """打印诊断摘要"""
    logger.info("\n" + "=" * 80)
    logger.info("Diagnosis Summary")
    logger.info("=" * 80)
    
    for name, stat in stats.items():
        logger.info(f"\n{name}:")
        logger.info(f"  Mean:   {stat['mean']*1000:.2f} ms")
        logger.info(f"  Median: {stat['median']*1000:.2f} ms")
        logger.info(f"  Min:    {stat['min']*1000:.2f} ms")
        logger.info(f"  Max:    {stat['max']*1000:.2f} ms")
        logger.info(f"  Std:    {stat['std']*1000:.2f} ms")
    
    # 计算总时间占比
    if "dataloader" in stats and "collate" in stats:
        total_time = stats["dataloader"]["mean"]
        collate_time = stats["collate"]["mean"]
        if total_time > 0:
            collate_ratio = (collate_time / total_time) * 100
            logger.info(f"\nCollate time ratio: {collate_ratio:.1f}% of DataLoader time")
            if collate_ratio > 50:
                logger.warning("  ⚠️  Collate function is taking >50% of DataLoader time - consider optimizing!")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Quick training diagnosis")
    parser.add_argument("--data_path", type=str, default="data/test_alignment/train.jsonl",
                       help="Path to training data")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size")
    parser.add_argument("--num_workers", type=int, default=2,
                       help="Number of DataLoader workers")
    parser.add_argument("--use_preprocessed", action="store_true",
                       help="Use preprocessed data")
    parser.add_argument("--preprocessed_dir", type=str, default=None,
                       help="Preprocessed data directory")
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("Quick Training Diagnosis")
    logger.info("=" * 80)
    logger.info(f"Config: batch_size={args.batch_size}, num_workers={args.num_workers}")
    
    # 加载数据
    logger.info("\nLoading data...")
    data_list = load_data_from_jsonl(args.data_path)
    dataset = AlignmentDataset(
        data_list=data_list[:args.batch_size * 50],  # 只加载需要的样本
        device="cpu",
        use_preprocessed=args.use_preprocessed,
        preprocessed_dir=Path(args.preprocessed_dir) if args.preprocessed_dir else None
    )
    logger.info(f"Loaded {len(dataset)} samples")
    
    # 测量各项时间
    stats = {}
    
    # 1. DataLoader时间
    stats["dataloader"] = measure_dataloader_time(
        dataset, args.batch_size, args.num_workers, num_iterations=20
    )
    
    # 2. Collate时间
    stats["collate"] = measure_collate_time(
        dataset, args.batch_size, num_iterations=30
    )
    
    # 3. Numpy→Tensor转换时间
    logger.info("\nPreparing batch for numpy→tensor conversion test...")
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,  # 单线程，避免干扰
        collate_fn=collate_batch,
        pin_memory=False
    )
    batch_features, modality_masks = next(iter(train_loader))
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    stats["numpy_to_tensor"] = measure_numpy_to_tensor_time(
        batch_features, modality_masks, device, num_iterations=50
    )
    
    # 4. GPU利用率检查
    check_gpu_utilization()
    
    # 打印摘要
    print_summary(stats)
    
    logger.info("\n" + "=" * 80)
    logger.info("Diagnosis complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
