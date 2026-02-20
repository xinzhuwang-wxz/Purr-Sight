"""
Training Profiling Script

使用 torch.profiler 分析训练循环的性能瓶颈，识别 DataLoader、collate、forward、backward、validation 的耗时。

使用方法:
    python scripts/profile_training.py

输出:
    - Chrome trace file: profile_trace.json (可在 chrome://tracing 中打开)
    - 性能报告: 各阶段的平均耗时和占比
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any
import time

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.profiler import profile, record_function, ProfilerActivity

from purrsight.utils.logging import logger
from train.train_alignment.train_align_conf import AlignmentConfig
from train.train_alignment.train import train_loop_per_worker, collate_batch, load_data_from_jsonl
from train.train_alignment.dataset import AlignmentDataset
from train.train_alignment.lightning_module import ContrastiveAlignmentModule


def profile_training_loop(
    config: AlignmentConfig,
    num_batches: int = 10,
    output_dir: Path = None
):
    """
    分析训练循环的性能瓶颈
    
    Args:
        config: 训练配置
        num_batches: 分析的batch数量（用于快速分析）
        output_dir: 输出目录，如果为None则使用当前目录
    """
    if output_dir is None:
        output_dir = project_root / "profiles"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("Training Performance Profiling")
    logger.info("=" * 80)
    logger.info(f"Config: batch_size={config.batch_size}, num_workers={config.num_workers}")
    logger.info(f"Profiling {num_batches} batches...")
    
    # 加载数据
    logger.info("Loading data...")
    if config.data_path.endswith('.jsonl'):
        data_list = load_data_from_jsonl(config.data_path)
    else:
        raise ValueError(f"Unsupported data format: {config.data_path}")
    
    # 创建数据集
    dataset = AlignmentDataset(
        data_list=data_list[:config.batch_size * num_batches * 2],  # 只加载需要的样本
        device="cpu",
        use_preprocessed=config.use_preprocessed,
        preprocessed_dir=Path(config.preprocessed_dir) if config.preprocessed_dir else None
    )
    
    # 创建DataLoader
# scripts/profile_training.py 第75-83行
    train_loader = DataLoader(
    dataset,
    batch_size=config.batch_size,
    shuffle=False,  # 不shuffle以便可复现
    num_workers=config.num_workers,
    collate_fn=collate_batch,
    pin_memory=False,  # MPS不支持pin_memory
    persistent_workers=(config.num_workers > 0),
    prefetch_factor=2 if config.num_workers > 0 else None  # 添加这一行，与训练代码保持一致
    )
    
    # 创建模型
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = ContrastiveAlignmentModule(config).to(device)
    model.train()
    
    # 创建优化器
    optimizer = torch.optim.AdamW(
        model.aligner.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # 性能统计
    timing_stats = {
        "dataloader": [],
        "collate": [],
        "numpy_to_tensor": [],
        "encode_batch": [],
        "forward": [],
        "loss": [],
        "backward": [],
        "total": []
    }
    
    logger.info("Starting profiling...")
    
    # 使用 torch.profiler 进行详细分析
    # 🔧 修复：MPS不支持ProfilerActivity.MPS，只能使用CPU profiling
    # CUDA设备可以使用ProfilerActivity.CUDA，MPS设备只能使用CPU profiling
    if torch.backends.mps.is_available():
        activities = [ProfilerActivity.CPU]  # MPS只能使用CPU profiling
    elif torch.cuda.is_available():
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    else:
        activities = [ProfilerActivity.CPU]
    
    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= num_batches:
                break
            
            batch_start = time.time()
            
            with record_function("dataloader_iteration"):
                # DataLoader迭代时间已在batch获取时测量
                pass
            
            with record_function("collate_batch"):
                # collate已在DataLoader中完成
                pass
            
            with record_function("numpy_to_tensor"):
                # 转换numpy到tensor
                batch_features, modality_masks, batch_size = batch
                import numpy as np
                tensor_features = {}
                for key, feat in batch_features.items():
                    if key in ["_video_metadata", "_modality_sources"]:
                        tensor_features[key] = feat
                        continue
                    if feat is not None and isinstance(feat, torch.Tensor):
                        tensor_features[key] = feat.to(device)
                    elif feat is not None and isinstance(feat, np.ndarray):
                        if feat.dtype == np.int64:
                            tensor_features[key] = torch.from_numpy(feat).long().to(device)
                        else:
                            tensor_features[key] = torch.from_numpy(feat.astype(np.float32)).float().to(device)
                    else:
                        tensor_features[key] = feat
            
            with record_function("encode_batch"):
                encoder_outputs = model.encode_batch(tensor_features)
            
            with record_function("forward"):
                import numpy as np
                modality_masks_gpu = {}
                for k, v in modality_masks.items():
                    if isinstance(v, np.ndarray):
                        modality_masks_gpu[k] = torch.from_numpy(v).bool().to(device)
                    elif isinstance(v, torch.Tensor):
                        modality_masks_gpu[k] = v.to(device)
                    else:
                        modality_masks_gpu[k] = v
                aligned_features, modality_presence = model.forward(encoder_outputs, modality_masks_gpu)
            
            with record_function("loss"):
                total_loss, loss_dict = model.loss_fn(
                    aligned_features,
                    modality_presence=modality_presence,
                    logit_scales=model.aligner.get_logit_scales() if model.aligner.use_temperature_scaling else None,
                    modality_masks=modality_masks_gpu
                )
            
            with record_function("backward"):
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
            
            batch_end = time.time()
            timing_stats["total"].append(batch_end - batch_start)
            
            if (batch_idx + 1) % 5 == 0:
                logger.info(f"Processed {batch_idx + 1}/{num_batches} batches")
    
    # 保存trace文件
    trace_file = output_dir / "profile_trace.json"
    prof.export_chrome_trace(str(trace_file))
    logger.info(f"Chrome trace saved to: {trace_file}")
    logger.info("Open chrome://tracing in Chrome browser and load this file to visualize")
    
    # 打印性能报告
    logger.info("\n" + "=" * 80)
    logger.info("Performance Summary")
    logger.info("=" * 80)
    
    # 分析profiler结果
    events = prof.key_averages(group_by_input_shape=True)
    
    logger.info("\nTop time-consuming operations:")
    for event in events[:20]:
        if event.key != "<built-in method ...>":
            logger.info(f"  {event.key}: {event.cpu_time_total_str} (CPU), {event.cuda_time_total_str if hasattr(event, 'cuda_time_total_str') else 'N/A'} (GPU)")
    
    logger.info("\nMemory usage:")
    for event in events[:10]:
        if hasattr(event, 'cpu_memory_usage') and event.cpu_memory_usage > 0:
            logger.info(f"  {event.key}: {event.cpu_memory_usage / 1024**2:.2f} MB (CPU)")
    
    return timing_stats, trace_file


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Profile training performance")
    parser.add_argument("--data_path", type=str, default="data/test_alignment/train.jsonl",
                       help="Path to training data")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size")
    parser.add_argument("--num_workers", type=int, default=2,
                       help="Number of DataLoader workers")
    parser.add_argument("--num_batches", type=int, default=10,
                       help="Number of batches to profile")
    parser.add_argument("--use_preprocessed", action="store_true",
                       help="Use preprocessed data")
    parser.add_argument("--preprocessed_dir", type=str, default=None,
                       help="Preprocessed data directory")
    
    args = parser.parse_args()
    
    # 创建配置
    config = AlignmentConfig(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=1,  # 只用于初始化，不会真正训练
        use_preprocessed=args.use_preprocessed,
        preprocessed_dir=args.preprocessed_dir
    )
    
    # 运行profiling
    timing_stats, trace_file = profile_training_loop(
        config,
        num_batches=args.num_batches
    )
    
    logger.info(f"\nProfiling complete! Trace file: {trace_file}")


if __name__ == "__main__":
    main()
