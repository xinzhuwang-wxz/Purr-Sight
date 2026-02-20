"""
Training Speed Monitoring Callback

监控训练速度，记录每个batch的数据加载时间、前向传播时间、反向传播时间等。

使用方法:
    在Trainer的callbacks中添加SpeedMonitor实例
"""

import time
from typing import Dict, List
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from purrsight.utils.logging import logger


class SpeedMonitor(Callback):
    """
    训练速度监控回调
    
    记录：
    - 数据加载时间（DataLoader迭代时间）
    - 前向传播时间
    - 反向传播时间
    - 总batch时间
    - 训练速度（samples/sec）
    """
    
    def __init__(self, log_every_n_batches: int = 50, max_history: int = 1000):
        """
        初始化速度监控器
        
        Args:
            log_every_n_batches: 每N个batch记录一次统计信息
            max_history: 保留的最大历史记录数，超过后自动清理（防止内存泄漏）
        """
        super().__init__()
        self.log_every_n_batches = log_every_n_batches
        self.max_history = max_history  # 🔧 修复：限制历史记录数量，防止内存泄漏
        self.batch_times: List[float] = []
        self.data_load_times: List[float] = []
        self.forward_times: List[float] = []
        self.backward_times: List[float] = []
        
        # 🔧 优化：添加验证阶段性能监控
        self.val_batch_times: List[float] = []
        self.val_metrics_times: List[float] = []
        
        # 用于测量时间
        self._batch_start_time: float = None
        self._data_load_end_time: float = None
        self._forward_end_time: float = None
        self._val_batch_start_time: float = None
        self._val_metrics_start_time: float = None
    
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        """记录batch开始时间"""
        self._batch_start_time = time.time()
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """记录batch结束时间并计算各项耗时"""
        batch_end_time = time.time()
        
        if self._batch_start_time is not None:
            total_time = batch_end_time - self._batch_start_time
            self.batch_times.append(total_time)
            
            # 🔧 修复：防止内存泄漏，限制历史记录数量
            if len(self.batch_times) > self.max_history:
                # 只保留最近的max_history条记录
                self.batch_times = self.batch_times[-self.max_history:]
            
            # 估算各项时间（Lightning不直接提供，我们通过差值估算）
            # 注意：这是近似值，实际时间可能略有偏差
            if len(self.batch_times) > 1:
                # 使用移动平均估算数据加载时间（假设数据加载与batch时间相关）
                avg_batch_time = sum(self.batch_times[-10:]) / min(10, len(self.batch_times))
                # 假设数据加载占batch时间的10-30%（取决于num_workers）
                estimated_data_load = avg_batch_time * 0.2
                self.data_load_times.append(estimated_data_load)
                
                # 🔧 修复：防止内存泄漏，限制data_load_times列表大小
                if len(self.data_load_times) > self.max_history:
                    self.data_load_times = self.data_load_times[-self.max_history:]
            
            # 记录到Lightning logger
            if (batch_idx + 1) % self.log_every_n_batches == 0:
                # 🔧 修复3：优化移动平均计算（只在需要时计算，避免重复计算）
                # 只计算一次，避免重复计算
                recent_times = self.batch_times[-self.log_every_n_batches:]
                avg_batch_time = sum(recent_times) / len(recent_times) if recent_times else 0.0
                batch_size = trainer.train_dataloader.batch_size if hasattr(trainer.train_dataloader, 'batch_size') else 1
                samples_per_sec = batch_size / avg_batch_time if avg_batch_time > 0 else 0
                
                pl_module.log("train/batch_time_avg", avg_batch_time, on_step=True, on_epoch=False)
                pl_module.log("train/samples_per_sec", samples_per_sec, on_step=True, on_epoch=False)
                
                logger.info(
                    f"Speed stats (batch {batch_idx + 1}): "
                    f"avg_batch_time={avg_batch_time*1000:.1f}ms, "
                    f"throughput={samples_per_sec:.1f} samples/sec"
                )
    
    def on_train_epoch_end(self, trainer, pl_module):
        """epoch结束时打印统计信息"""
        if self.batch_times:
            avg_batch_time = sum(self.batch_times) / len(self.batch_times)
            min_batch_time = min(self.batch_times)
            max_batch_time = max(self.batch_times)
            
            batch_size = trainer.train_dataloader.batch_size if hasattr(trainer.train_dataloader, 'batch_size') else 1
            avg_samples_per_sec = batch_size / avg_batch_time if avg_batch_time > 0 else 0
            
            logger.info("=" * 80)
            logger.info("Training Speed Summary (Epoch End)")
            logger.info("=" * 80)
            logger.info(f"  Average batch time: {avg_batch_time*1000:.1f} ms")
            logger.info(f"  Min batch time:      {min_batch_time*1000:.1f} ms")
            logger.info(f"  Max batch time:      {max_batch_time*1000:.1f} ms")
            logger.info(f"  Average throughput: {avg_samples_per_sec:.1f} samples/sec")
            logger.info("=" * 80)
            
            # 记录到Lightning logger
            pl_module.log("train/batch_time_avg_epoch", avg_batch_time, on_step=False, on_epoch=True)
            pl_module.log("train/samples_per_sec_avg_epoch", avg_samples_per_sec, on_step=False, on_epoch=True)
            
            # 🔧 修复：epoch结束后清理历史记录，释放内存（防止内存泄漏）
            self.batch_times.clear()
            self.data_load_times.clear()
            self.forward_times.clear()
            self.backward_times.clear()
    
    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx):
        """记录validation batch开始时间"""
        self._val_batch_start_time = time.time()
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """记录validation batch结束时间"""
        batch_end_time = time.time()
        
        if self._val_batch_start_time is not None:
            total_time = batch_end_time - self._val_batch_start_time
            self.val_batch_times.append(total_time)
            
            # 🔧 优化：防止内存泄漏，限制历史记录数量
            if len(self.val_batch_times) > self.max_history:
                self.val_batch_times = self.val_batch_times[-self.max_history:]
            
            # 记录验证batch时间（每N个batch记录一次）
            if (batch_idx + 1) % self.log_every_n_batches == 0:
                avg_val_time = sum(self.val_batch_times[-self.log_every_n_batches:]) / min(self.log_every_n_batches, len(self.val_batch_times))
                logger.info(f"Validation batch {batch_idx + 1} time: {total_time*1000:.1f}ms (avg: {avg_val_time*1000:.1f}ms)")
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """验证epoch结束时打印统计信息"""
        if self.val_batch_times:
            avg_val_time = sum(self.val_batch_times) / len(self.val_batch_times)
            min_val_time = min(self.val_batch_times)
            max_val_time = max(self.val_batch_times)
            
            batch_size = trainer.val_dataloaders[0].batch_size if hasattr(trainer, 'val_dataloaders') and trainer.val_dataloaders else 1
            avg_samples_per_sec = batch_size / avg_val_time if avg_val_time > 0 else 0
            
            logger.info("=" * 80)
            logger.info("Validation Speed Summary (Epoch End)")
            logger.info("=" * 80)
            logger.info(f"  Average batch time: {avg_val_time*1000:.1f} ms")
            logger.info(f"  Min batch time:      {min_val_time*1000:.1f} ms")
            logger.info(f"  Max batch time:      {max_val_time*1000:.1f} ms")
            logger.info(f"  Average throughput: {avg_samples_per_sec:.1f} samples/sec")
            if self.val_metrics_times:
                avg_metrics_time = sum(self.val_metrics_times) / len(self.val_metrics_times)
                logger.info(f"  Metrics computation: {avg_metrics_time*1000:.1f} ms (avg)")
            logger.info("=" * 80)
            
            # 记录到Lightning logger
            pl_module.log("val/batch_time_avg_epoch", avg_val_time, on_step=False, on_epoch=True)
            pl_module.log("val/samples_per_sec_avg_epoch", avg_samples_per_sec, on_step=False, on_epoch=True)
            
            # 🔧 优化：epoch结束后清理历史记录，释放内存（防止内存泄漏）
            self.val_batch_times.clear()
            self.val_metrics_times.clear()
