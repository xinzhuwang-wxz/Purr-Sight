#!/usr/bin/env python3
"""
简化的 Phase 2 训练脚本 - 用于快速测试

直接在CPU上运行，使用最小配置
"""

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from pathlib import Path

from train.train_llm.dataset import InstructionDataset, collate_fn
from train.train_llm.multimodal_llm_module import MultiModalLLMModule
from purrsight.LLM.model import PurrSightMMLLM
from purrsight.utils.logging import logger

# 配置
CONFIG = {
    'llm_model_path': 'models/Qwen2.5-0.5B-Instruct',
    'phase1_checkpoint': 'checkpoints/alignment/1652a3d3446641e0a8a03a427c171eeb_20260131_025643/aligner.pt',
    'data_path': 'data/phase2/train.jsonl',
    'batch_size': 1,
    'epochs': 2,
    'learning_rate': 5e-5,
    'max_length': 512,
}

def main():
    logger.info("=" * 80)
    logger.info("Phase 2 简化训练脚本")
    logger.info("=" * 80)
    
    # 1. 初始化tokenizer
    logger.info("加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG['llm_model_path'],
        trust_remote_code=True,
        padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 2. 创建dataset
    logger.info("加载数据集...")
    dataset = InstructionDataset(
        data_path=CONFIG['data_path'],
        tokenizer=tokenizer,
        max_length=CONFIG['max_length']
    )
    logger.info(f"数据集大小: {len(dataset)} 样本")
    
    dataloader = DataLoader(
        dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        drop_last=False
    )
    
    # 3. 创建模型
    logger.info("初始化模型...")
    lora_config = {
        'enabled': True,
        'r': 16,
        'lora_alpha': 32,
        'lora_dropout': 0.1,
        'target_modules': ['q_proj', 'v_proj', 'k_proj', 'o_proj'],
        'task_type': 'CAUSAL_LM',
        'inference_mode': False
    }
    
    projector_config = {
        'hidden_dim': 2048,
        'num_tokens': 4
    }
    
    model = PurrSightMMLLM(
        llm_model_path=CONFIG['llm_model_path'],
        aligner_weights_path=CONFIG['phase1_checkpoint'],
        freeze_encoders=True,
        freeze_projector=False,
        freeze_llm=False,
        lora_config=lora_config,
        projector_config=projector_config
    )
    
    # 4. 创建Lightning模块
    logger.info("创建Lightning模块...")
    lightning_module = MultiModalLLMModule(
        model=model,
        learning_rate=CONFIG['learning_rate'],
        projector_lr=CONFIG['learning_rate'] * 5.0,  # 2.5e-4
        lora_lr=CONFIG['learning_rate'] * 0.5,  # 2.5e-5
        weight_decay=0.01,
        warmup_steps=10,
        gradient_clip_val=0.5,
        max_epochs=CONFIG['epochs'],
        log_every_n_steps=1
    )
    
    # 5. 创建trainer
    logger.info("配置Trainer...")
    
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints/phase2',
        filename='phase2-simple-{epoch:02d}-{train_loss:.4f}',
        monitor='train_loss',
        mode='min',
        save_top_k=2,
        save_last=True,
        every_n_epochs=1
    )
    
    mlflow_logger = MLFlowLogger(
        experiment_name="phase2_simple_test",
        tracking_uri="file:./mlruns"
    )
    
    trainer = pl.Trainer(
        max_epochs=CONFIG['epochs'],
        accelerator='cpu',
        devices=1,
        precision=32,
        gradient_clip_val=0.5,
        log_every_n_steps=1,
        callbacks=[checkpoint_callback],
        logger=mlflow_logger,
        enable_progress_bar=True,
        enable_model_summary=True
    )
    
    # 6. 开始训练
    logger.info("开始训练...")
    logger.info(f"配置:")
    logger.info(f"  - 样本数: {len(dataset)}")
    logger.info(f"  - Batch size: {CONFIG['batch_size']}")
    logger.info(f"  - Epochs: {CONFIG['epochs']}")
    logger.info(f"  - Learning rate: {CONFIG['learning_rate']}")
    logger.info(f"  - Projector LR: {CONFIG['learning_rate'] * 5.0}")
    logger.info(f"  - LoRA LR: {CONFIG['learning_rate'] * 0.5}")
    
    trainer.fit(
        model=lightning_module,
        train_dataloaders=dataloader
    )
    
    logger.info("=" * 80)
    logger.info("训练完成！")
    logger.info(f"Checkpoint保存在: checkpoints/phase2/")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()
