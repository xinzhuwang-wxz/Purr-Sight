"""Phase 2 Training Entry Point.

This module handles the setup and execution of the instruction tuning phase (Phase 2)
for the Purr-Sight multimodal LLM. It manages MLflow tracking, data loading,
model initialization, and the training loop using PyTorch Lightning.
"""

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.utils.data import DataLoader
import torch
from pathlib import Path
import os
import mlflow
from datetime import datetime

from .train_llm_conf import LLMConfig
from .dataset import InstructionDataset, collate_fn
from .lightning_module import LLMTuningModule
from purrsight.utils.logging import logger, MLflowLogger
from purrsight.config import CHECKPOINT_MONITOR, CHECKPOINT_MONITOR_MODE

def train_llm(config: LLMConfig):
    logger.info("Starting Phase 2: Instruction Tuning")
    
    # Setup MLflow tracking.
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    try:
        experiment = mlflow.get_experiment_by_name(config.experiment_name)
        if experiment is None:
            mlflow.create_experiment(
                config.experiment_name,
                tags={"description": f"Instruction Tuning Phase 2: {config.experiment_name}"}
            )
            logger.info(f"Created new MLflow experiment: {config.experiment_name}")
        mlflow.set_experiment(config.experiment_name)
    except Exception as e:
        logger.warning(f"Failed to setup MLflow experiment: {e}")

    # Generate unique run name based on timestamp.
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{config.experiment_name}_{run_timestamp}"

    # Start MLflow Run context（同一 run 内记录 params 与 metrics，保证 UI 中曲线可见）
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(config.__dict__)

        # Initialize Lightning Module and load LLM.
        model_module = LLMTuningModule(config)
        
        # Initialize dataset.
        # Use the tokenizer from the loaded model.
        tokenizer = model_module.model.tokenizer
        
        logger.info(f"Loading dataset from {config.data_path}")
        train_dataset = InstructionDataset(
            data_path=config.data_path, 
            tokenizer=tokenizer, 
            max_length=config.max_length
        )
        
        if len(train_dataset) == 0:
            logger.error("Dataset is empty! Aborting training.")
            return
            
        logger.info(f"Dataset size: {len(train_dataset)}")
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )
        
        # 4. Callbacks：统一用 MLflow 记录指标，checkpoint 仅按 CHECKPOINT_MONITOR 保存
        checkpoint_dir = Path(config.output_dir) / f"{run_name}" / "checkpoints"
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="{epoch}-{step}-{" + CHECKPOINT_MONITOR + ":.2f}",
            save_top_k=3,
            monitor=CHECKPOINT_MONITOR,
            mode=CHECKPOINT_MONITOR_MODE,
            save_last=True,
        )
        
        lr_monitor = LearningRateMonitor(logging_interval='step')

        # 使用当前 run_id，使 Lightning 的指标写入同一 run
        run_id_for_logger = None
        try:
            ar = mlflow.active_run()
            run_id_for_logger = ar.info.run_id if ar else None
        except Exception:
            pass
        try:
            if run_id_for_logger:
                mlf_logger = MLflowLogger(
                    experiment_name=config.experiment_name,
                    tracking_uri=config.mlflow_tracking_uri,
                    run_name=run_name,
                    run_id=run_id_for_logger,
                )
            else:
                mlf_logger = MLflowLogger(
                    experiment_name=config.experiment_name,
                    tracking_uri=config.mlflow_tracking_uri,
                    run_name=run_name,
                )
        except TypeError:
            mlf_logger = MLflowLogger(
                experiment_name=config.experiment_name,
                tracking_uri=config.mlflow_tracking_uri,
                run_name=run_name,
            )

        # Initialize Trainer.
        # Configure accelerator and devices.
        accelerator = "auto"
        devices = 1 # Default
        
        if config.device == "cuda": 
            accelerator = "gpu"
            if torch.cuda.is_available():
                devices = torch.cuda.device_count() # Use all GPUs by default if cuda
        elif config.device == "mps": 
            accelerator = "mps"
            devices = 1
        elif config.device == "cpu": 
            accelerator = "cpu"
            devices = "auto"
        
        logger.info(f"Initializing Trainer (Accelerator: {accelerator}, Devices: {devices})")
        
        # DDP Strategy for multi-gpu
        strategy = "auto"
        if accelerator == "gpu" and devices > 1:
            strategy = "ddp_find_unused_parameters_true"
            
        trainer = pl.Trainer(
            default_root_dir=str(Path(config.output_dir) / config.experiment_name),
            max_epochs=config.epochs,
            accelerator=accelerator,
            devices=devices, 
            strategy=strategy,
            accumulate_grad_batches=config.gradient_accumulation_steps,
            callbacks=[checkpoint_callback, lr_monitor],
            logger=mlf_logger,  # Use MLflow Logger
            log_every_n_steps=config.log_every,
            precision="16-mixed" if accelerator != "cpu" else 32 # Mixed precision for GPU
        )
        
        # Start training loop.
        logger.info("Starting training loop...")
        trainer.fit(model_module, train_loader)
        
        logger.info(f"Training finished. Checkpoints saved to {checkpoint_dir}")
        
        # Log Final Artifacts.
        mlflow.log_param("final_checkpoint_dir", str(checkpoint_dir))
