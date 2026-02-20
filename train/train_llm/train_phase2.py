#!/usr/bin/env python3
"""
Main Training Script for Phase 2: Multi-Modal LLM Training

This script integrates all Phase 2 components to create a complete training pipeline:
- Loads Phase 1 checkpoint and freezes aligner parameters
- Initializes trainable projector modules
- Applies LoRA to LLM for parameter-efficient fine-tuning
- Sets up PyTorch Lightning Trainer with MLflow logging
- Handles training execution with proper error handling
- Supports resuming from checkpoints

Usage:
    python train.py --config config/train_config.yaml
    python train.py --config config/train_config.yaml --resume
    python train.py --config config/train_config.yaml --resume --checkpoint path/to/checkpoint.ckpt
"""

import os
import sys
import argparse
import signal
import traceback
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import MLFlowLogger
from torch.utils.data import DataLoader

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from train.train_llm.train_llm_conf import TrainingConfig, load_config, validate_config
from train.train_llm.multimodal_llm_module import MultiModalLLMModule
from train.train_llm.dataset import InstructionDataset, collate_fn  # Changed to InstructionDataset
from purrsight.LLM.model import PurrSightMMLLM
from purrsight.utils.logging import logger
from purrsight.config import CHECKPOINT_MONITOR, CHECKPOINT_MONITOR_MODE
from transformers import AutoTokenizer

# Try to import mlflow, but make it optional
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None


class TrainingManager:
    """Manages the complete Phase 2 training pipeline.
    
    This class orchestrates all components of Phase 2 training:
    - Configuration loading and validation
    - Model initialization with Phase 1 checkpoint loading
    - Dataset and DataLoader creation
    - PyTorch Lightning Trainer setup
    - Training execution with error handling
    - Checkpoint management and recovery
    """
    
    def __init__(self, config: TrainingConfig, resume_from_checkpoint: Optional[str] = None):
        """Initialize training manager.
        
        Args:
            config: Validated training configuration
            resume_from_checkpoint: Optional path to checkpoint for resuming training
        """
        self.config = config
        self.resume_from_checkpoint = resume_from_checkpoint
        self.model = None
        self.lightning_module = None
        self.trainer = None
        self.train_dataloader = None
        self.val_dataloader = None
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info(f"Initialized TrainingManager with config: {config.mlflow_experiment_name}")
    
    def _signal_handler(self, signum, frame):
        """Handle interrupt signals for graceful shutdown."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        
        if self.trainer and self.trainer.training:
            logger.info("Saving emergency checkpoint...")
            try:
                emergency_path = Path(self.config.checkpoint_dir) / "emergency_checkpoint.ckpt"
                self.trainer.save_checkpoint(emergency_path)
                logger.info(f"Emergency checkpoint saved to: {emergency_path}")
            except Exception as e:
                logger.error(f"Failed to save emergency checkpoint: {e}")
        
        sys.exit(0)
    
    def setup_model(self) -> None:
        """Initialize the complete multi-modal LLM model.
        
        This method:
        1. Creates the PurrSightMMLLM model
        2. Loads Phase 1 checkpoint if provided
        3. Applies LoRA configuration
        4. Wraps in Lightning module
        """
        logger.info("Setting up multi-modal LLM model...")
        
        try:
            # Create LoRA configuration dictionary
            lora_config = {
                'enabled': True,
                'r': self.config.lora_r,
                'lora_alpha': self.config.lora_alpha,
                'lora_dropout': self.config.lora_dropout,
                'target_modules': self.config.lora_target_modules,
                'task_type': 'CAUSAL_LM',
                'inference_mode': False
            }
            
            # Create projector configuration
            projector_config = {
                'hidden_dim': self.config.projector_hidden_dim,
                'num_tokens': 4  # Default number of tokens per modality
            }
            
            # Initialize the complete model
            self.model = PurrSightMMLLM(
                llm_model_path=self.config.llm_model_name,
                aligner_weights_path=self.config.phase1_checkpoint_path,
                freeze_encoders=True,  # Always freeze encoders in Phase 2
                freeze_projector=False,  # Train projectors
                freeze_llm=False,  # LoRA will handle LLM freezing
                lora_config=lora_config,
                projector_config=projector_config
            )
            
            logger.info("Model initialized successfully")
            
            # Wrap in Lightning module
            self.lightning_module = MultiModalLLMModule(
                model=self.model,
                learning_rate=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                warmup_steps=self.config.warmup_steps,
                gradient_clip_val=self.config.gradient_clip_val,
                max_epochs=self.config.num_epochs,
                log_every_n_steps=self.config.log_every_n_steps,
                phase1_checkpoint_path=self.config.phase1_checkpoint_path,
                lora_config=lora_config
            )
            
            logger.info("Lightning module created successfully")
            
        except Exception as e:
            error_msg = f"Failed to setup model: {str(e)}"
            logger.error(error_msg)
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(error_msg) from e
    
    def setup_data(self) -> None:
        """Setup datasets and data loaders.
        
        Creates training and validation datasets with proper preprocessing
        and error handling for missing or corrupted files.
        """
        logger.info("Setting up datasets and data loaders...")
        
        try:
            # Initialize tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.llm_model_name,
                trust_remote_code=True,
                padding_side="right"
            )
            
            # Set pad token if not present
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                logger.info("Set pad_token to eos_token")
            
            # Create training dataset
            train_data_path = Path(self.config.data_dir) / "train.jsonl"
            if not train_data_path.exists():
                raise FileNotFoundError(f"Training data not found: {train_data_path}")
            
            train_dataset = InstructionDataset(
                data_path=str(train_data_path),
                tokenizer=tokenizer,
                max_length=self.config.max_text_length
            )
            
            logger.info(f"Training dataset created with {len(train_dataset)} samples")
            
            # Create training dataloader
            self.train_dataloader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=self.config.num_workers,
                collate_fn=collate_fn,
                pin_memory=True,
                drop_last=True  # Ensure consistent batch sizes
            )
            
            # Create validation dataset if validation data exists
            val_data_path = Path(self.config.data_dir) / "val.jsonl"
            
            if val_data_path.exists():
                try:
                    val_dataset = InstructionDataset(
                        data_path=str(val_data_path),
                        tokenizer=tokenizer,
                        max_length=self.config.max_text_length
                    )
                    
                    self.val_dataloader = DataLoader(
                        val_dataset,
                        batch_size=self.config.batch_size,
                        shuffle=False,
                        num_workers=self.config.num_workers,
                        collate_fn=collate_fn,
                        pin_memory=True
                    )
                    
                    logger.info(f"Validation dataset created with {len(val_dataset)} samples")
                    
                except Exception as e:
                    logger.warning(f"Failed to create validation dataset: {e}")
                    self.val_dataloader = None
            else:
                logger.info("No validation data found, skipping validation dataset")
                self.val_dataloader = None
            
        except Exception as e:
            error_msg = f"Failed to setup data: {str(e)}"
            logger.error(error_msg)
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(error_msg) from e
    
    def setup_trainer(self) -> None:
        """Setup PyTorch Lightning trainer with callbacks and loggers.
        
        Configures:
        - MLflow logger for experiment tracking (Phase 1 style)
        - Model checkpointing (only last and best)
        - Learning rate monitoring
        - Gradient clipping and mixed precision
        """
        logger.info("Setting up PyTorch Lightning trainer...")
        
        try:
            # Setup MLflow logger FIRST to get the run_id (Phase 1 style)
            mlflow_logger = None
            run_id = None
            
            if MLFLOW_AVAILABLE:
                try:
                    # Set MLflow tracking URI if provided
                    if self.config.mlflow_tracking_uri:
                        mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
                    
                    # Start MLflow run (let MLflow generate the run_id)
                    mlflow.set_experiment(self.config.mlflow_experiment_name)
                    from datetime import datetime
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    run_name = f"phase2_{timestamp}"
                    mlflow.start_run(run_name=run_name)
                    
                    # Get the MLflow-generated run_id
                    mlflow_run = mlflow.active_run()
                    if mlflow_run:
                        run_id = mlflow_run.info.run_id
                        self.mlflow_run_id = run_id
                        
                        logger.info(f"MLflow run started with ID: {run_id}")
                    else:
                        raise RuntimeError("Failed to start MLflow run")
                    
                except Exception as e:
                    logger.warning(f"Failed to initialize MLflow: {e}")
                    # Fall back to UUID if MLflow fails
                    import uuid
                    run_id = uuid.uuid4().hex
                    self.mlflow_run_id = None
            else:
                logger.warning("MLflow not available")
                # Fall back to UUID if MLflow not available
                import uuid
                run_id = uuid.uuid4().hex
                self.mlflow_run_id = None
            
            # Create checkpoint directory with run_id_timestamp (Phase 1 style)
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            checkpoint_dir = Path(self.config.checkpoint_dir) / f"{run_id}_{timestamp}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.checkpoint_dir = checkpoint_dir
            self.run_id = f"{run_id}_{timestamp}"  # Full directory name
            
            logger.info(f"Run ID: {run_id}")
            logger.info(f"Timestamp: {timestamp}")
            logger.info(f"Checkpoints will be saved to: {checkpoint_dir}")
            
            # Continue with MLflow logger setup if available
            if MLFLOW_AVAILABLE and self.mlflow_run_id:
                try:
                    # Create MLflow logger with the active run
                    mlflow_logger = MLFlowLogger(
                        experiment_name=self.config.mlflow_experiment_name,
                        tracking_uri=self.config.mlflow_tracking_uri,
                        run_id=self.mlflow_run_id
                    )
                    
                    # Log configuration parameters
                    mlflow_logger.log_hyperparams(self.config.to_dict())
                    
                    # Log tags
                    mlflow.set_tag("phase", "phase2")
                    mlflow.set_tag("checkpoint_dir", str(checkpoint_dir))
                    mlflow.set_tag("timestamp", timestamp)
                    
                    logger.info(f"MLflow logger initialized for experiment: {self.config.mlflow_experiment_name}")
                    
                    # Save MLflow run_id to checkpoint directory for easy reference
                    mlflow_info_path = checkpoint_dir / "MLFLOW_RUN_ID.txt"
                    with open(mlflow_info_path, 'w') as f:
                        f.write(f"MLflow Run ID: {self.mlflow_run_id}\n")
                        f.write(f"Checkpoint Dir: {checkpoint_dir}\n")
                        f.write(f"Experiment: {self.config.mlflow_experiment_name}\n")
                        f.write(f"Timestamp: {timestamp}\n")
                        f.write(f"\nTo view in MLflow UI:\n")
                        f.write(f"  mlflow ui\n")
                        f.write(f"  Then navigate to experiment '{self.config.mlflow_experiment_name}'\n")
                        f.write(f"  and find run with ID: {self.mlflow_run_id}\n")
                        f.write(f"\nNote: The checkpoint directory name includes the MLflow run_id:\n")
                        f.write(f"  {checkpoint_dir.name} = {self.mlflow_run_id}_{timestamp}\n")
                    logger.info(f"Saved MLflow run info to: {mlflow_info_path}")
                    
                except Exception as e:
                    logger.warning(f"Failed to setup MLflow logger: {e}")
                    mlflow_logger = None
            
            # Setup checkpoint callback (only last and best)
            # Save to lightning_checkpoints subdirectory
            checkpoint_callback = ModelCheckpoint(
                dirpath=str(checkpoint_dir / "lightning_checkpoints"),
                filename="best-{epoch:02d}-{train_loss:.4f}",
                monitor=CHECKPOINT_MONITOR,
                mode=CHECKPOINT_MONITOR_MODE,
                save_top_k=1,  # Only save best
                save_last=True,  # Also save last
                every_n_epochs=self.config.save_every_n_epochs,
                verbose=True,
            )
            
            # Setup learning rate monitoring
            lr_monitor = LearningRateMonitor(logging_interval='step')
            
            # Determine precision and accelerator
            precision = 16 if torch.cuda.is_available() else 32
            accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
            devices = self.config.num_gpus if torch.cuda.is_available() else 1
            
            # Setup trainer
            trainer_kwargs = {
                'max_epochs': self.config.num_epochs,
                'accelerator': accelerator,
                'devices': devices,
                'precision': precision,
                'gradient_clip_val': self.config.gradient_clip_val,
                'accumulate_grad_batches': 1,  # Can be configured if needed
                'log_every_n_steps': self.config.log_every_n_steps,
                'callbacks': [checkpoint_callback, lr_monitor],
                'enable_checkpointing': True,
                'enable_progress_bar': True,
                'enable_model_summary': True,
                'deterministic': False,  # Set to True for reproducibility (slower)
                'benchmark': True,  # Optimize for consistent input sizes
            }
            
            # Add logger if available
            if mlflow_logger:
                trainer_kwargs['logger'] = mlflow_logger
            
            # Add distributed training strategy if multiple GPUs
            if devices > 1:
                trainer_kwargs['strategy'] = self.config.distributed_backend
                logger.info(f"Using distributed training with {devices} GPUs and {self.config.distributed_backend} backend")
            
            # Note: resume checkpoint is passed to trainer.fit(), not Trainer.__init__()
            if self.resume_from_checkpoint:
                logger.info(f"Will resume training from: {self.resume_from_checkpoint}")
            
            self.trainer = pl.Trainer(**trainer_kwargs)
            
            logger.info(f"Trainer configured: {accelerator}={devices}, precision={precision}")
            
        except Exception as e:
            error_msg = f"Failed to setup trainer: {str(e)}"
            logger.error(error_msg)
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(error_msg) from e
    
    def run_training(self) -> None:
        """Execute the complete training pipeline.
        
        Runs training with proper error handling and logging.
        Handles validation if validation data is available.
        """
        logger.info("Starting Phase 2 training...")
        
        try:
            # Log training start
            logger.info(
                f"Training configuration:\n"
                f"  Model: {self.config.llm_model_name}\n"
                f"  Phase 1 checkpoint: {self.config.phase1_checkpoint_path}\n"
                f"  Epochs: {self.config.num_epochs}\n"
                f"  Batch size: {self.config.batch_size}\n"
                f"  Learning rate: {self.config.learning_rate}\n"
                f"  Training samples: {len(self.train_dataloader.dataset)}\n"
                f"  Validation samples: {len(self.val_dataloader.dataset) if self.val_dataloader else 0}"
            )
            
            # Run training
            self.trainer.fit(
                model=self.lightning_module,
                train_dataloaders=self.train_dataloader,
                val_dataloaders=self.val_dataloader,
                ckpt_path=self.resume_from_checkpoint
            )
            
            logger.info("Training completed successfully!")
            
            # Log final metrics
            if self.trainer.callback_metrics:
                final_metrics = {k: v.item() if hasattr(v, 'item') else v 
                               for k, v in self.trainer.callback_metrics.items()}
                logger.info(f"Final metrics: {final_metrics}")
            
            # Save artifacts to MLflow (Phase 1 style)
            self._save_artifacts_to_mlflow()
            
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            raise
        except Exception as e:
            error_msg = f"Training failed: {str(e)}"
            logger.error(error_msg)
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(error_msg) from e
    
    def _save_artifacts_to_mlflow(self) -> None:
        """
        Save training artifacts to MLflow (Phase 1 style).
        
        Saves:
        1. Model weights (model.pt) - for deployment
        2. Configuration file (config.json) - training config and metadata
        3. Training visualizations (training curves, etc.)
        
        Note: Lightning checkpoints are saved locally in lightning_checkpoints/
        """
        if not MLFLOW_AVAILABLE:
            logger.warning("MLflow not available, skipping artifacts saving")
            return
        
        active_run = mlflow.active_run()
        if active_run is None:
            logger.warning("No active MLflow run, skipping artifacts saving")
            return
        
        if not hasattr(self, 'checkpoint_dir'):
            logger.warning("checkpoint_dir not set, skipping artifacts saving")
            return
        
        logger.info("Saving artifacts to MLflow...")
        
        try:
            import tempfile
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            # 1. Save model weights (model.pt for deployment)
            model_path = self.checkpoint_dir / "model.pt"
            try:
                # Save the complete model state
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'config': self.config.to_dict(),
                    'run_id': self.run_id
                }, model_path)
                mlflow.log_artifact(str(model_path), artifact_path="model")
                logger.info(f"  ✓ Saved model.pt to artifacts/model/")
            except Exception as e:
                logger.warning(f"Failed to save model.pt: {e}")
            
            # 2. Save configuration file
            config_path = self.checkpoint_dir / "config.json"
            try:
                import json
                with open(config_path, 'w') as f:
                    json.dump(self.config.to_dict(), f, indent=2)
                mlflow.log_artifact(str(config_path), artifact_path="config")
                logger.info(f"  ✓ Saved config.json to artifacts/config/")
            except Exception as e:
                logger.warning(f"Failed to save config.json: {e}")
            
            # 3. Save training visualizations
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmpdir = Path(tmpdir)
                    
                    # Get training history from MLflow
                    try:
                        from mlflow.tracking import MlflowClient
                        client = MlflowClient()
                        run_id = active_run.info.run_id
                        
                        # Get epoch and loss history
                        epoch_history = client.get_metric_history(run_id, "epoch")
                        train_loss_history = client.get_metric_history(run_id, "train_loss_epoch")
                        val_loss_history = client.get_metric_history(run_id, "val_loss")
                        
                        # Build epoch to step mapping
                        epoch_to_last_step = {}
                        for m in epoch_history:
                            if m.value is not None:
                                epoch_num = int(m.value)
                                if epoch_num not in epoch_to_last_step or m.step > epoch_to_last_step[epoch_num]:
                                    epoch_to_last_step[epoch_num] = m.step
                        
                        # Build step to loss mapping
                        train_loss_by_step = {m.step: m.value for m in train_loss_history}
                        val_loss_by_step = {m.step: m.value for m in val_loss_history}
                        
                        # Extract losses by epoch
                        epochs = []
                        train_losses = []
                        val_losses = []
                        
                        for epoch_num in sorted(epoch_to_last_step.keys()):
                            step = epoch_to_last_step[epoch_num]
                            epochs.append(epoch_num + 1)
                            
                            if step in train_loss_by_step:
                                train_losses.append(train_loss_by_step[step])
                            if step in val_loss_by_step:
                                val_losses.append(val_loss_by_step[step])
                        
                        # Plot training curve
                        if epochs and train_losses:
                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
                            if val_losses:
                                ax.plot(epochs[:len(val_losses)], val_losses, 'r-', label='Val Loss', linewidth=2)
                            ax.set_xlabel('Epoch', fontsize=12)
                            ax.set_ylabel('Loss', fontsize=12)
                            ax.set_title('Phase 2 Training Loss Curve', fontsize=14, fontweight='bold')
                            ax.legend(fontsize=11)
                            ax.grid(True, alpha=0.3)
                            ax.set_xlim(left=0)
                            
                            plot_path = tmpdir / "training_curve.png"
                            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                            plt.close()
                            
                            mlflow.log_artifact(str(plot_path), artifact_path="plots")
                            logger.info(f"  ✓ Saved training_curve.png to artifacts/plots/")
                        else:
                            logger.warning("No training metrics found for visualization")
                    
                    except Exception as e:
                        logger.warning(f"Failed to create training plots: {e}")
            
            except Exception as e:
                logger.warning(f"Failed to save training visualizations: {e}")
            
            logger.info("Artifacts saved successfully to MLflow")
            
            # Create README in checkpoint directory explaining the structure
            readme_path = self.checkpoint_dir / "README.md"
            try:
                with open(readme_path, 'w') as f:
                    f.write(f"# Phase 2 Training Run\n\n")
                    f.write(f"## Run Information\n\n")
                    f.write(f"- **MLflow Run ID**: `{self.mlflow_run_id}`\n")
                    f.write(f"- **Timestamp**: `{self.run_id.split('_', 1)[1] if '_' in self.run_id else 'N/A'}`\n")
                    f.write(f"- **Experiment**: `{self.config.mlflow_experiment_name}`\n")
                    f.write(f"- **Checkpoint Directory**: `{self.checkpoint_dir}`\n\n")
                    
                    f.write(f"## Directory Naming Convention\n\n")
                    f.write(f"The checkpoint directory name follows Phase 1 convention:\n")
                    f.write(f"```\n")
                    f.write(f"{self.checkpoint_dir.name} = {{mlflow_run_id}}_{{timestamp}}\n")
                    f.write(f"                     = {self.mlflow_run_id}_{self.run_id.split('_', 1)[1] if '_' in self.run_id else 'N/A'}\n")
                    f.write(f"```\n\n")
                    f.write(f"This ensures the checkpoint directory and MLflow run are easily matched!\n\n")
                    
                    f.write(f"## Directory Structure\n\n")
                    f.write(f"```\n")
                    f.write(f"{self.checkpoint_dir.name}/\n")
                    f.write(f"├── lightning_checkpoints/    # PyTorch Lightning checkpoints\n")
                    f.write(f"│   ├── best-*.ckpt          # Best model checkpoint\n")
                    f.write(f"│   └── last.ckpt            # Last epoch checkpoint\n")
                    f.write(f"├── model.pt                 # Model weights for deployment\n")
                    f.write(f"├── config.json              # Training configuration\n")
                    f.write(f"├── MLFLOW_RUN_ID.txt        # MLflow run information\n")
                    f.write(f"└── README.md                # This file\n")
                    f.write(f"```\n\n")
                    
                    f.write(f"## MLflow Artifacts\n\n")
                    f.write(f"Training artifacts are saved in MLflow:\n\n")
                    f.write(f"```\n")
                    f.write(f"mlruns/{{experiment_id}}/{self.mlflow_run_id}/\n")
                    f.write(f"├── artifacts/\n")
                    f.write(f"│   ├── model/model.pt       # Model weights\n")
                    f.write(f"│   ├── config/config.json   # Configuration\n")
                    f.write(f"│   └── plots/               # Training visualizations\n")
                    f.write(f"│       └── training_curve.png\n")
                    f.write(f"├── metrics/                 # Training metrics\n")
                    f.write(f"├── params/                  # Hyperparameters\n")
                    f.write(f"└── tags/                    # Run tags\n")
                    f.write(f"```\n\n")
                    
                    f.write(f"## Viewing Results\n\n")
                    f.write(f"### MLflow UI\n\n")
                    f.write(f"```bash\n")
                    f.write(f"mlflow ui\n")
                    f.write(f"```\n\n")
                    f.write(f"Then navigate to:\n")
                    f.write(f"- Experiment: `{self.config.mlflow_experiment_name}`\n")
                    f.write(f"- Run ID: `{self.mlflow_run_id}`\n\n")
                    
                    f.write(f"### Training Metrics\n\n")
                    if self.trainer and self.trainer.callback_metrics:
                        f.write(f"Final metrics:\n")
                        for k, v in self.trainer.callback_metrics.items():
                            val = v.item() if hasattr(v, 'item') else v
                            f.write(f"- **{k}**: {val:.4f}\n")
                    
                    f.write(f"\n## Configuration\n\n")
                    f.write(f"- Model: `{self.config.llm_model_name}`\n")
                    f.write(f"- Phase 1 Checkpoint: `{self.config.phase1_checkpoint_path}`\n")
                    f.write(f"- Epochs: {self.config.num_epochs}\n")
                    f.write(f"- Batch Size: {self.config.batch_size}\n")
                    f.write(f"- Learning Rate: {self.config.learning_rate}\n")
                    f.write(f"- LoRA Rank: {self.config.lora_r}\n")
                    f.write(f"- LoRA Alpha: {self.config.lora_alpha}\n")
                
                logger.info(f"Created README at: {readme_path}")
            except Exception as e:
                logger.warning(f"Failed to create README: {e}")
            
        except Exception as e:
            logger.warning(f"Failed to save artifacts: {e}")
        
        finally:
            # End MLflow run
            try:
                mlflow.end_run()
                logger.info("MLflow run ended successfully")
            except Exception as e:
                logger.warning(f"Failed to end MLflow run: {e}")
    
    def run(self) -> None:
        """Run the complete training pipeline.
        
        Orchestrates all setup and training steps with comprehensive error handling.
        """
        try:
            logger.info("=" * 80)
            logger.info("PHASE 2 TRAINING PIPELINE STARTING")
            logger.info("=" * 80)
            
            # Setup all components
            self.setup_model()
            self.setup_data()
            self.setup_trainer()
            
            # Run training
            self.run_training()
            
            logger.info("=" * 80)
            logger.info("PHASE 2 TRAINING PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error("=" * 80)
            logger.error("PHASE 2 TRAINING PIPELINE FAILED")
            logger.error("=" * 80)
            logger.error(f"Error: {str(e)}")
            raise


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Phase 2 Multi-Modal LLM Training Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training
  python train.py --config config/train_config.yaml
  
  # Resume from latest checkpoint
  python train.py --config config/train_config.yaml --resume
  
  # Resume from specific checkpoint
  python train.py --config config/train_config.yaml --resume --checkpoint path/to/checkpoint.ckpt
  
  # Override configuration values
  python train.py --config config/train_config.yaml --batch-size 16 --learning-rate 1e-4
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--config', '-c',
        type=str,
        required=True,
        help='Path to training configuration YAML file'
    )
    
    # Optional arguments
    parser.add_argument(
        '--resume', '-r',
        action='store_true',
        help='Resume training from the latest checkpoint'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='Specific checkpoint path to resume from (overrides --resume)'
    )
    
    # Configuration overrides
    parser.add_argument(
        '--batch-size',
        type=int,
        help='Override batch size from config'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        help='Override learning rate from config'
    )
    
    parser.add_argument(
        '--num-epochs',
        type=int,
        help='Override number of epochs from config'
    )
    
    parser.add_argument(
        '--num-gpus',
        type=int,
        help='Override number of GPUs from config'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        help='Override random seed from config'
    )
    
    return parser.parse_args()


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Find the latest checkpoint in the checkpoint directory.
    
    Args:
        checkpoint_dir: Directory to search for checkpoints
        
    Returns:
        Path to latest checkpoint or None if no checkpoints found
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        return None
    
    # Look for checkpoint files
    checkpoint_patterns = ['*.ckpt', 'last.ckpt', 'phase2-*.ckpt']
    checkpoint_files = []
    
    for pattern in checkpoint_patterns:
        checkpoint_files.extend(checkpoint_dir.glob(pattern))
    
    if not checkpoint_files:
        return None
    
    # Sort by modification time and return the latest
    latest_checkpoint = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
    return str(latest_checkpoint)


def setup_environment(config: TrainingConfig) -> None:
    """Setup training environment.
    
    Args:
        config: Training configuration
    """
    # Set random seeds for reproducibility
    if config.seed is not None:
        pl.seed_everything(config.seed, workers=True)
        logger.info(f"Set random seed to {config.seed}")
    
    # Set environment variables for distributed training
    if config.num_gpus > 1:
        os.environ.setdefault('NCCL_DEBUG', 'INFO')
        os.environ.setdefault('PYTHONUNBUFFERED', '1')
    
    # Create output directories
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # Set device if specified
    if config.device != "auto":
        if config.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
        elif config.device == "mps" and not torch.backends.mps.is_available():
            logger.warning("MPS requested but not available, falling back to CPU")


def main():
    """Main entry point for the training script."""
    try:
        # Parse command-line arguments
        args = parse_arguments()
        
        # Load and validate configuration
        logger.info(f"Loading configuration from: {args.config}")
        config = load_config(args.config)
        
        # Apply command-line overrides
        if args.batch_size is not None:
            config.batch_size = args.batch_size
            logger.info(f"Override batch_size: {args.batch_size}")
        
        if args.learning_rate is not None:
            config.learning_rate = args.learning_rate
            logger.info(f"Override learning_rate: {args.learning_rate}")
        
        if args.num_epochs is not None:
            config.num_epochs = args.num_epochs
            logger.info(f"Override num_epochs: {args.num_epochs}")
        
        if args.num_gpus is not None:
            config.num_gpus = args.num_gpus
            logger.info(f"Override num_gpus: {args.num_gpus}")
        
        if args.seed is not None:
            config.seed = args.seed
            logger.info(f"Override seed: {args.seed}")
        
        # Validate final configuration
        validate_config(config)
        logger.info("Configuration validation passed")
        
        # Setup environment
        setup_environment(config)
        
        # Determine checkpoint for resuming
        resume_checkpoint = None
        if args.checkpoint:
            resume_checkpoint = args.checkpoint
            logger.info(f"Resuming from specified checkpoint: {resume_checkpoint}")
        elif args.resume:
            resume_checkpoint = find_latest_checkpoint(config.checkpoint_dir)
            if resume_checkpoint:
                logger.info(f"Resuming from latest checkpoint: {resume_checkpoint}")
            else:
                logger.warning("No checkpoint found for resuming, starting from scratch")
        
        # Create and run training manager
        training_manager = TrainingManager(
            config=config,
            resume_from_checkpoint=resume_checkpoint
        )
        
        training_manager.run()
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()