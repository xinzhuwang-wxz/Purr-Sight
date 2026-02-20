"""MultiModal LLM Lightning Module for Phase 2 Training.

This module implements the complete training pipeline for Phase 2, integrating:
- Phase 1 checkpoint loading with parameter freezing
- LoRA application for efficient LLM fine-tuning
- Projector modules for multi-modal feature transformation
- Complete forward pass through encoders → aligner → projectors → LLM
- Training step with loss computation and logging
- Optimizer configuration with AdamW and cosine decay scheduling
- Mixed precision and gradient clipping support
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, Any, Optional, Tuple
from transformers import get_cosine_schedule_with_warmup
import math

# Try to import mlflow, but make it optional
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None

from purrsight.utils.logging import logger
from purrsight.config import FeatureKey  # Add FeatureKey import
from .checkpoint_loader import CheckpointLoader
from .lora_manager import LoRAManager
from purrsight.LLM.model import PurrSightMMLLM


class MultiModalLLMModule(pl.LightningModule):
    """Lightning module for Phase 2 multi-modal LLM training.
    
    This module orchestrates the complete Phase 2 training process:
    1. Loads Phase 1 checkpoint and freezes aligner parameters
    2. Initializes trainable projector modules
    3. Applies LoRA to LLM for parameter-efficient fine-tuning
    4. Implements forward pass through complete multi-modal pipeline
    5. Computes language modeling loss and logs training metrics
    6. Configures optimizers with warmup and cosine decay scheduling
    
    The module integrates with existing components:
    - CheckpointLoader for Phase 1 weight loading
    - LoRAManager for LoRA configuration and application
    - PurrSightMMLLM for the complete model architecture
    - ModalityProjector for feature transformation
    
    Attributes:
        model: Complete multi-modal LLM model
        learning_rate: Peak learning rate for training
        weight_decay: Weight decay coefficient for regularization
        warmup_steps: Number of warmup steps for learning rate scheduling
        gradient_clip_val: Maximum gradient norm for clipping
        save_hyperparameters: Whether to save hyperparameters to checkpoints
    """
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 2e-4,
        projector_lr: Optional[float] = None,  # 新增：投影头学习率
        lora_lr: Optional[float] = None,  # 新增：LoRA学习率
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        gradient_clip_val: float = 1.0,
        max_epochs: int = 10,
        log_every_n_steps: int = 10,
        phase1_checkpoint_path: Optional[str] = None,
        lora_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Initialize Lightning module.
        
        Args:
            model: Complete multi-modal LLM model (PurrSightMMLLM)
            learning_rate: Base learning rate (default: 2e-4) - used if projector_lr/lora_lr not specified
            projector_lr: Learning rate for projector (default: 5x learning_rate for faster convergence)
            lora_lr: Learning rate for LoRA adapters (default: 0.5x learning_rate for stability)
            weight_decay: Weight decay coefficient (default: 0.01)
            warmup_steps: Number of warmup steps (default: 500)
            gradient_clip_val: Maximum gradient norm for clipping (default: 1.0)
            max_epochs: Maximum number of training epochs (default: 10)
            log_every_n_steps: Logging frequency in steps (default: 10)
            phase1_checkpoint_path: Path to Phase 1 checkpoint (optional)
            lora_config: LoRA configuration dictionary (optional)
            **kwargs: Additional arguments passed to parent class
        """
        super().__init__()
        
        # Save hyperparameters for checkpointing and reproducibility
        self.save_hyperparameters(ignore=['model'])
        
        # Store configuration
        self.model = model
        self.learning_rate = learning_rate
        
        # Set learning rates with smart defaults
        # Projector: 从头训练，需要较大学习率 (5x base)
        # LoRA: 微调预训练模型，需要较小学习率 (0.5x base)
        self.projector_lr = projector_lr if projector_lr is not None else learning_rate * 5.0
        self.lora_lr = lora_lr if lora_lr is not None else learning_rate * 0.5
        
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.gradient_clip_val = gradient_clip_val
        self.max_epochs = max_epochs
        self.log_every_n_steps = log_every_n_steps
        
        # Initialize components if checkpoint path provided
        if phase1_checkpoint_path:
            self._load_phase1_checkpoint(phase1_checkpoint_path)
        
        # Apply LoRA if configuration provided
        if lora_config:
            self._apply_lora(lora_config)
        
        # Log model architecture info
        self._log_model_info()
        
        logger.info(
            f"Initialized MultiModalLLMModule with:\n"
            f"  Base LR: {learning_rate}\n"
            f"  Projector LR: {self.projector_lr} (5x base for faster convergence)\n"
            f"  LoRA LR: {self.lora_lr} (0.5x base for stability)\n"
            f"  Weight Decay: {weight_decay}\n"
            f"  Warmup Steps: {warmup_steps}"
        )
    
    def _load_phase1_checkpoint(self, checkpoint_path: str) -> None:
        """Load Phase 1 checkpoint and freeze aligner parameters.
        
        Args:
            checkpoint_path: Path to Phase 1 checkpoint file
        """
        try:
            # Load checkpoint into model
            metadata = CheckpointLoader.load_phase1_checkpoint(
                checkpoint_path=checkpoint_path,
                model=self.model,
                strict=False  # Allow missing keys for flexibility
            )
            
            # Freeze aligner parameters
            CheckpointLoader.freeze_aligner_parameters(self.model)
            
            logger.info(f"Successfully loaded Phase 1 checkpoint: {metadata}")
            
        except Exception as e:
            logger.error(f"Failed to load Phase 1 checkpoint: {e}")
            raise
    
    def _apply_lora(self, lora_config: Dict[str, Any]) -> None:
        """Apply LoRA configuration to the LLM.
        
        Args:
            lora_config: LoRA configuration dictionary
        """
        try:
            # Apply LoRA to the LLM component
            if hasattr(self.model, 'llm'):
                self.model.llm = LoRAManager.apply_lora(
                    model=self.model.llm,
                    lora_config=lora_config
                )
                
                # Verify trainable parameters
                param_info = LoRAManager.verify_trainable_parameters(self.model)
                logger.info(f"LoRA applied successfully: {param_info}")
            else:
                logger.warning("Model does not have 'llm' attribute, skipping LoRA application")
                
        except Exception as e:
            logger.error(f"Failed to apply LoRA: {e}")
            raise
    
    def _log_model_info(self) -> None:
        """Log model architecture and parameter information."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        trainable_percentage = (trainable_params / total_params * 100) if total_params > 0 else 0
        
        logger.info(
            f"Model Parameter Summary:\n"
            f"  Total: {total_params:,}\n"
            f"  Trainable: {trainable_params:,} ({trainable_percentage:.2f}%)\n"
            f"  Frozen: {frozen_params:,}"
        )
        
        # Log to MLflow if available
        try:
            if MLFLOW_AVAILABLE and mlflow.active_run():
                mlflow.log_param("total_parameters", total_params)
                mlflow.log_param("trainable_parameters", trainable_params)
                mlflow.log_param("trainable_percentage", trainable_percentage)
        except Exception as e:
            logger.warning(f"Failed to log model info to MLflow: {e}")
    
    def forward(
        self,
        image: torch.Tensor,
        audio: torch.Tensor,
        text_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Forward pass through complete model.
        
        Processes multi-modal inputs through the complete pipeline:
        1. Image/audio encoding through frozen encoders
        2. Feature alignment through frozen aligner
        3. Feature projection through trainable projectors
        4. LLM processing with LoRA fine-tuning
        
        Args:
            image: Image tensor (batch, channels, height, width) or (batch, frames, channels, height, width)
            audio: Audio tensor (batch, time, features)
            text_tokens: Text token IDs (batch, seq_len)
            attention_mask: Attention mask (batch, seq_len)
            **kwargs: Additional arguments passed to model
            
        Returns:
            Output logits (batch, seq_len, vocab_size)
        """
        # Prepare inputs dictionary for the model
        inputs = {
            'image': image,
            'audio': audio,
            'input_ids': text_tokens,
            'attention_mask': attention_mask
        }
        
        # Add any additional inputs from kwargs
        inputs.update(kwargs)
        
        # Forward pass through complete model
        outputs = self.model(inputs)
        
        return outputs.logits if hasattr(outputs, 'logits') else outputs
    
    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        """Execute single training step.
        
        Performs forward pass, computes loss, and logs metrics.
        
        Args:
            batch: Dictionary containing all input tensors and labels:
                - image: Image tensor
                - audio: Audio tensor  
                - input_ids: Text token IDs
                - attention_mask: Attention mask
                - labels: Target labels for language modeling
            batch_idx: Index of current batch
            
        Returns:
            Loss tensor for backpropagation
        """
        # Prepare inputs - InstructionDataset returns 'input_ids', not 'text_tokens'
        inputs = {
            'image': batch.get('image', batch.get(FeatureKey.IMAGE, None)),
            'audio': batch.get('audio', batch.get(FeatureKey.AUDIO, None)),
            'input_ids': batch.get('input_ids', batch.get('text_tokens', None)),
            'attention_mask': batch.get('attention_mask', None)
        }
        
        # Forward pass - pass labels separately, not in inputs dict
        labels = batch.get('labels', None)
        
        # Debug: Check input validity before forward pass
        if inputs['input_ids'] is not None and torch.isnan(inputs['input_ids'].float()).any():
            logger.error("NaN detected in input_ids!")
        if inputs['image'] is not None and torch.isnan(inputs['image']).any():
            logger.error(f"NaN detected in image! Shape: {inputs['image'].shape}")
        if inputs['audio'] is not None and torch.isnan(inputs['audio']).any():
            logger.error(f"NaN detected in audio! Shape: {inputs['audio'].shape}")
        
        try:
            outputs = self.model(inputs, labels=labels)
        except Exception as e:
            logger.error(f"Forward pass failed: {e}")
            logger.error(f"Input shapes - image: {inputs['image'].shape if inputs['image'] is not None else None}, "
                        f"audio: {inputs['audio'].shape if inputs['audio'] is not None else None}, "
                        f"input_ids: {inputs['input_ids'].shape if inputs['input_ids'] is not None else None}")
            raise
        
        # Extract loss
        if hasattr(outputs, 'loss') and outputs.loss is not None:
            loss = outputs.loss
        else:
            # Fallback: compute cross-entropy loss manually
            if labels is not None:
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100
                )
            else:
                raise ValueError("No loss available and no labels provided for loss computation")
        
        # Validate loss is finite
        if not torch.isfinite(loss):
            logger.error(f"Non-finite loss detected: {loss}")
            raise ValueError(f"Loss is not finite: {loss}")
        
        # 仅通过 Lightning 的 MLflowLogger 记录（统一用 MLflow）
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("learning_rate", self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True, logger=True)
        if self.trainer.global_step % self.log_every_n_steps == 0:
            grad_norm = self._compute_gradient_norm()
            if grad_norm is not None:
                self.log("grad_norm", grad_norm, on_step=True, logger=True)
        return loss
    
    def validation_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        """Execute single validation step.
        
        Args:
            batch: Dictionary containing validation inputs and labels
            batch_idx: Index of current batch
            
        Returns:
            Validation loss tensor
        """
        # Prepare inputs (same as training step)
        inputs = {
            'image': batch['image'],
            'audio': batch['audio'],
            'input_ids': batch['input_ids'],
            'attention_mask': batch['attention_mask']
        }
        
        # Forward pass - pass labels separately, not in inputs dict
        labels = batch.get('labels', None)
        outputs = self.model(inputs, labels=labels)
        
        # Extract loss
        if hasattr(outputs, 'loss') and outputs.loss is not None:
            loss = outputs.loss
        else:
            if labels is not None:
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100
                )
            else:
                # For validation, we might not always have labels
                loss = torch.tensor(0.0, device=self.device)
        
        # 仅通过 Lightning 的 MLflowLogger 记录（统一用 MLflow）
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler with parameter groups.
        
        Uses AdamW optimizer with different learning rates for different components:
        - Projector: Higher LR (5x base) for faster convergence from scratch
        - LoRA: Lower LR (0.5x base) for stable fine-tuning
        
        Returns:
            Dictionary containing optimizer and scheduler configuration
        """
        # Separate parameters into groups based on their role
        projector_params = []
        lora_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            # Identify parameter group
            if 'projector' in name.lower() or 'projection' in name.lower():
                projector_params.append(param)
            elif 'lora' in name.lower():
                lora_params.append(param)
            else:
                other_params.append(param)
        
        # Create parameter groups with different learning rates
        param_groups = []
        
        if projector_params:
            param_groups.append({
                'params': projector_params,
                'lr': self.projector_lr,
                'weight_decay': self.weight_decay,
                'name': 'projector'
            })
            logger.info(f"Projector group: {len(projector_params)} parameters, LR={self.projector_lr}")
        
        if lora_params:
            param_groups.append({
                'params': lora_params,
                'lr': self.lora_lr,
                'weight_decay': self.weight_decay,
                'name': 'lora'
            })
            logger.info(f"LoRA group: {len(lora_params)} parameters, LR={self.lora_lr}")
        
        if other_params:
            param_groups.append({
                'params': other_params,
                'lr': self.learning_rate,
                'weight_decay': self.weight_decay,
                'name': 'other'
            })
            logger.info(f"Other group: {len(other_params)} parameters, LR={self.learning_rate}")
        
        if not param_groups:
            logger.warning("No trainable parameters found! Check model configuration.")
            # Create dummy parameter to avoid optimizer errors
            dummy_param = nn.Parameter(torch.tensor(0.0, requires_grad=True))
            param_groups = [{'params': [dummy_param], 'lr': self.learning_rate}]
        
        # Create AdamW optimizer with parameter groups
        optimizer = torch.optim.AdamW(
            param_groups,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Calculate total training steps
        if self.trainer.estimated_stepping_batches:
            total_steps = self.trainer.estimated_stepping_batches
        else:
            # Fallback estimation
            steps_per_epoch = len(self.trainer.datamodule.train_dataloader()) if hasattr(self.trainer, 'datamodule') else 1000
            total_steps = steps_per_epoch * self.max_epochs
            logger.warning(f"Could not estimate stepping batches, using fallback: {total_steps}")
        
        # Create cosine schedule with warmup
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps
        )
        
        logger.info(
            f"Configured optimizer with {len(param_groups)} parameter groups:\n"
            f"  Projector LR: {self.projector_lr}\n"
            f"  LoRA LR: {self.lora_lr}\n"
            f"  Scheduler: Cosine with warmup (warmup={self.warmup_steps}, total={total_steps})"
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # Update every step
                "frequency": 1,
                "name": "cosine_with_warmup"
            }
        }
    
    def configure_gradient_clipping(
        self,
        optimizer,
        gradient_clip_val: Optional[float] = None,
        gradient_clip_algorithm: Optional[str] = None
    ):
        """Configure gradient clipping.
        
        Args:
            optimizer: The optimizer
            gradient_clip_val: Maximum gradient norm (uses self.gradient_clip_val if None)
            gradient_clip_algorithm: Clipping algorithm (default: "norm")
        """
        if gradient_clip_val is None:
            gradient_clip_val = self.gradient_clip_val
        
        if gradient_clip_val > 0:
            self.clip_gradients(
                optimizer,
                gradient_clip_val=gradient_clip_val,
                gradient_clip_algorithm=gradient_clip_algorithm or "norm"
            )
    
    def _compute_gradient_norm(self) -> Optional[float]:
        """Compute the gradient norm for logging.
        
        Returns:
            Gradient norm as float, or None if no gradients available
        """
        try:
            total_norm = 0.0
            param_count = 0
            
            for p in self.model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    param_count += 1
            
            if param_count > 0:
                total_norm = total_norm ** (1. / 2)
                return total_norm
            else:
                return None
                
        except Exception as e:
            logger.warning(f"Failed to compute gradient norm: {e}")
            return None
    
    def on_train_epoch_start(self):
        """Called at the start of each training epoch."""
        logger.info(f"Starting training epoch {self.current_epoch + 1}/{self.max_epochs}")
        # Can't log on_step=True in epoch hooks, only on_epoch=True
        if self.trainer is not None:
            self.log("epoch", float(self.current_epoch), prog_bar=False, logger=True, on_step=False, on_epoch=True)
    
    def on_train_epoch_end(self):
        """Called at the end of each training epoch."""
        # Log epoch summary
        train_loss = self.trainer.callback_metrics.get('train_loss_epoch', None)
        if train_loss is not None:
            logger.info(f"Epoch {self.current_epoch + 1} completed. Train loss: {train_loss:.4f}")
    
    def on_validation_epoch_end(self):
        """Called at the end of each validation epoch."""
        # Log validation summary
        val_loss = self.trainer.callback_metrics.get('val_loss', None)
        if val_loss is not None:
            logger.info(f"Validation completed. Val loss: {val_loss:.4f}")
    
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Called when saving a checkpoint.
        
        Args:
            checkpoint: The checkpoint dictionary to be saved
        """
        # Add custom metadata to checkpoint
        checkpoint['model_info'] = {
            'total_params': sum(p.numel() for p in self.model.parameters()),
            'trainable_params': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            'model_type': type(self.model).__name__
        }
        
        # Add training state
        checkpoint['training_state'] = {
            'current_epoch': self.current_epoch,
            'global_step': self.trainer.global_step,
            'learning_rate': self.trainer.optimizers[0].param_groups[0]['lr'] if self.trainer.optimizers else None
        }
        
        logger.info(f"Saving checkpoint at epoch {self.current_epoch}, step {self.trainer.global_step}")
    
    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Called when loading a checkpoint.
        
        Args:
            checkpoint: The checkpoint dictionary being loaded
        """
        # Log checkpoint loading info
        if 'training_state' in checkpoint:
            state = checkpoint['training_state']
            logger.info(
                f"Loading checkpoint from epoch {state.get('current_epoch', 'unknown')}, "
                f"step {state.get('global_step', 'unknown')}"
            )
        
        if 'model_info' in checkpoint:
            info = checkpoint['model_info']
            logger.info(
                f"Checkpoint model info: {info.get('trainable_params', 'unknown')} trainable parameters"
            )