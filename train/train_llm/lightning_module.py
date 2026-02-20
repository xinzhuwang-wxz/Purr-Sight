"""
LLM Tuning Lightning Module
"""

import pytorch_lightning as pl
import torch
import torch.nn as nn
from typing import Dict, Any
from transformers import get_cosine_schedule_with_warmup

from purrsight.LLM.model import PurrSightMMLLM
from .train_llm_conf import LLMConfig
from purrsight.utils.logging import logger

class LLMTuningModule(pl.LightningModule):
    def __init__(self, config: LLMConfig):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        # Initialize Model
        self.model = PurrSightMMLLM(
            llm_model_path=config.llm_model_path,
            aligner_weights_path=config.adapter_path,
            freeze_encoders=config.freeze_encoders,
            freeze_projector=config.freeze_projector,
            freeze_llm=config.freeze_llm,
            lora_config=config.lora if config.lora.get('enabled') else None,
            projector_config=config.projector
        )
        
    def forward(self, batch):
        """Performs a forward pass through the model.

        Args:
            batch: A batch of data containing input tensors and labels.

        Returns:
            The model output containing loss and logits.
        """
        return self.model(batch, labels=batch.get('labels'))

    def training_step(self, batch, batch_idx):
        """Performs a single training step.

        Args:
            batch: The training batch.
            batch_idx: The index of the batch.

        Returns:
            The computed loss value.
        """
        outputs = self.model(batch, labels=batch['labels'])
        loss = outputs.loss
        
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Performs a single validation step.

        Args:
            batch: The validation batch.
            batch_idx: The index of the batch.

        Returns:
            The computed validation loss.
        """
        outputs = self.model(batch, labels=batch['labels'])
        loss = outputs.loss
        
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        """Configures optimizers and learning rate schedulers.

        Returns:
            A dictionary containing the optimizer and scheduler configuration.
        """
        # Only optimize parameters that require gradients
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        if not trainable_params:
            logger.warning("No trainable parameters found! Check freeze settings.")
            
        optimizer = torch.optim.AdamW(
            trainable_params, 
            lr=self.config.learning_rate, 
            weight_decay=self.config.weight_decay
        )
        
        # Scheduler
        if self.trainer.estimated_stepping_batches:
            total_steps = self.trainer.estimated_stepping_batches
        else:
            # Fallback estimation
            total_steps = self.config.epochs * 1000 # Placeholder
            
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=warmup_steps, 
            num_training_steps=total_steps
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }
        
    def on_save_checkpoint(self, checkpoint):
        """Called when saving a checkpoint.

        Args:
            checkpoint: The checkpoint dictionary to be saved.
        """
        # Handle Peft/LoRA saving separately if needed, 
        # but Lightning usually saves state_dict which includes LoRA weights.
        # However, for LoRA, we usually want to save only adapters.
        # This hook allows custom saving logic if needed.
        pass
