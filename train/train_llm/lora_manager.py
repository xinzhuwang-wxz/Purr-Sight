"""LoRA Manager for Phase 2 Training.

This module provides utilities for applying LoRA (Low-Rank Adaptation) to LLMs,
verifying trainable parameters, and validating LoRA configurations.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from transformers import PreTrainedModel

# Try to import peft, but make it optional for testing
try:
    from peft import LoraConfig, get_peft_model, PeftModel, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    LoraConfig = None
    get_peft_model = None
    PeftModel = None
    TaskType = None

# Try to import mlflow, but make it optional
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None

from purrsight.utils.logging import logger


class LoRAManager:
    """Manages LoRA configuration and application for LLM fine-tuning.
    
    This class provides methods to:
    - Apply LoRA adapters to pre-trained LLMs
    - Verify that only LoRA parameters are trainable
    - Validate LoRA configuration parameters
    - Log LoRA configuration to MLflow
    
    LoRA (Low-Rank Adaptation) enables parameter-efficient fine-tuning by
    adding trainable low-rank matrices to specific modules while keeping
    the base model frozen.
    """
    
    @staticmethod
    def apply_lora(
        model: PreTrainedModel,
        lora_config: Dict[str, Any]
    ) -> 'PeftModel':
        """Apply LoRA configuration to LLM.
        
        This function wraps a pre-trained language model with LoRA adapters,
        enabling parameter-efficient fine-tuning. It validates the configuration,
        applies LoRA to specified target modules, and logs the configuration.
        
        Args:
            model: Base LLM model (e.g., Qwen2.5-0.5B, Llama)
            lora_config: LoRA configuration dictionary containing:
                - r: LoRA rank (positive integer, typically 8-64)
                - lora_alpha: LoRA scaling factor (positive integer, typically 16-32)
                - target_modules: List of module names to apply LoRA (e.g., ["q_proj", "v_proj"])
                - lora_dropout: Dropout probability (float between 0 and 1)
                - task_type: Optional task type (defaults to CAUSAL_LM)
                - inference_mode: Optional inference mode flag (defaults to False)
                
        Returns:
            Model wrapped with LoRA adapters (PeftModel)
            
        Raises:
            ImportError: If peft library is not installed
            ValueError: If configuration is invalid (negative rank, invalid modules, etc.)
            RuntimeError: If LoRA application fails
        """
        # Check if peft is available
        if not PEFT_AVAILABLE:
            error_msg = (
                "peft library is not installed. "
                "Please install it with: pip install peft"
            )
            logger.error(error_msg)
            raise ImportError(error_msg)
        
        # Validate configuration
        LoRAManager._validate_lora_config(lora_config)
        
        # Extract configuration parameters with defaults
        r = lora_config.get('r', 16)
        lora_alpha = lora_config.get('lora_alpha', 32)
        lora_dropout = lora_config.get('lora_dropout', 0.1)
        target_modules = lora_config.get('target_modules', ["q_proj", "v_proj", "k_proj", "o_proj"])
        task_type = lora_config.get('task_type', TaskType.CAUSAL_LM)
        inference_mode = lora_config.get('inference_mode', False)
        
        logger.info(
            f"Applying LoRA with config: r={r}, alpha={lora_alpha}, "
            f"dropout={lora_dropout}, target_modules={target_modules}"
        )
        
        try:
            # Create LoRA configuration
            peft_config = LoraConfig(
                task_type=task_type,
                inference_mode=inference_mode,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules
            )
            
            # Apply LoRA to model
            peft_model = get_peft_model(model, peft_config)
            
            # Unfreeze projector parameters (they should remain trainable)
            for name, param in peft_model.named_parameters():
                if 'projector' in name.lower():
                    param.requires_grad = True
            
            # Log trainable parameters
            if hasattr(peft_model, 'print_trainable_parameters'):
                peft_model.print_trainable_parameters()
            
            logger.info("Successfully applied LoRA to model")
            
            # Log to MLflow if active run exists
            try:
                if MLFLOW_AVAILABLE and mlflow.active_run():
                    mlflow.log_param("lora_r", r)
                    mlflow.log_param("lora_alpha", lora_alpha)
                    mlflow.log_param("lora_dropout", lora_dropout)
                    mlflow.log_param("lora_target_modules", ','.join(target_modules))
                    logger.info("Logged LoRA configuration to MLflow")
            except Exception as e:
                logger.warning(f"Failed to log LoRA config to MLflow: {e}")
            
            return peft_model
            
        except Exception as e:
            error_msg = f"Failed to apply LoRA to model: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    @staticmethod
    def _validate_lora_config(lora_config: Dict[str, Any]) -> None:
        """Validate LoRA configuration parameters.
        
        Checks that all required parameters are present and have valid values.
        
        Args:
            lora_config: LoRA configuration dictionary
            
        Raises:
            ValueError: If configuration is invalid
        """
        errors = []
        
        # Check rank (r)
        r = lora_config.get('r')
        if r is not None:
            if not isinstance(r, int):
                errors.append(f"LoRA rank 'r' must be an integer, got {type(r).__name__}")
            elif r <= 0:
                errors.append(f"LoRA rank 'r' must be positive, got {r}")
            elif r > 256:
                errors.append(f"LoRA rank 'r' is unusually large ({r}), typically should be 8-64")
        
        # Check alpha
        lora_alpha = lora_config.get('lora_alpha')
        if lora_alpha is not None:
            if not isinstance(lora_alpha, (int, float)):
                errors.append(f"LoRA alpha must be numeric, got {type(lora_alpha).__name__}")
            elif lora_alpha <= 0:
                errors.append(f"LoRA alpha must be positive, got {lora_alpha}")
        
        # Check dropout
        lora_dropout = lora_config.get('lora_dropout')
        if lora_dropout is not None:
            if not isinstance(lora_dropout, (int, float)):
                errors.append(f"LoRA dropout must be numeric, got {type(lora_dropout).__name__}")
            elif lora_dropout < 0 or lora_dropout > 1:
                errors.append(f"LoRA dropout must be between 0 and 1, got {lora_dropout}")
        
        # Check target_modules
        target_modules = lora_config.get('target_modules')
        if target_modules is not None:
            if not isinstance(target_modules, list):
                errors.append(f"target_modules must be a list, got {type(target_modules).__name__}")
            elif len(target_modules) == 0:
                errors.append("target_modules cannot be empty")
            elif not all(isinstance(m, str) for m in target_modules):
                errors.append("All target_modules must be strings")
            else:
                # Check for common valid module names
                valid_module_patterns = [
                    'q_proj', 'k_proj', 'v_proj', 'o_proj',  # Attention projections
                    'gate_proj', 'up_proj', 'down_proj',      # MLP projections
                    'query', 'key', 'value', 'dense',         # Alternative names
                    'fc1', 'fc2', 'c_attn', 'c_proj'          # Other common names
                ]
                # Warn if modules don't match common patterns (but don't error)
                for module in target_modules:
                    if not any(pattern in module for pattern in valid_module_patterns):
                        logger.warning(
                            f"Target module '{module}' doesn't match common patterns. "
                            f"Make sure it exists in your model."
                        )
        
        # If there are errors, raise ValueError with all error messages
        if errors:
            error_msg = "Invalid LoRA configuration:\n" + "\n".join(f"  - {err}" for err in errors)
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    @staticmethod
    def verify_trainable_parameters(model: nn.Module) -> Dict[str, int]:
        """Verify that only LoRA and projector parameters are trainable.
        
        This function counts trainable and frozen parameters in the model,
        categorizing them by type (LoRA adapters, projectors, base model).
        It helps verify that LoRA was applied correctly and only the intended
        parameters are being fine-tuned.
        
        Args:
            model: Model with LoRA applied (can be PeftModel or regular nn.Module)
            
        Returns:
            Dictionary with parameter counts:
                - total_params: Total number of parameters
                - trainable_params: Number of trainable parameters
                - frozen_params: Number of frozen parameters
                - lora_params: Number of LoRA adapter parameters
                - projector_params: Number of projector parameters
                - other_trainable_params: Number of other trainable parameters
                - trainable_percentage: Percentage of trainable parameters
        """
        total_params = 0
        trainable_params = 0
        frozen_params = 0
        lora_params = 0
        projector_params = 0
        other_trainable_params = 0
        
        for name, param in model.named_parameters():
            total_params += param.numel()
            
            if param.requires_grad:
                trainable_params += param.numel()
                
                # Categorize trainable parameters
                if 'lora' in name.lower():
                    lora_params += param.numel()
                elif 'projector' in name.lower():
                    projector_params += param.numel()
                else:
                    other_trainable_params += param.numel()
            else:
                frozen_params += param.numel()
        
        # Calculate percentage
        trainable_percentage = (trainable_params / total_params * 100) if total_params > 0 else 0
        
        # Create result dictionary
        result = {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'frozen_params': frozen_params,
            'lora_params': lora_params,
            'projector_params': projector_params,
            'other_trainable_params': other_trainable_params,
            'trainable_percentage': trainable_percentage
        }
        
        # Log summary
        logger.info(
            f"Parameter Summary:\n"
            f"  Total: {total_params:,}\n"
            f"  Trainable: {trainable_params:,} ({trainable_percentage:.2f}%)\n"
            f"  Frozen: {frozen_params:,}\n"
            f"  LoRA: {lora_params:,}\n"
            f"  Projector: {projector_params:,}\n"
            f"  Other Trainable: {other_trainable_params:,}"
        )
        
        # Log to MLflow if active run exists
        try:
            if MLFLOW_AVAILABLE and mlflow.active_run():
                mlflow.log_metric("total_params", total_params)
                mlflow.log_metric("trainable_params", trainable_params)
                mlflow.log_metric("trainable_percentage", trainable_percentage)
                mlflow.log_metric("lora_params", lora_params)
                mlflow.log_metric("projector_params", projector_params)
                logger.info("Logged parameter counts to MLflow")
        except Exception as e:
            logger.warning(f"Failed to log parameter counts to MLflow: {e}")
        
        return result
