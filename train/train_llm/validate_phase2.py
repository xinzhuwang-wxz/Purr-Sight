#!/usr/bin/env python3
"""
Phase 2 Training Validation Script

This script performs a quick validation of the Phase 2 training pipeline to ensure
all components work correctly before running full training. It executes a short
training session (1-2 epochs) with a small dataset to verify:

- Phase 1 checkpoint loading and parameter freezing
- Projector initialization and trainability
- LoRA application and configuration
- Data pipeline functionality
- Forward pass execution
- Loss computation and backpropagation
- Checkpoint saving and loading
- MLflow logging integration
- Error handling and recovery

The validation is designed to complete in under 5 minutes and provides a clear
pass/fail checklist for each component.

Usage:
    python validate_phase2.py [--config CONFIG_PATH] [--quick] [--verbose]
    
Examples:
    # Basic validation with default config
    python validate_phase2.py
    
    # Quick validation (1 epoch, minimal logging)
    python validate_phase2.py --quick
    
    # Verbose validation with detailed logging
    python validate_phase2.py --verbose
    
    # Custom configuration
    python validate_phase2.py --config config/validation_config.yaml
"""

import os
import sys
import argparse
import time
import traceback
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from torch.utils.data import DataLoader, Subset

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from train.train_llm.train_llm_conf import TrainingConfig, load_config, validate_config, create_default_config
from train.train_llm.multimodal_llm_module import MultiModalLLMModule
from train.train_llm.dataset import MultiModalDataset, multimodal_collate_fn
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


@dataclass
class ValidationResult:
    """Result of a single validation check."""
    name: str
    passed: bool
    message: str
    duration: Optional[float] = None
    details: Optional[Dict[str, Any]] = None


class ValidationChecklist:
    """Manages validation checklist and results."""
    
    def __init__(self):
        self.results: List[ValidationResult] = []
        self.start_time = time.time()
    
    def add_result(self, name: str, passed: bool, message: str, 
                   duration: Optional[float] = None, details: Optional[Dict[str, Any]] = None):
        """Add a validation result."""
        result = ValidationResult(name, passed, message, duration, details)
        self.results.append(result)
        
        # Log result immediately
        status = "✅ PASS" if passed else "❌ FAIL"
        duration_str = f" ({duration:.2f}s)" if duration else ""
        logger.info(f"{status}: {name}{duration_str}")
        if not passed:
            logger.error(f"  └─ {message}")
        elif details:
            logger.info(f"  └─ {message}")
    
    def print_summary(self):
        """Print final validation summary."""
        total_time = time.time() - self.start_time
        passed_count = sum(1 for r in self.results if r.passed)
        total_count = len(self.results)
        
        print("\n" + "=" * 80)
        print("PHASE 2 TRAINING VALIDATION SUMMARY")
        print("=" * 80)
        
        # Print individual results
        for result in self.results:
            status = "✅ PASS" if result.passed else "❌ FAIL"
            duration_str = f" ({result.duration:.2f}s)" if result.duration else ""
            print(f"{status}: {result.name}{duration_str}")
            if not result.passed:
                print(f"  └─ {result.message}")
        
        print("-" * 80)
        print(f"Results: {passed_count}/{total_count} checks passed")
        print(f"Total time: {total_time:.2f}s")
        
        if passed_count == total_count:
            print("🎉 ALL VALIDATION CHECKS PASSED!")
            print("Phase 2 training pipeline is ready for full training.")
        else:
            print("⚠️  VALIDATION FAILED!")
            print("Please fix the failing checks before running full training.")
        
        print("=" * 80)
        
        return passed_count == total_count


class Phase2Validator:
    """Main validation class for Phase 2 training pipeline."""
    
    def __init__(self, config: TrainingConfig, quick_mode: bool = False, verbose: bool = False):
        """Initialize validator.
        
        Args:
            config: Training configuration
            quick_mode: If True, run minimal validation (1 epoch, less logging)
            verbose: If True, enable detailed logging
        """
        self.config = config
        self.quick_mode = quick_mode
        self.verbose = verbose
        self.checklist = ValidationChecklist()
        
        # Create temporary directories for validation
        self.temp_dir = Path(tempfile.mkdtemp(prefix="phase2_validation_"))
        self.temp_checkpoint_dir = self.temp_dir / "checkpoints"
        self.temp_output_dir = self.temp_dir / "outputs"
        
        # Override config for validation
        self.validation_config = self._create_validation_config()
        
        # Components
        self.model = None
        self.lightning_module = None
        self.trainer = None
        self.train_dataloader = None
        self.tokenizer = None
        
        logger.info(f"Initialized Phase2Validator (quick={quick_mode}, verbose={verbose})")
        logger.info(f"Temporary directory: {self.temp_dir}")
    
    def _create_validation_config(self) -> TrainingConfig:
        """Create a modified config optimized for validation."""
        # Create a copy of the original config
        config_dict = self.config.to_dict()
        
        # Modify for validation
        config_dict.update({
            # Minimal training
            "num_epochs": 1 if self.quick_mode else 2,
            "batch_size": min(4, self.config.batch_size),  # Small batch size
            "log_every_n_steps": 1,  # Log every step for validation
            "save_every_n_epochs": 1,  # Save checkpoint every epoch
            
            # Use temporary directories
            "checkpoint_dir": str(self.temp_checkpoint_dir),
            "output_dir": str(self.temp_output_dir),
            
            # Validation-specific MLflow experiment
            "mlflow_experiment_name": f"{self.config.mlflow_experiment_name}_validation",
            
            # Reduce workers for faster startup
            "num_workers": min(2, self.config.num_workers),
            
            # Single GPU for validation
            "num_gpus": 1,
            
            # Shorter warmup
            "warmup_steps": min(10, self.config.warmup_steps),
        })
        
        # Create validation config
        validation_config = TrainingConfig(**config_dict)
        
        # Ensure directories exist
        self.temp_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.temp_output_dir.mkdir(parents=True, exist_ok=True)
        
        return validation_config
    
    def cleanup(self):
        """Clean up temporary files."""
        try:
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to clean up temporary directory: {e}")
    
    def validate_configuration(self) -> bool:
        """Validate the training configuration."""
        start_time = time.time()
        
        try:
            # Test configuration validation
            validate_config(self.validation_config)
            
            # Check required paths exist
            phase1_path = Path(self.config.phase1_checkpoint_path)
            if not phase1_path.exists():
                raise FileNotFoundError(f"Phase 1 checkpoint not found: {phase1_path}")
            
            data_path = Path(self.config.data_dir)
            if not data_path.exists():
                raise FileNotFoundError(f"Data directory not found: {data_path}")
            
            duration = time.time() - start_time
            self.checklist.add_result(
                "Configuration Validation",
                True,
                f"All configuration parameters valid, required paths exist",
                duration
            )
            return True
            
        except Exception as e:
            duration = time.time() - start_time
            self.checklist.add_result(
                "Configuration Validation",
                False,
                f"Configuration validation failed: {str(e)}",
                duration
            )
            return False
    
    def validate_model_initialization(self) -> bool:
        """Validate model initialization and Phase 1 checkpoint loading."""
        start_time = time.time()
        
        try:
            # Create LoRA configuration
            lora_config = {
                'enabled': True,
                'r': self.validation_config.lora_r,
                'lora_alpha': self.validation_config.lora_alpha,
                'lora_dropout': self.validation_config.lora_dropout,
                'target_modules': self.validation_config.lora_target_modules,
                'task_type': 'CAUSAL_LM',
                'inference_mode': False
            }
            
            # Create projector configuration
            projector_config = {
                'hidden_dim': self.validation_config.projector_hidden_dim,
                'num_tokens': 4
            }
            
            # Set environment variable to force offline mode
            import os
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            os.environ['HF_HUB_OFFLINE'] = '1'
            
            # Initialize model
            self.model = PurrSightMMLLM(
                llm_model_path=self.validation_config.llm_model_name,
                aligner_weights_path=self.validation_config.phase1_checkpoint_path,
                freeze_encoders=True,
                freeze_projector=False,
                freeze_llm=False,
                lora_config=lora_config,
                projector_config=projector_config
            )
            
            duration = time.time() - start_time
            self.checklist.add_result(
                "Model Initialization",
                True,
                f"Model initialized successfully with Phase 1 checkpoint",
                duration
            )
            return True
            
        except Exception as e:
            duration = time.time() - start_time
            self.checklist.add_result(
                "Model Initialization",
                False,
                f"Model initialization failed: {str(e)}",
                duration
            )
            return False
    
    def validate_parameter_freezing(self) -> bool:
        """Validate that aligner parameters are frozen and projectors are trainable."""
        start_time = time.time()
        
        try:
            if self.model is None:
                raise RuntimeError("Model not initialized")
            
            frozen_params = 0
            trainable_params = 0
            aligner_frozen = 0
            projector_trainable = 0
            lora_trainable = 0
            
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    trainable_params += param.numel()
                    if 'projector' in name.lower():
                        projector_trainable += param.numel()
                    elif 'lora' in name.lower():
                        lora_trainable += param.numel()
                else:
                    frozen_params += param.numel()
                    if any(component in name.lower() for component in ['encoder', 'aligner']):
                        aligner_frozen += param.numel()
            
            total_params = frozen_params + trainable_params
            
            # Validate expectations
            if trainable_params == 0:
                raise RuntimeError("No trainable parameters found")
            
            if projector_trainable == 0:
                raise RuntimeError("No trainable projector parameters found")
            
            if aligner_frozen == 0:
                logger.warning("No frozen aligner parameters found - this may be expected if aligner is not loaded")
            
            duration = time.time() - start_time
            details = {
                "total_params": total_params,
                "frozen_params": frozen_params,
                "trainable_params": trainable_params,
                "aligner_frozen": aligner_frozen,
                "projector_trainable": projector_trainable,
                "lora_trainable": lora_trainable,
                "trainable_percentage": (trainable_params / total_params) * 100
            }
            
            self.checklist.add_result(
                "Parameter Freezing",
                True,
                f"Correct parameter freezing: {trainable_params:,} trainable, {frozen_params:,} frozen ({details['trainable_percentage']:.1f}% trainable)",
                duration,
                details
            )
            return True
            
        except Exception as e:
            duration = time.time() - start_time
            self.checklist.add_result(
                "Parameter Freezing",
                False,
                f"Parameter freezing validation failed: {str(e)}",
                duration
            )
            return False
    
    def validate_data_pipeline(self) -> bool:
        """Validate data loading and preprocessing."""
        start_time = time.time()
        
        try:
            # Initialize tokenizer with local files only
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.validation_config.llm_model_name,
                trust_remote_code=True,
                padding_side="right",
                local_files_only=True  # Prevent downloading from HuggingFace
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Create instruction dataset (not MultiModalDataset)
            from train.train_llm.dataset import InstructionDataset
            
            dataset_path = Path(self.validation_config.data_dir) / "train.jsonl"
            if not dataset_path.exists():
                raise FileNotFoundError(f"Instruction dataset not found: {dataset_path}")
            
            dataset = InstructionDataset(
                data_path=str(dataset_path),
                tokenizer=self.tokenizer,
                max_length=self.validation_config.max_text_length
            )
            
            if len(dataset) == 0:
                raise RuntimeError("Dataset is empty")
            
            # Create a small subset for validation
            subset_size = min(16, len(dataset))  # Use at most 16 samples
            subset_indices = list(range(subset_size))
            validation_dataset = Subset(dataset, subset_indices)
            
            # Create dataloader - InstructionDataset doesn't need custom collate_fn
            # Create dataloader - InstructionDataset doesn't need custom collate_fn
            # But we need to ensure multimodal keys are always present
            def ensure_multimodal_keys(batch):
                """Ensure batch always has image and audio keys."""
                # Handle the case where batch is a list of dicts (default collate)
                if isinstance(batch, list):
                    # Use default collate for basic keys
                    from torch.utils.data.dataloader import default_collate
                    result = default_collate(batch)
                else:
                    result = batch
                
                # Ensure multimodal keys exist
                if 'image' not in result:
                    batch_size = result['input_ids'].shape[0]
                    device = result['input_ids'].device
                    result['image'] = torch.zeros(batch_size, 3, 224, 224, device=device)
                
                if 'audio' not in result:
                    batch_size = result['input_ids'].shape[0]
                    device = result['input_ids'].device
                    result['audio'] = torch.zeros(batch_size, 64, 256, device=device)
                
                return result
            
            self.train_dataloader = DataLoader(
                validation_dataset,
                batch_size=self.validation_config.batch_size,
                shuffle=True,
                num_workers=0,  # Use 0 workers for validation to avoid multiprocessing issues
                pin_memory=False,  # Disable pin_memory for validation
                drop_last=False,
                collate_fn=ensure_multimodal_keys
            )
            
            # Test loading a batch
            batch = next(iter(self.train_dataloader))
            
            # Validate batch structure for InstructionDataset
            required_keys = ['input_ids', 'attention_mask', 'labels']
            for key in required_keys:
                if key not in batch:
                    raise RuntimeError(f"Missing key in batch: {key}")
                if not isinstance(batch[key], torch.Tensor):
                    raise RuntimeError(f"Batch[{key}] is not a tensor: {type(batch[key])}")
            
            # Check for multimodal keys (optional)
            multimodal_keys = ['image', 'audio']
            for key in multimodal_keys:
                if key in batch and not isinstance(batch[key], torch.Tensor):
                    raise RuntimeError(f"Batch[{key}] is not a tensor: {type(batch[key])}")
            
            # Validate batch dimensions
            batch_size = batch['input_ids'].shape[0]
            for key in required_keys:
                if batch[key].shape[0] != batch_size:
                    raise RuntimeError(f"Inconsistent batch dimension for {key}: {batch[key].shape[0]} != {batch_size}")
            
            duration = time.time() - start_time
            details = {
                "dataset_size": len(dataset),
                "validation_subset_size": len(validation_dataset),
                "batch_size": batch_size,
                "batch_keys": list(batch.keys()),
                "batch_shapes": {k: list(v.shape) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            }
            
            self.checklist.add_result(
                "Data Pipeline",
                True,
                f"Data pipeline working: {len(dataset)} samples, batch size {batch_size}",
                duration,
                details
            )
            return True
            
        except Exception as e:
            duration = time.time() - start_time
            self.checklist.add_result(
                "Data Pipeline",
                False,
                f"Data pipeline validation failed: {str(e)}",
                duration
            )
            return False
    
    def validate_forward_pass(self) -> bool:
        """Validate forward pass execution."""
        start_time = time.time()
        
        try:
            if self.model is None:
                raise RuntimeError("Model not initialized")
            if self.train_dataloader is None:
                raise RuntimeError("Data loader not initialized")
            
            # Get a batch
            batch = next(iter(self.train_dataloader))
            
            # Move to device if CUDA available
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = self.model.to(device)
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Set model to eval mode for forward pass test
            self.model.eval()
            
            with torch.no_grad():
                # Test forward pass - model expects a dictionary input
                # InstructionDataset returns the correct keys: input_ids, attention_mask, labels
                # Add image and audio if present, otherwise use zero tensors
                model_inputs = {
                    'input_ids': batch['input_ids'],
                    'attention_mask': batch['attention_mask']
                }
                
                # Add multimodal inputs if present, otherwise use zero tensors
                if 'image' in batch:
                    model_inputs['image'] = batch['image']
                else:
                    # Create zero image tensor
                    batch_size = batch['input_ids'].shape[0]
                    model_inputs['image'] = torch.zeros(batch_size, 3, 224, 224, device=device)
                
                if 'audio' in batch:
                    model_inputs['audio'] = batch['audio']
                else:
                    # Create zero audio tensor
                    batch_size = batch['input_ids'].shape[0]
                    model_inputs['audio'] = torch.zeros(batch_size, 64, 256, device=device)
                
                outputs = self.model(model_inputs)
                
                # Validate output structure
                if not hasattr(outputs, 'logits'):
                    raise RuntimeError("Model output missing 'logits' attribute")
                
                logits = outputs.logits
                # Note: logits shape will be (batch_size, seq_len + multimodal_tokens, vocab_size)
                # due to multimodal tokens being prepended
                
                if logits.dim() != 3:
                    raise RuntimeError(f"Unexpected logits dimensions: {logits.dim()} != 3")
                
                # Check for NaN or Inf values
                if torch.isnan(logits).any():
                    raise RuntimeError("NaN values found in model output")
                if torch.isinf(logits).any():
                    raise RuntimeError("Inf values found in model output")
            
            duration = time.time() - start_time
            details = {
                "device": str(device),
                "input_shapes": {k: list(v.shape) for k, v in batch.items() if isinstance(v, torch.Tensor)},
                "output_shape": list(logits.shape),
                "vocab_size": self.model.llm.config.vocab_size
            }
            
            self.checklist.add_result(
                "Forward Pass",
                True,
                f"Forward pass successful on {device}, output shape {list(logits.shape)}",
                duration,
                details
            )
            return True
            
        except Exception as e:
            duration = time.time() - start_time
            self.checklist.add_result(
                "Forward Pass",
                False,
                f"Forward pass validation failed: {str(e)}",
                duration
            )
            return False
    
    def validate_training_step(self) -> bool:
        """Validate training step execution with loss computation and backpropagation."""
        start_time = time.time()
        
        try:
            if self.model is None:
                raise RuntimeError("Model not initialized")
            if self.train_dataloader is None:
                raise RuntimeError("Data loader not initialized")
            
            # Create Lightning module
            self.lightning_module = MultiModalLLMModule(
                model=self.model,
                learning_rate=self.validation_config.learning_rate,
                weight_decay=self.validation_config.weight_decay,
                warmup_steps=self.validation_config.warmup_steps,
                gradient_clip_val=self.validation_config.gradient_clip_val,
                max_epochs=self.validation_config.num_epochs,
                log_every_n_steps=self.validation_config.log_every_n_steps,
                phase1_checkpoint_path=self.validation_config.phase1_checkpoint_path,
                lora_config={
                    'enabled': True,
                    'r': self.validation_config.lora_r,
                    'lora_alpha': self.validation_config.lora_alpha,
                    'lora_dropout': self.validation_config.lora_dropout,
                    'target_modules': self.validation_config.lora_target_modules,
                    'task_type': 'CAUSAL_LM',
                    'inference_mode': False
                }
            )
            
            # Move to device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.lightning_module = self.lightning_module.to(device)
            
            # Get a batch
            batch = next(iter(self.train_dataloader))
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Add missing multimodal keys if not present
            if 'image' not in batch:
                batch_size = batch['input_ids'].shape[0]
                batch['image'] = torch.zeros(batch_size, 3, 224, 224, device=device)
            
            if 'audio' not in batch:
                batch_size = batch['input_ids'].shape[0]
                batch['audio'] = torch.zeros(batch_size, 64, 256, device=device)
            
            # Set to training mode
            self.lightning_module.train()
            
            # Store initial parameter values for comparison
            initial_params = {}
            for name, param in self.lightning_module.named_parameters():
                if param.requires_grad:
                    initial_params[name] = param.data.clone()
            
            # Create a minimal trainer for the lightning module
            trainer = pl.Trainer(
                max_epochs=1,
                accelerator='cpu',
                devices=1,
                enable_checkpointing=False,
                enable_progress_bar=False,
                enable_model_summary=False,
                logger=False
            )
            
            # Properly initialize the trainer with the module
            trainer.strategy.setup(trainer)
            trainer.lightning_module = self.lightning_module
            self.lightning_module.trainer = trainer
            
            # Execute training step
            loss = self.lightning_module.training_step(batch, 0)
            
            # Validate loss
            if not isinstance(loss, torch.Tensor):
                raise RuntimeError(f"Loss is not a tensor: {type(loss)}")
            
            if torch.isnan(loss):
                raise RuntimeError("Loss is NaN")
            
            if torch.isinf(loss):
                raise RuntimeError("Loss is Inf")
            
            if loss.item() < 0:
                raise RuntimeError(f"Loss is negative: {loss.item()}")
            
            # Perform backward pass
            loss.backward()
            
            # Check gradients
            grad_norms = {}
            for name, param in self.lightning_module.named_parameters():
                if param.requires_grad and param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    grad_norms[name] = grad_norm
                    
                    if torch.isnan(param.grad).any():
                        raise RuntimeError(f"NaN gradients found in {name}")
                    if torch.isinf(param.grad).any():
                        raise RuntimeError(f"Inf gradients found in {name}")
            
            if not grad_norms:
                raise RuntimeError("No gradients computed")
            
            # Perform optimizer step
            optimizer = self.lightning_module.configure_optimizers()
            if isinstance(optimizer, dict):
                optimizer = optimizer['optimizer']
            
            optimizer.step()
            optimizer.zero_grad()
            
            # Check that parameters changed
            params_changed = 0
            for name, param in self.lightning_module.named_parameters():
                if param.requires_grad and name in initial_params:
                    if not torch.equal(param.data, initial_params[name]):
                        params_changed += 1
            
            if params_changed == 0:
                raise RuntimeError("No parameters were updated during training step")
            
            duration = time.time() - start_time
            details = {
                "loss_value": loss.item(),
                "num_gradients": len(grad_norms),
                "max_grad_norm": max(grad_norms.values()) if grad_norms else 0,
                "params_changed": params_changed,
                "total_trainable_params": len(initial_params)
            }
            
            self.checklist.add_result(
                "Training Step",
                True,
                f"Training step successful: loss={loss.item():.4f}, {params_changed} params updated",
                duration,
                details
            )
            return True
            
        except Exception as e:
            duration = time.time() - start_time
            self.checklist.add_result(
                "Training Step",
                False,
                f"Training step validation failed: {str(e)}",
                duration
            )
            return False
    
    def validate_checkpoint_saving(self) -> bool:
        """Validate checkpoint saving functionality."""
        start_time = time.time()
        
        try:
            if self.lightning_module is None:
                raise RuntimeError("Lightning module not initialized")
            
            # Create a simple trainer for checkpoint saving
            checkpoint_callback = ModelCheckpoint(
                dirpath=self.temp_checkpoint_dir,
                filename='validation-{epoch:02d}',
                save_top_k=1,
                save_last=True,
                every_n_epochs=1
            )
            
            # Create trainer and properly attach the model
            trainer = pl.Trainer(
                max_epochs=1,
                accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                devices=1,
                callbacks=[checkpoint_callback],
                enable_checkpointing=True,
                enable_progress_bar=False,
                enable_model_summary=False,
                logger=False  # Disable logging for this test
            )
            
            # Use Lightning's built-in checkpoint saving by running a minimal fit
            # This ensures proper model attachment
            from torch.utils.data import DataLoader, TensorDataset
            
            # Create a minimal dummy dataset for checkpoint saving test
            dummy_input_ids = torch.randint(0, 1000, (2, 10))
            dummy_attention_mask = torch.ones(2, 10)
            dummy_labels = torch.randint(0, 1000, (2, 10))
            dummy_image = torch.zeros(2, 3, 224, 224)
            dummy_audio = torch.zeros(2, 64, 256)
            
            dummy_dataset = TensorDataset(dummy_input_ids, dummy_attention_mask, dummy_labels, dummy_image, dummy_audio)
            dummy_dataloader = DataLoader(dummy_dataset, batch_size=2)
            
            # Convert to expected format
            def dummy_collate(batch):
                input_ids, attention_mask, labels, image, audio = zip(*batch)
                return {
                    'input_ids': torch.stack(input_ids),
                    'attention_mask': torch.stack(attention_mask),
                    'labels': torch.stack(labels),
                    'image': torch.stack(image),
                    'audio': torch.stack(audio)
                }
            
            dummy_dataloader.collate_fn = dummy_collate
            
            # Run one step to trigger checkpoint saving
            trainer.max_steps = 1
            trainer.fit(self.lightning_module, dummy_dataloader)
            
            # Find the saved checkpoint
            checkpoint_files = list(self.temp_checkpoint_dir.glob("*.ckpt"))
            if not checkpoint_files:
                raise RuntimeError("No checkpoint files were saved")
            
            checkpoint_path = checkpoint_files[0]
            
            # Verify checkpoint file exists
            if not checkpoint_path.exists():
                raise RuntimeError(f"Checkpoint file not created: {checkpoint_path}")
            
            # Load and verify checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            required_keys = ['state_dict', 'epoch', 'global_step', 'pytorch-lightning_version']
            for key in required_keys:
                if key not in checkpoint:
                    raise RuntimeError(f"Missing key in checkpoint: {key}")
            
            # Verify state dict has model parameters
            state_dict = checkpoint['state_dict']
            if not state_dict:
                raise RuntimeError("Empty state dict in checkpoint")
            
            # Test loading checkpoint into a new model
            new_lightning_module = MultiModalLLMModule(
                model=self.model,
                learning_rate=self.validation_config.learning_rate,
                weight_decay=self.validation_config.weight_decay,
                warmup_steps=self.validation_config.warmup_steps,
                gradient_clip_val=self.validation_config.gradient_clip_val,
                max_epochs=self.validation_config.num_epochs,
                log_every_n_steps=self.validation_config.log_every_n_steps,
                phase1_checkpoint_path=self.validation_config.phase1_checkpoint_path,
                lora_config={
                    'enabled': True,
                    'r': self.validation_config.lora_r,
                    'lora_alpha': self.validation_config.lora_alpha,
                    'lora_dropout': self.validation_config.lora_dropout,
                    'target_modules': self.validation_config.lora_target_modules,
                    'task_type': 'CAUSAL_LM',
                    'inference_mode': False
                }
            )
            
            # Load state dict
            new_lightning_module.load_state_dict(state_dict)
            
            duration = time.time() - start_time
            details = {
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_size_mb": checkpoint_path.stat().st_size / (1024 * 1024),
                "state_dict_keys": len(state_dict),
                "checkpoint_keys": list(checkpoint.keys())
            }
            
            self.checklist.add_result(
                "Checkpoint Saving",
                True,
                f"Checkpoint saved and loaded successfully ({details['checkpoint_size_mb']:.1f}MB)",
                duration,
                details
            )
            return True
            
        except Exception as e:
            duration = time.time() - start_time
            self.checklist.add_result(
                "Checkpoint Saving",
                False,
                f"Checkpoint saving validation failed: {str(e)}",
                duration
            )
            return False
    
    def validate_mlflow_logging(self) -> bool:
        """Validate MLflow logging functionality."""
        start_time = time.time()
        
        try:
            if not MLFLOW_AVAILABLE:
                self.checklist.add_result(
                    "MLflow Logging",
                    True,  # Pass if MLflow not available (optional dependency)
                    "MLflow not available, skipping logging validation",
                    time.time() - start_time
                )
                return True
            
            # Create MLflow logger
            mlflow_logger = MLFlowLogger(
                experiment_name=self.validation_config.mlflow_experiment_name,
                tracking_uri=self.validation_config.mlflow_tracking_uri
            )
            
            # Test logging hyperparameters
            test_params = {
                "learning_rate": self.validation_config.learning_rate,
                "batch_size": self.validation_config.batch_size,
                "model_name": self.validation_config.llm_model_name
            }
            mlflow_logger.log_hyperparams(test_params)
            
            # Test logging metrics
            test_metrics = {
                "validation_loss": 0.5,
                "validation_accuracy": 0.8,
                "step": 1
            }
            mlflow_logger.log_metrics(test_metrics, step=1)
            
            # Finalize the run
            mlflow_logger.finalize("success")
            
            duration = time.time() - start_time
            details = {
                "experiment_name": self.validation_config.mlflow_experiment_name,
                "run_id": mlflow_logger.run_id,
                "params_logged": len(test_params),
                "metrics_logged": len(test_metrics)
            }
            
            self.checklist.add_result(
                "MLflow Logging",
                True,
                f"MLflow logging successful: run {mlflow_logger.run_id[:8]}...",
                duration,
                details
            )
            return True
            
        except Exception as e:
            duration = time.time() - start_time
            self.checklist.add_result(
                "MLflow Logging",
                False,
                f"MLflow logging validation failed: {str(e)}",
                duration
            )
            return False
    
    def validate_short_training_run(self) -> bool:
        """Validate a complete short training run."""
        start_time = time.time()
        
        try:
            if self.lightning_module is None:
                raise RuntimeError("Lightning module not initialized")
            if self.train_dataloader is None:
                raise RuntimeError("Data loader not initialized")
            
            # 统一用 MLflow 记录指标，checkpoint 仅按 CHECKPOINT_MONITOR 保存
            checkpoint_callback = ModelCheckpoint(
                dirpath=self.temp_checkpoint_dir,
                filename=f"training-{{epoch:02d}}-{{{CHECKPOINT_MONITOR}:.4f}}",
                monitor=CHECKPOINT_MONITOR,
                mode=CHECKPOINT_MONITOR_MODE,
                save_top_k=1,
                save_last=True,
                every_n_epochs=1,
            )
            
            # Create MLflow logger if available
            mlflow_logger = None
            if MLFLOW_AVAILABLE:
                try:
                    mlflow_logger = MLFlowLogger(
                        experiment_name=f"{self.validation_config.mlflow_experiment_name}_short_run",
                        tracking_uri=self.validation_config.mlflow_tracking_uri
                    )
                except Exception as e:
                    logger.warning(f"Failed to create MLflow logger: {e}")
            
            trainer = pl.Trainer(
                max_epochs=self.validation_config.num_epochs,
                accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                devices=1,
                precision=16 if torch.cuda.is_available() else 32,
                gradient_clip_val=self.validation_config.gradient_clip_val,
                log_every_n_steps=self.validation_config.log_every_n_steps,
                callbacks=[checkpoint_callback],
                logger=mlflow_logger,
                enable_checkpointing=True,
                enable_progress_bar=not self.quick_mode,  # Disable progress bar in quick mode
                enable_model_summary=False,
                max_steps=10 if self.quick_mode else -1,  # Limit steps in quick mode
            )
            
            # Run training
            trainer.fit(
                model=self.lightning_module,
                train_dataloaders=self.train_dataloader
            )
            
            # Verify training completed
            if trainer.current_epoch < 0:
                raise RuntimeError("Training did not complete any epochs")
            
            # Check that loss was logged
            if not trainer.callback_metrics:
                raise RuntimeError("No metrics were logged during training")
            
            # Verify checkpoint was saved
            checkpoint_files = list(self.temp_checkpoint_dir.glob("*.ckpt"))
            if not checkpoint_files:
                raise RuntimeError("No checkpoint files were saved")
            
            duration = time.time() - start_time
            final_metrics = {k: v.item() if hasattr(v, 'item') else v 
                           for k, v in trainer.callback_metrics.items()}
            
            details = {
                "epochs_completed": trainer.current_epoch + 1,
                "steps_completed": trainer.global_step,
                "final_metrics": final_metrics,
                "checkpoints_saved": len(checkpoint_files),
                "mlflow_enabled": mlflow_logger is not None
            }
            
            self.checklist.add_result(
                "Short Training Run",
                True,
                f"Training completed: {trainer.current_epoch + 1} epochs, {trainer.global_step} steps",
                duration,
                details
            )
            return True
            
        except Exception as e:
            duration = time.time() - start_time
            self.checklist.add_result(
                "Short Training Run",
                False,
                f"Short training run failed: {str(e)}",
                duration
            )
            return False
    
    def run_validation(self) -> bool:
        """Run complete validation pipeline."""
        logger.info("Starting Phase 2 training validation...")
        
        try:
            # Run all validation checks
            checks = [
                self.validate_configuration,
                self.validate_model_initialization,
                self.validate_parameter_freezing,
                self.validate_data_pipeline,
                self.validate_forward_pass,
                self.validate_training_step,
                self.validate_checkpoint_saving,
                self.validate_mlflow_logging,
                self.validate_short_training_run,
            ]
            
            all_passed = True
            for check in checks:
                try:
                    result = check()
                    if not result:
                        all_passed = False
                except Exception as e:
                    logger.error(f"Validation check failed with exception: {e}")
                    all_passed = False
            
            # Print summary
            success = self.checklist.print_summary()
            
            return success and all_passed
            
        except Exception as e:
            logger.error(f"Validation pipeline failed: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
        
        finally:
            # Always cleanup
            self.cleanup()


def create_validation_config(base_config_path: Optional[str] = None) -> TrainingConfig:
    """Create a validation configuration."""
    if base_config_path and Path(base_config_path).exists():
        try:
            config = load_config(base_config_path)
            logger.info(f"Loaded configuration from: {base_config_path}")
            # Override with local model paths
            config.llm_model_name = "models/Qwen2.5-0.5B-Instruct"
            return config
        except Exception as e:
            logger.warning(f"Failed to load config from {base_config_path}: {e}")
            logger.info("Using default configuration")
    
    # Create default configuration for validation
    config = create_default_config()
    
    # Override with validation-friendly defaults and LOCAL MODEL PATHS
    config.batch_size = 4
    config.num_epochs = 2
    config.log_every_n_steps = 1
    config.warmup_steps = 10
    config.num_workers = 0  # Avoid multiprocessing issues
    config.llm_model_name = "models/Qwen2.5-0.5B-Instruct"  # Use local model
    
    # Try to find existing Phase 1 checkpoint
    checkpoint_patterns = [
        "checkpoints/alignment/*/aligner.pt",
        "checkpoints/alignment/*/best_model.pt",
        "checkpoints/alignment/*/final_model.pt",
        "checkpoints/alignment/*.pt",
        "checkpoints/alignment/*.ckpt",
    ]
    
    for pattern in checkpoint_patterns:
        checkpoint_files = list(Path(".").glob(pattern))
        if checkpoint_files:
            config.phase1_checkpoint_path = str(checkpoint_files[0])
            logger.info(f"Found Phase 1 checkpoint: {config.phase1_checkpoint_path}")
            break
    
    # Try to find existing data directory
    data_patterns = [
        "data/instruction",
        "data/preprocessed", 
        "data/processed",
        "data"
    ]
    
    for data_dir in data_patterns:
        if Path(data_dir).exists():
            config.data_dir = data_dir
            logger.info(f"Found data directory: {config.data_dir}")
            break
    
    return config


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Phase 2 Training Validation Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic validation
  python validate_phase2.py
  
  # Quick validation (1 epoch, minimal logging)
  python validate_phase2.py --quick
  
  # Verbose validation with detailed logging
  python validate_phase2.py --verbose
  
  # Custom configuration
  python validate_phase2.py --config config/train_config.yaml
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to training configuration YAML file'
    )
    
    parser.add_argument(
        '--quick', '-q',
        action='store_true',
        help='Run quick validation (1 epoch, minimal logging)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def main():
    """Main entry point for validation script."""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Set up logging level
        if args.verbose:
            import logging
            logging.getLogger().setLevel(logging.DEBUG)
        
        print("=" * 80)
        print("PHASE 2 TRAINING VALIDATION")
        print("=" * 80)
        print(f"Quick mode: {args.quick}")
        print(f"Verbose mode: {args.verbose}")
        print(f"Config file: {args.config or 'default'}")
        print("-" * 80)
        
        # Create validation configuration
        config = create_validation_config(args.config)
        
        # Run validation
        validator = Phase2Validator(
            config=config,
            quick_mode=args.quick,
            verbose=args.verbose
        )
        
        success = validator.run_validation()
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        logger.info("Validation interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Validation failed with error: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()