#!/usr/bin/env python3
"""
Simple Phase 2 Training Validation Script

A simplified version of the Phase 2 validation that focuses on core functionality
without complex multiprocessing or Lightning trainer setup. This script validates:

1. Configuration loading and validation
2. Model initialization and checkpoint loading
3. Parameter freezing verification
4. Data pipeline functionality
5. Forward pass execution
6. Basic training step validation

Usage:
    python validate_phase2_simple.py [--config CONFIG_PATH]
"""

import os
import sys
import argparse
import time
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from train.train_llm.train_llm_conf import TrainingConfig, load_config, validate_config
from train.train_llm.dataset import MultiModalDataset, multimodal_collate_fn
from purrsight.LLM.model import PurrSightMMLLM
from purrsight.utils.logging import logger
from transformers import AutoTokenizer


class SimpleValidator:
    """Simple validator for Phase 2 training pipeline."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.results = []
        self.model = None
        self.tokenizer = None
        self.dataset = None
        self.dataloader = None
    
    def log_result(self, name: str, passed: bool, message: str, duration: float = 0):
        """Log a validation result."""
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name} ({duration:.2f}s)")
        if message:
            print(f"  └─ {message}")
        self.results.append((name, passed, message, duration))
    
    def validate_configuration(self) -> bool:
        """Validate configuration."""
        start_time = time.time()
        try:
            validate_config(self.config)
            
            # Check paths exist
            phase1_path = Path(self.config.phase1_checkpoint_path)
            if not phase1_path.exists():
                raise FileNotFoundError(f"Phase 1 checkpoint not found: {phase1_path}")
            
            data_path = Path(self.config.data_dir)
            if not data_path.exists():
                raise FileNotFoundError(f"Data directory not found: {data_path}")
            
            duration = time.time() - start_time
            self.log_result("Configuration Validation", True, 
                          f"Config valid, paths exist", duration)
            return True
        except Exception as e:
            duration = time.time() - start_time
            self.log_result("Configuration Validation", False, str(e), duration)
            return False
    
    def validate_model_initialization(self) -> bool:
        """Validate model initialization."""
        start_time = time.time()
        try:
            lora_config = {
                'enabled': True,
                'r': self.config.lora_r,
                'lora_alpha': self.config.lora_alpha,
                'lora_dropout': self.config.lora_dropout,
                'target_modules': self.config.lora_target_modules,
                'task_type': 'CAUSAL_LM',
                'inference_mode': False
            }
            
            projector_config = {
                'hidden_dim': self.config.projector_hidden_dim,
                'num_tokens': 4
            }
            
            self.model = PurrSightMMLLM(
                llm_model_path=self.config.llm_model_name,
                aligner_weights_path=self.config.phase1_checkpoint_path,
                freeze_encoders=True,
                freeze_projector=False,
                freeze_llm=False,
                lora_config=lora_config,
                projector_config=projector_config
            )
            
            duration = time.time() - start_time
            self.log_result("Model Initialization", True, 
                          "Model created successfully", duration)
            return True
        except Exception as e:
            duration = time.time() - start_time
            self.log_result("Model Initialization", False, str(e), duration)
            return False
    
    def validate_parameter_freezing(self) -> bool:
        """Validate parameter freezing."""
        start_time = time.time()
        try:
            if self.model is None:
                raise RuntimeError("Model not initialized")
            
            frozen_count = 0
            trainable_count = 0
            projector_trainable = 0
            lora_trainable = 0
            
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    trainable_count += 1
                    if 'projector' in name.lower():
                        projector_trainable += 1
                    elif 'lora' in name.lower():
                        lora_trainable += 1
                else:
                    frozen_count += 1
            
            if trainable_count == 0:
                raise RuntimeError("No trainable parameters found")
            
            if projector_trainable == 0:
                raise RuntimeError("No trainable projector parameters found")
            
            duration = time.time() - start_time
            self.log_result("Parameter Freezing", True, 
                          f"{trainable_count} trainable, {frozen_count} frozen", duration)
            return True
        except Exception as e:
            duration = time.time() - start_time
            self.log_result("Parameter Freezing", False, str(e), duration)
            return False
    
    def validate_data_pipeline(self) -> bool:
        """Validate data pipeline."""
        start_time = time.time()
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.llm_model_name,
                trust_remote_code=True,
                padding_side="right"
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.dataset = MultiModalDataset(
                data_dir=self.config.data_dir,
                split='train',
                tokenizer=self.tokenizer,
                max_length=self.config.max_text_length
            )
            
            if len(self.dataset) == 0:
                raise RuntimeError("Dataset is empty")
            
            # Create small subset
            subset_size = min(4, len(self.dataset))
            subset_indices = list(range(subset_size))
            validation_dataset = Subset(self.dataset, subset_indices)
            
            self.dataloader = DataLoader(
                validation_dataset,
                batch_size=min(2, self.config.batch_size),
                shuffle=False,
                num_workers=0,
                collate_fn=multimodal_collate_fn,
                pin_memory=False
            )
            
            # Test loading a batch
            batch = next(iter(self.dataloader))
            
            required_keys = ['image', 'audio', 'text_tokens', 'attention_mask', 'labels']
            for key in required_keys:
                if key not in batch:
                    raise RuntimeError(f"Missing key in batch: {key}")
            
            duration = time.time() - start_time
            self.log_result("Data Pipeline", True, 
                          f"Dataset: {len(self.dataset)} samples, batch loaded", duration)
            return True
        except Exception as e:
            duration = time.time() - start_time
            self.log_result("Data Pipeline", False, str(e), duration)
            return False
    
    def validate_forward_pass(self) -> bool:
        """Validate forward pass."""
        start_time = time.time()
        try:
            if self.model is None or self.dataloader is None:
                raise RuntimeError("Model or dataloader not initialized")
            
            batch = next(iter(self.dataloader))
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = self.model.to(device)
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(
                    image=batch['image'],
                    audio=batch['audio'],
                    text_tokens=batch['text_tokens'],
                    attention_mask=batch['attention_mask']
                )
                
                if not hasattr(outputs, 'logits'):
                    raise RuntimeError("Model output missing 'logits'")
                
                logits = outputs.logits
                if torch.isnan(logits).any():
                    raise RuntimeError("NaN values in output")
                if torch.isinf(logits).any():
                    raise RuntimeError("Inf values in output")
            
            duration = time.time() - start_time
            self.log_result("Forward Pass", True, 
                          f"Output shape: {list(logits.shape)}", duration)
            return True
        except Exception as e:
            duration = time.time() - start_time
            self.log_result("Forward Pass", False, str(e), duration)
            return False
    
    def validate_training_step(self) -> bool:
        """Validate training step."""
        start_time = time.time()
        try:
            if self.model is None or self.dataloader is None:
                raise RuntimeError("Model or dataloader not initialized")
            
            batch = next(iter(self.dataloader))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            self.model.train()
            
            # Store initial parameters
            initial_params = {}
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    initial_params[name] = param.data.clone()
            
            # Forward pass
            outputs = self.model(
                image=batch['image'],
                audio=batch['audio'],
                text_tokens=batch['text_tokens'],
                attention_mask=batch['attention_mask']
            )
            
            # Compute loss (simple cross-entropy)
            logits = outputs.logits
            labels = batch['labels']
            
            # Shift for causal LM
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            
            if torch.isnan(loss) or torch.isinf(loss):
                raise RuntimeError(f"Invalid loss: {loss.item()}")
            
            # Backward pass
            loss.backward()
            
            # Check gradients
            grad_count = 0
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    grad_count += 1
                    if torch.isnan(param.grad).any():
                        raise RuntimeError(f"NaN gradients in {name}")
            
            if grad_count == 0:
                raise RuntimeError("No gradients computed")
            
            # Simple optimizer step
            optimizer = torch.optim.AdamW(
                [p for p in self.model.parameters() if p.requires_grad],
                lr=self.config.learning_rate
            )
            optimizer.step()
            optimizer.zero_grad()
            
            # Check parameters changed
            changed_count = 0
            for name, param in self.model.named_parameters():
                if param.requires_grad and name in initial_params:
                    if not torch.equal(param.data, initial_params[name]):
                        changed_count += 1
            
            if changed_count == 0:
                raise RuntimeError("No parameters were updated")
            
            duration = time.time() - start_time
            self.log_result("Training Step", True, 
                          f"Loss: {loss.item():.4f}, {changed_count} params updated", duration)
            return True
        except Exception as e:
            duration = time.time() - start_time
            self.log_result("Training Step", False, str(e), duration)
            return False
    
    def run_validation(self) -> bool:
        """Run all validation checks."""
        print("=" * 80)
        print("SIMPLE PHASE 2 TRAINING VALIDATION")
        print("=" * 80)
        
        checks = [
            self.validate_configuration,
            self.validate_model_initialization,
            self.validate_parameter_freezing,
            self.validate_data_pipeline,
            self.validate_forward_pass,
            self.validate_training_step,
        ]
        
        all_passed = True
        for check in checks:
            try:
                result = check()
                if not result:
                    all_passed = False
            except Exception as e:
                print(f"❌ FAIL: {check.__name__} - Exception: {e}")
                all_passed = False
        
        print("-" * 80)
        passed_count = sum(1 for _, passed, _, _ in self.results if passed)
        total_count = len(self.results)
        print(f"Results: {passed_count}/{total_count} checks passed")
        
        if all_passed:
            print("🎉 ALL VALIDATION CHECKS PASSED!")
            print("Phase 2 training pipeline is ready for full training.")
        else:
            print("⚠️  VALIDATION FAILED!")
            print("Please fix the failing checks before running full training.")
        
        print("=" * 80)
        return all_passed


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Simple Phase 2 Training Validation")
    parser.add_argument('--config', '-c', type=str, 
                       default='config/validation_config.yaml',
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        if Path(args.config).exists():
            config = load_config(args.config)
            print(f"Loaded configuration from: {args.config}")
        else:
            print(f"Configuration file not found: {args.config}")
            print("Please create a configuration file or check the path.")
            return 1
        
        # Run validation
        validator = SimpleValidator(config)
        success = validator.run_validation()
        
        return 0 if success else 1
        
    except Exception as e:
        print(f"Validation failed with error: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    sys.exit(main())