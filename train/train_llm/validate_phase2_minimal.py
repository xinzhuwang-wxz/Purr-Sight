#!/usr/bin/env python3
"""
Minimal Phase 2 Training Validation Script

A minimal validation script that tests core Phase 2 functionality without
complex dependencies. This validates:

1. Configuration loading
2. Model initialization 
3. Parameter freezing
4. Basic forward pass

Usage:
    python validate_phase2_minimal.py [--config CONFIG_PATH]
"""

import os
import sys
import argparse
import time
import traceback
from pathlib import Path

import torch
import torch.nn as nn

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from train.train_llm.train_llm_conf import TrainingConfig, load_config, validate_config


def validate_configuration(config: TrainingConfig) -> bool:
    """Validate configuration."""
    print("Testing configuration validation...")
    start_time = time.time()
    
    try:
        validate_config(config)
        
        # Check paths exist
        phase1_path = Path(config.phase1_checkpoint_path)
        if not phase1_path.exists():
            raise FileNotFoundError(f"Phase 1 checkpoint not found: {phase1_path}")
        
        data_path = Path(config.data_dir)
        if not data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {data_path}")
        
        duration = time.time() - start_time
        print(f"✅ PASS: Configuration Validation ({duration:.2f}s)")
        print(f"  └─ Config valid, paths exist")
        return True
        
    except Exception as e:
        duration = time.time() - start_time
        print(f"❌ FAIL: Configuration Validation ({duration:.2f}s)")
        print(f"  └─ {str(e)}")
        return False


def validate_model_initialization(config: TrainingConfig) -> tuple:
    """Validate model initialization."""
    print("Testing model initialization...")
    start_time = time.time()
    
    try:
        # Import here to avoid early import issues
        from purrsight.LLM.model import PurrSightMMLLM
        
        lora_config = {
            'enabled': True,
            'r': config.lora_r,
            'lora_alpha': config.lora_alpha,
            'lora_dropout': config.lora_dropout,
            'target_modules': config.lora_target_modules,
            'task_type': 'CAUSAL_LM',
            'inference_mode': False
        }
        
        projector_config = {
            'hidden_dim': config.projector_hidden_dim,
            'num_tokens': 4
        }
        
        model = PurrSightMMLLM(
            llm_model_path=config.llm_model_name,
            aligner_weights_path=config.phase1_checkpoint_path,
            freeze_encoders=True,
            freeze_projector=False,
            freeze_llm=False,
            lora_config=lora_config,
            projector_config=projector_config
        )
        
        duration = time.time() - start_time
        print(f"✅ PASS: Model Initialization ({duration:.2f}s)")
        print(f"  └─ Model created successfully")
        return True, model
        
    except Exception as e:
        duration = time.time() - start_time
        print(f"❌ FAIL: Model Initialization ({duration:.2f}s)")
        print(f"  └─ {str(e)}")
        return False, None


def validate_parameter_freezing(model) -> bool:
    """Validate parameter freezing."""
    print("Testing parameter freezing...")
    start_time = time.time()
    
    try:
        if model is None:
            raise RuntimeError("Model not initialized")
        
        frozen_count = 0
        trainable_count = 0
        projector_trainable = 0
        lora_trainable = 0
        
        for name, param in model.named_parameters():
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
        
        if projector_trainable == 0 and lora_trainable == 0:
            raise RuntimeError("No trainable projector or LoRA parameters found")
        
        duration = time.time() - start_time
        print(f"✅ PASS: Parameter Freezing ({duration:.2f}s)")
        print(f"  └─ {trainable_count} trainable, {frozen_count} frozen")
        print(f"      Projector: {projector_trainable}, LoRA: {lora_trainable}")
        return True
        
    except Exception as e:
        duration = time.time() - start_time
        print(f"❌ FAIL: Parameter Freezing ({duration:.2f}s)")
        print(f"  └─ {str(e)}")
        return False


def validate_basic_forward_pass(model, config: TrainingConfig) -> bool:
    """Validate basic forward pass with dummy data."""
    print("Testing basic forward pass...")
    start_time = time.time()
    
    try:
        if model is None:
            raise RuntimeError("Model not initialized")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        
        # Create dummy inputs
        batch_size = 2
        seq_len = 32
        
        # Dummy image (batch_size, 3, 224, 224)
        dummy_image = torch.randn(batch_size, 3, 224, 224, device=device)
        
        # Dummy audio (batch_size, time_steps, n_mels)
        dummy_audio = torch.randn(batch_size, 100, 128, device=device)
        
        # Dummy text tokens
        dummy_text_tokens = torch.randint(0, 1000, (batch_size, seq_len), device=device)
        dummy_attention_mask = torch.ones(batch_size, seq_len, device=device)
        
        with torch.no_grad():
            outputs = model(
                image=dummy_image,
                audio=dummy_audio,
                text_tokens=dummy_text_tokens,
                attention_mask=dummy_attention_mask
            )
            
            if not hasattr(outputs, 'logits'):
                raise RuntimeError("Model output missing 'logits'")
            
            logits = outputs.logits
            if torch.isnan(logits).any():
                raise RuntimeError("NaN values in output")
            if torch.isinf(logits).any():
                raise RuntimeError("Inf values in output")
            
            expected_shape = (batch_size, seq_len, model.llm.config.vocab_size)
            if logits.shape != expected_shape:
                raise RuntimeError(f"Unexpected output shape: {logits.shape} != {expected_shape}")
        
        duration = time.time() - start_time
        print(f"✅ PASS: Basic Forward Pass ({duration:.2f}s)")
        print(f"  └─ Output shape: {list(logits.shape)}, device: {device}")
        return True
        
    except Exception as e:
        duration = time.time() - start_time
        print(f"❌ FAIL: Basic Forward Pass ({duration:.2f}s)")
        print(f"  └─ {str(e)}")
        return False


def validate_basic_training_step(model, config: TrainingConfig) -> bool:
    """Validate basic training step."""
    print("Testing basic training step...")
    start_time = time.time()
    
    try:
        if model is None:
            raise RuntimeError("Model not initialized")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.train()
        
        # Store initial parameters
        initial_params = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                initial_params[name] = param.data.clone()
        
        # Create dummy inputs
        batch_size = 2
        seq_len = 32
        
        dummy_image = torch.randn(batch_size, 3, 224, 224, device=device)
        dummy_audio = torch.randn(batch_size, 100, 128, device=device)
        dummy_text_tokens = torch.randint(1, 1000, (batch_size, seq_len), device=device)
        dummy_attention_mask = torch.ones(batch_size, seq_len, device=device)
        dummy_labels = dummy_text_tokens.clone()
        
        # Forward pass
        outputs = model(
            image=dummy_image,
            audio=dummy_audio,
            text_tokens=dummy_text_tokens,
            attention_mask=dummy_attention_mask
        )
        
        # Compute loss
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = dummy_labels[..., 1:].contiguous()
        
        loss_fn = nn.CrossEntropyLoss()
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
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_count += 1
                if torch.isnan(param.grad).any():
                    raise RuntimeError(f"NaN gradients in {name}")
        
        if grad_count == 0:
            raise RuntimeError("No gradients computed")
        
        # Simple optimizer step
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=config.learning_rate
        )
        optimizer.step()
        optimizer.zero_grad()
        
        # Check parameters changed
        changed_count = 0
        for name, param in model.named_parameters():
            if param.requires_grad and name in initial_params:
                if not torch.equal(param.data, initial_params[name]):
                    changed_count += 1
        
        if changed_count == 0:
            raise RuntimeError("No parameters were updated")
        
        duration = time.time() - start_time
        print(f"✅ PASS: Basic Training Step ({duration:.2f}s)")
        print(f"  └─ Loss: {loss.item():.4f}, {changed_count} params updated")
        return True
        
    except Exception as e:
        duration = time.time() - start_time
        print(f"❌ FAIL: Basic Training Step ({duration:.2f}s)")
        print(f"  └─ {str(e)}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Minimal Phase 2 Training Validation")
    parser.add_argument('--config', '-c', type=str, 
                       default='config/validation_config.yaml',
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("MINIMAL PHASE 2 TRAINING VALIDATION")
    print("=" * 80)
    
    try:
        # Load configuration
        if Path(args.config).exists():
            config = load_config(args.config)
            print(f"Loaded configuration from: {args.config}")
        else:
            print(f"Configuration file not found: {args.config}")
            print("Please create a configuration file or check the path.")
            return 1
        
        print()
        
        # Run validation checks
        results = []
        
        # 1. Configuration validation
        result = validate_configuration(config)
        results.append(result)
        
        # 2. Model initialization
        result, model = validate_model_initialization(config)
        results.append(result)
        
        if model is not None:
            # 3. Parameter freezing
            result = validate_parameter_freezing(model)
            results.append(result)
            
            # 4. Basic forward pass
            result = validate_basic_forward_pass(model, config)
            results.append(result)
            
            # 5. Basic training step
            result = validate_basic_training_step(model, config)
            results.append(result)
        
        # Summary
        print()
        print("-" * 80)
        passed_count = sum(results)
        total_count = len(results)
        print(f"Results: {passed_count}/{total_count} checks passed")
        
        if passed_count == total_count:
            print("🎉 ALL VALIDATION CHECKS PASSED!")
            print("Phase 2 training pipeline core functionality is working.")
        else:
            print("⚠️  VALIDATION FAILED!")
            print("Please fix the failing checks before running full training.")
        
        print("=" * 80)
        return 0 if passed_count == total_count else 1
        
    except Exception as e:
        print(f"Validation failed with error: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    sys.exit(main())