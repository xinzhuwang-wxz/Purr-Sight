#!/usr/bin/env python3
"""
Phase 2 Training Validation Script - Local Mode

This is a modified version of validate_phase2.py that ensures all model loading
uses local files only and doesn't connect to HuggingFace Hub.

Key modifications:
- Sets TRANSFORMERS_OFFLINE=1 and HF_HUB_OFFLINE=1 environment variables
- Patches AutoModelForCausalLM and AutoTokenizer to use local_files_only=True
- Uses existing Phase 1 checkpoint and local model paths
"""

import os
import sys
from pathlib import Path

# Set offline mode BEFORE importing transformers
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Monkey patch transformers to force local_files_only
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

# Store original methods
_original_model_from_pretrained = AutoModelForCausalLM.from_pretrained
_original_tokenizer_from_pretrained = AutoTokenizer.from_pretrained

def patched_model_from_pretrained(model_name_or_path, *args, **kwargs):
    """Patched version that forces local_files_only=True"""
    kwargs['local_files_only'] = True
    return _original_model_from_pretrained(model_name_or_path, *args, **kwargs)

def patched_tokenizer_from_pretrained(model_name_or_path, *args, **kwargs):
    """Patched version that forces local_files_only=True"""
    kwargs['local_files_only'] = True
    return _original_tokenizer_from_pretrained(model_name_or_path, *args, **kwargs)

# Apply patches
AutoModelForCausalLM.from_pretrained = patched_model_from_pretrained
AutoTokenizer.from_pretrained = patched_tokenizer_from_pretrained

# Now import and run the original validation script
from validate_phase2 import main, create_validation_config, Phase2Validator
from purrsight.utils.logging import logger

def create_local_validation_config():
    """Create validation config with local paths."""
    config = create_validation_config()
    
    # Ensure we use local model path
    config.llm_model_name = "models/Qwen2.5-0.5B-Instruct"
    
    # Find Phase 1 checkpoint
    checkpoint_path = "checkpoints/alignment/5715425b468c42ed9153039b095fca69_20260127_013849/aligner.pt"
    if Path(checkpoint_path).exists():
        config.phase1_checkpoint_path = checkpoint_path
        logger.info(f"Using Phase 1 checkpoint: {checkpoint_path}")
    else:
        logger.error(f"Phase 1 checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    # Use instruction data
    if Path("data/instruction").exists():
        config.data_dir = "data/instruction"
        logger.info(f"Using data directory: {config.data_dir}")
    else:
        logger.error("Data directory not found: data/instruction")
        sys.exit(1)
    
    return config

def main_local():
    """Main entry point for local validation."""
    try:
        print("=" * 80)
        print("PHASE 2 TRAINING VALIDATION - LOCAL MODE")
        print("=" * 80)
        print("Running in offline mode - no HuggingFace connections")
        print("Using local models and Phase 1 checkpoint")
        print("-" * 80)
        
        # Create local validation configuration
        config = create_local_validation_config()
        
        # Run validation in quick mode for faster execution
        validator = Phase2Validator(
            config=config,
            quick_mode=True,  # Use quick mode for faster validation
            verbose=True
        )
        
        success = validator.run_validation()
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        logger.info("Validation interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Validation failed with error: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main_local()