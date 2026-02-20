"""Checkpoint Loader for Phase 2 Training.

This module provides utilities for loading Phase 1 aligner checkpoints,
verifying weight integrity, and freezing parameters for Phase 2 training.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional

# Try to import mlflow, but make it optional
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None

from purrsight.utils.logging import logger


class CheckpointLoader:
    """Handles loading and verification of Phase 1 checkpoints for Phase 2 training.
    
    This class provides methods to:
    - Load Phase 1 checkpoint files
    - Verify that all required aligner components are present
    - Freeze aligner parameters to prevent updates during Phase 2
    - Log checkpoint metadata to MLflow
    """
    
    @staticmethod
    def load_phase1_checkpoint(
        checkpoint_path: str,
        model: nn.Module,
        strict: bool = True
    ) -> Dict[str, Any]:
        """Load Phase 1 checkpoint into model.
        
        This function loads a Phase 1 checkpoint file and loads the weights into
        the provided model. It handles different checkpoint formats (raw state_dict
        or wrapped in a dictionary) and provides detailed error messages on failure.
        
        Args:
            checkpoint_path: Path to Phase 1 checkpoint file
            model: Model instance to load weights into (should have aligner attribute)
            strict: Whether to strictly enforce key matching (default: True)
            
        Returns:
            Dictionary containing checkpoint metadata including:
                - checkpoint_path: Path to the loaded checkpoint
                - epoch: Training epoch (if available)
                - global_step: Training step (if available)
                - keys_loaded: Number of keys successfully loaded
                - missing_keys: List of keys missing from checkpoint
                - unexpected_keys: List of unexpected keys in checkpoint
            
        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            RuntimeError: If checkpoint is corrupted or incompatible
        """
        checkpoint_path = Path(checkpoint_path)
        
        # Verify checkpoint file exists
        if not checkpoint_path.exists():
            error_msg = f"Checkpoint file not found: {checkpoint_path.absolute()}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        logger.info(f"Loading Phase 1 checkpoint from: {checkpoint_path}")
        
        try:
            # Load checkpoint with CPU mapping for compatibility
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Extract state_dict (handle different checkpoint formats)
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                    epoch = checkpoint.get('epoch', None)
                    global_step = checkpoint.get('global_step', None)
                else:
                    # Assume the entire dict is the state_dict
                    state_dict = checkpoint
                    epoch = None
                    global_step = None
            else:
                error_msg = f"Invalid checkpoint format. Expected dict, got {type(checkpoint)}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            # Verify aligner weights are present
            if not CheckpointLoader.verify_aligner_weights(state_dict):
                error_msg = (
                    f"Checkpoint {checkpoint_path} is missing required aligner components. "
                    "Expected keys for: image_encoder, audio_encoder, text_encoder, aligner"
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            # Load weights into the aligner submodule
            # Phase 1 saves aligner.state_dict(), so keys don't have "aligner." prefix
            # We load directly into the aligner submodule to match keys
            if hasattr(model, 'aligner'):
                load_result = model.aligner.load_state_dict(state_dict, strict=strict)
                missing_keys = load_result.missing_keys if hasattr(load_result, 'missing_keys') else []
                unexpected_keys = load_result.unexpected_keys if hasattr(load_result, 'unexpected_keys') else []
            else:
                error_msg = "Model does not have 'aligner' attribute. Cannot load checkpoint."
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            # Prepare metadata
            metadata = {
                'checkpoint_path': str(checkpoint_path.absolute()),
                'epoch': epoch,
                'global_step': global_step,
                'keys_loaded': len(state_dict),
                'missing_keys': missing_keys,
                'unexpected_keys': unexpected_keys
            }
            
            # Log success
            logger.info(
                f"Successfully loaded Phase 1 checkpoint: "
                f"{len(state_dict)} keys loaded, "
                f"{len(missing_keys)} missing, "
                f"{len(unexpected_keys)} unexpected"
            )
            
            if missing_keys:
                logger.warning(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys: {unexpected_keys}")
            
            # Log to MLflow if active run exists
            try:
                if MLFLOW_AVAILABLE and mlflow.active_run():
                    mlflow.log_param("phase1_checkpoint_path", str(checkpoint_path.absolute()))
                    mlflow.log_param("phase1_checkpoint_keys", len(state_dict))
                    if epoch is not None:
                        mlflow.log_param("phase1_checkpoint_epoch", epoch)
                    if global_step is not None:
                        mlflow.log_param("phase1_checkpoint_step", global_step)
                    logger.info("Logged checkpoint metadata to MLflow")
            except Exception as e:
                logger.warning(f"Failed to log checkpoint metadata to MLflow: {e}")
            
            return metadata
            
        except FileNotFoundError:
            # Re-raise FileNotFoundError as-is
            raise
        except RuntimeError:
            # Re-raise RuntimeError as-is
            raise
        except Exception as e:
            # Wrap other exceptions in RuntimeError with context
            error_msg = f"Failed to load checkpoint from {checkpoint_path}: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    @staticmethod
    def verify_aligner_weights(state_dict: Dict[str, torch.Tensor]) -> bool:
        """Verify that all required aligner components are present.
        
        Checks for the presence of key components from Phase 1 training:
        - projection_heads (for image, audio, text modalities)
        - temperature parameter (if using temperature scaling)
        
        Args:
            state_dict: Model state dictionary to verify
            
        Returns:
            True if all required components are present, False otherwise
        """
        required_prefixes = [
            'projection_heads',  # Aligner projection heads for each modality
        ]
        
        # Check for at least one key with each required prefix
        for prefix in required_prefixes:
            has_prefix = any(key.startswith(prefix) for key in state_dict.keys())
            if not has_prefix:
                logger.error(f"Missing required component: {prefix}")
                return False
        
        # Additional check: verify we have projection heads for expected modalities
        expected_modalities = ['image', 'audio', 'text']
        for modality in expected_modalities:
            modality_key = f'projection_heads.{modality}'
            has_modality = any(modality_key in key for key in state_dict.keys())
            if not has_modality:
                logger.warning(f"Missing projection head for modality: {modality}")
                # Don't fail, just warn - some checkpoints might not have all modalities
        
        logger.info("Aligner weight verification passed")
        return True
    
    @staticmethod
    def freeze_aligner_parameters(model: nn.Module) -> None:
        """Freeze all aligner parameters (encoders and alignment heads).
        
        Sets requires_grad=False for all parameters in:
        - image_encoder
        - audio_encoder
        - text_encoder (if present)
        - aligner
        
        This prevents these parameters from being updated during Phase 2 training,
        preserving the alignment learned in Phase 1.
        
        Args:
            model: Model instance with encoder and aligner attributes
        """
        frozen_count = 0
        frozen_modules = []
        
        # Freeze image encoder
        if hasattr(model, 'image_encoder'):
            model.image_encoder.eval()
            for param in model.image_encoder.parameters():
                param.requires_grad = False
                frozen_count += 1
            frozen_modules.append('image_encoder')
            logger.info("Froze image_encoder parameters")
        
        # Freeze audio encoder
        if hasattr(model, 'audio_encoder'):
            model.audio_encoder.eval()
            for param in model.audio_encoder.parameters():
                param.requires_grad = False
                frozen_count += 1
            frozen_modules.append('audio_encoder')
            logger.info("Froze audio_encoder parameters")
        
        # Freeze text encoder (if present)
        if hasattr(model, 'text_encoder'):
            model.text_encoder.eval()
            for param in model.text_encoder.parameters():
                param.requires_grad = False
                frozen_count += 1
            frozen_modules.append('text_encoder')
            logger.info("Froze text_encoder parameters")
        
        # Freeze aligner
        if hasattr(model, 'aligner'):
            model.aligner.eval()
            for param in model.aligner.parameters():
                param.requires_grad = False
                frozen_count += 1
            frozen_modules.append('aligner')
            logger.info("Froze aligner parameters")
        
        logger.info(
            f"Frozen {frozen_count} parameters across {len(frozen_modules)} modules: "
            f"{', '.join(frozen_modules)}"
        )
        
        # Log to MLflow if active run exists
        try:
            if MLFLOW_AVAILABLE and mlflow.active_run():
                mlflow.log_param("frozen_modules", ','.join(frozen_modules))
                mlflow.log_param("frozen_parameter_count", frozen_count)
                logger.info("Logged parameter freezing info to MLflow")
        except Exception as e:
            logger.warning(f"Failed to log freezing info to MLflow: {e}")
