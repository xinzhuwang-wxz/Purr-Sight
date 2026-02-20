"""Checkpoint Management Utilities for Phase 2 Training.

This module provides comprehensive checkpoint saving and loading utilities for
Phase 2 training, including:
- Complete training state saving (model, optimizer, scheduler)
- Training state restoration with validation
- Checkpoint filename formatting with epoch and metrics
- Emergency checkpoint handling on interrupts
- Round-trip consistency verification
"""

import os
import signal
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional, Union, Callable
from datetime import datetime
import json
import tempfile
import shutil

# Try to import mlflow, but make it optional
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None

from purrsight.utils.logging import logger
from purrsight.config import CHECKPOINT_MONITOR, CHECKPOINT_MONITOR_MODE


class CheckpointManager:
    """Manages checkpoint saving and loading for Phase 2 training.
    
    This class provides utilities for:
    - Saving complete training state (model, optimizer, scheduler, metadata)
    - Loading and restoring training state with validation
    - Formatting checkpoint filenames with epoch and metrics
    - Handling emergency checkpoints on interrupts
    - Verifying checkpoint integrity and round-trip consistency
    
    The checkpoint format includes:
    - model_state_dict: Complete model parameters
    - optimizer_state_dict: Optimizer state for resuming
    - scheduler_state_dict: Learning rate scheduler state
    - epoch: Current training epoch
    - global_step: Global training step counter
    - metrics: Training metrics (loss, learning rate, etc.)
    - config: Training configuration for reproducibility
    - metadata: Additional information (timestamp, versions, etc.)
    """
    
    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        save_top_k: int = 3,
        monitor_metric: str = CHECKPOINT_MONITOR,
        mode: str = CHECKPOINT_MONITOR_MODE,
    ):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to save checkpoints
            save_top_k: Number of best checkpoints to keep (default: 3)
            monitor_metric: Metric to monitor (default: project CHECKPOINT_MONITOR)
            mode: "min" or "max" for metric monitoring
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.save_top_k = save_top_k
        self.monitor_metric = monitor_metric
        self.mode = mode
        
        # Track saved checkpoints for cleanup
        self.saved_checkpoints = []
        self.best_checkpoints = []
        
        # Emergency checkpoint handling
        self._emergency_checkpoint_path = None
        self._original_sigint_handler = None
        self._original_sigterm_handler = None
        
        logger.info(
            f"Initialized CheckpointManager: dir={checkpoint_dir}, "
            f"save_top_k={save_top_k}, monitor={monitor_metric} ({mode})"
        )
    
    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        epoch: int = 0,
        global_step: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        config: Optional[Dict[str, Any]] = None,
        filename: Optional[str] = None,
        is_best: bool = False,
        is_emergency: bool = False
    ) -> str:
        """Save complete training state to checkpoint file.
        
        Args:
            model: Model to save
            optimizer: Optimizer to save
            scheduler: Optional learning rate scheduler
            epoch: Current training epoch
            global_step: Global training step
            metrics: Dictionary of training metrics
            config: Training configuration dictionary
            filename: Optional custom filename (auto-generated if None)
            is_best: Whether this is the best checkpoint so far
            is_emergency: Whether this is an emergency checkpoint
            
        Returns:
            Path to saved checkpoint file
            
        Raises:
            RuntimeError: If checkpoint saving fails
        """
        try:
            # Prepare metrics
            if metrics is None:
                metrics = {}
            
            # Generate filename if not provided
            if filename is None:
                filename = self._generate_checkpoint_filename(
                    epoch=epoch,
                    global_step=global_step,
                    metrics=metrics,
                    is_best=is_best,
                    is_emergency=is_emergency
                )
            
            checkpoint_path = self.checkpoint_dir / filename
            
            # Prepare checkpoint dictionary
            checkpoint = {
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": metrics.copy(),
                "timestamp": datetime.now().isoformat(),
                "pytorch_version": torch.__version__,
            }
            
            # Add scheduler state if available
            if scheduler is not None:
                checkpoint["scheduler_state_dict"] = scheduler.state_dict()
            
            # Add config if provided
            if config is not None:
                checkpoint["config"] = config.copy()
            
            # Add CUDA version if available
            if torch.cuda.is_available():
                checkpoint["cuda_version"] = torch.version.cuda
            
            # Add metadata
            checkpoint["metadata"] = {
                "model_type": type(model).__name__,
                "total_params": sum(p.numel() for p in model.parameters()),
                "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "is_best": is_best,
                "is_emergency": is_emergency,
                "monitor_metric": self.monitor_metric,
                "monitor_value": metrics.get(self.monitor_metric, None)
            }
            
            # Save to temporary file first for atomic write
            temp_path = checkpoint_path.with_suffix('.tmp')
            
            logger.info(f"Saving checkpoint to: {checkpoint_path}")
            torch.save(checkpoint, temp_path)
            
            # Atomic move to final location
            shutil.move(str(temp_path), str(checkpoint_path))
            
            # Update tracking lists
            if not is_emergency:
                self.saved_checkpoints.append(str(checkpoint_path))
                
                if is_best:
                    self.best_checkpoints.append({
                        'path': str(checkpoint_path),
                        'metric_value': metrics.get(self.monitor_metric, float('inf')),
                        'epoch': epoch,
                        'global_step': global_step
                    })
                    
                    # Keep only top-k best checkpoints
                    self._cleanup_best_checkpoints()
            
            # Log to MLflow if available
            try:
                if MLFLOW_AVAILABLE and mlflow.active_run():
                    mlflow.log_artifact(str(checkpoint_path), "checkpoints")
                    mlflow.log_param("latest_checkpoint", str(checkpoint_path))
                    if is_best:
                        mlflow.log_param("best_checkpoint", str(checkpoint_path))
            except Exception as e:
                logger.warning(f"Failed to log checkpoint to MLflow: {e}")
            
            logger.info(
                f"Checkpoint saved successfully: {checkpoint_path.name} "
                f"(epoch={epoch}, step={global_step}, size={checkpoint_path.stat().st_size / 1024 / 1024:.1f}MB)"
            )
            
            return str(checkpoint_path)
            
        except Exception as e:
            error_msg = f"Failed to save checkpoint: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def load_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        strict: bool = True,
        map_location: Optional[Union[str, torch.device]] = None
    ) -> Dict[str, Any]:
        """Load complete training state from checkpoint file.
        
        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load state into
            optimizer: Optional optimizer to load state into
            scheduler: Optional scheduler to load state into
            strict: Whether to strictly enforce state dict key matching
            map_location: Device to map tensors to
            
        Returns:
            Dictionary containing checkpoint metadata and loading info
            
        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            RuntimeError: If checkpoint loading fails
        """
        checkpoint_path = Path(checkpoint_path)
        
        # Verify checkpoint file exists
        if not checkpoint_path.exists():
            error_msg = f"Checkpoint file not found: {checkpoint_path.absolute()}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        
        try:
            # Load checkpoint with appropriate device mapping
            if map_location is None:
                map_location = 'cpu'  # Safe default
            
            checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
            
            # Validate checkpoint structure
            required_keys = ["model_state_dict", "epoch", "global_step"]
            missing_keys = [key for key in required_keys if key not in checkpoint]
            if missing_keys:
                error_msg = f"Checkpoint missing required keys: {missing_keys}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            # Load model state
            if "model_state_dict" in checkpoint:
                load_result = model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
                missing_model_keys = getattr(load_result, 'missing_keys', [])
                unexpected_model_keys = getattr(load_result, 'unexpected_keys', [])
                
                if missing_model_keys:
                    logger.warning(f"Missing model keys: {missing_model_keys}")
                if unexpected_model_keys:
                    logger.warning(f"Unexpected model keys: {unexpected_model_keys}")
            
            # Load optimizer state if provided
            optimizer_loaded = False
            if optimizer is not None and "optimizer_state_dict" in checkpoint:
                try:
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                    optimizer_loaded = True
                    logger.info("Optimizer state loaded successfully")
                except Exception as e:
                    logger.warning(f"Failed to load optimizer state: {e}")
            
            # Load scheduler state if provided
            scheduler_loaded = False
            if scheduler is not None and "scheduler_state_dict" in checkpoint:
                try:
                    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                    scheduler_loaded = True
                    logger.info("Scheduler state loaded successfully")
                except Exception as e:
                    logger.warning(f"Failed to load scheduler state: {e}")
            
            # Prepare metadata
            metadata = {
                'checkpoint_path': str(checkpoint_path.absolute()),
                'epoch': checkpoint.get('epoch', 0),
                'global_step': checkpoint.get('global_step', 0),
                'metrics': checkpoint.get('metrics', {}),
                'config': checkpoint.get('config', {}),
                'timestamp': checkpoint.get('timestamp', None),
                'pytorch_version': checkpoint.get('pytorch_version', None),
                'cuda_version': checkpoint.get('cuda_version', None),
                'metadata': checkpoint.get('metadata', {}),
                'model_keys_loaded': len(checkpoint.get('model_state_dict', {})),
                'missing_model_keys': missing_model_keys,
                'unexpected_model_keys': unexpected_model_keys,
                'optimizer_loaded': optimizer_loaded,
                'scheduler_loaded': scheduler_loaded
            }
            
            logger.info(
                f"Checkpoint loaded successfully: epoch={metadata['epoch']}, "
                f"step={metadata['global_step']}, "
                f"model_keys={metadata['model_keys_loaded']}"
            )
            
            # Log to MLflow if available
            try:
                if MLFLOW_AVAILABLE and mlflow.active_run():
                    mlflow.log_param("loaded_checkpoint", str(checkpoint_path.absolute()))
                    mlflow.log_param("loaded_epoch", metadata['epoch'])
                    mlflow.log_param("loaded_global_step", metadata['global_step'])
            except Exception as e:
                logger.warning(f"Failed to log checkpoint loading to MLflow: {e}")
            
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
    
    def setup_emergency_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        get_current_state: Optional[Callable[[], Dict[str, Any]]] = None
    ) -> None:
        """Setup emergency checkpoint saving on interrupt signals.
        
        Args:
            model: Model to save in emergency
            optimizer: Optimizer to save in emergency
            scheduler: Optional scheduler to save in emergency
            get_current_state: Optional function to get current training state
        """
        self._emergency_checkpoint_path = self.checkpoint_dir / "emergency_checkpoint.pt"
        
        def emergency_handler(signum, frame):
            """Handle interrupt signals by saving emergency checkpoint."""
            logger.info(f"Received signal {signum}, saving emergency checkpoint...")
            
            try:
                # Get current state if function provided
                current_state = {}
                if get_current_state:
                    current_state = get_current_state()
                
                # Save emergency checkpoint
                emergency_path = self.save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=current_state.get('epoch', 0),
                    global_step=current_state.get('global_step', 0),
                    metrics=current_state.get('metrics', {}),
                    config=current_state.get('config', {}),
                    is_emergency=True
                )
                
                logger.info(f"Emergency checkpoint saved to: {emergency_path}")
                
            except Exception as e:
                logger.error(f"Failed to save emergency checkpoint: {e}")
            
            # Restore original handler and re-raise signal
            if signum == signal.SIGINT and self._original_sigint_handler:
                signal.signal(signal.SIGINT, self._original_sigint_handler)
            elif signum == signal.SIGTERM and self._original_sigterm_handler:
                signal.signal(signal.SIGTERM, self._original_sigterm_handler)
            
            # Re-raise the signal
            os.kill(os.getpid(), signum)
        
        # Install signal handlers
        self._original_sigint_handler = signal.signal(signal.SIGINT, emergency_handler)
        self._original_sigterm_handler = signal.signal(signal.SIGTERM, emergency_handler)
        
        logger.info("Emergency checkpoint handler installed for SIGINT and SIGTERM")
    
    def cleanup_emergency_checkpoint(self) -> None:
        """Remove emergency checkpoint handlers and cleanup."""
        # Restore original signal handlers
        if self._original_sigint_handler:
            signal.signal(signal.SIGINT, self._original_sigint_handler)
            self._original_sigint_handler = None
        
        if self._original_sigterm_handler:
            signal.signal(signal.SIGTERM, self._original_sigterm_handler)
            self._original_sigterm_handler = None
        
        logger.info("Emergency checkpoint handlers removed")
    
    def find_latest_checkpoint(self) -> Optional[str]:
        """Find the latest checkpoint in the checkpoint directory.
        
        Returns:
            Path to latest checkpoint or None if no checkpoints found
        """
        if not self.checkpoint_dir.exists():
            return None
        
        # Look for checkpoint files
        checkpoint_files = list(self.checkpoint_dir.glob("*.pt"))
        checkpoint_files.extend(self.checkpoint_dir.glob("*.ckpt"))
        
        if not checkpoint_files:
            return None
        
        # Filter out emergency checkpoints for regular latest search
        regular_checkpoints = [
            f for f in checkpoint_files 
            if "emergency" not in f.name.lower()
        ]
        
        if not regular_checkpoints:
            return None
        
        # Sort by modification time and return the latest
        latest_checkpoint = max(regular_checkpoints, key=lambda p: p.stat().st_mtime)
        return str(latest_checkpoint)
    
    def find_best_checkpoint(self) -> Optional[str]:
        """Find the best checkpoint based on monitored metric.
        
        Returns:
            Path to best checkpoint or None if no checkpoints found
        """
        if not self.best_checkpoints:
            return None
        
        # Sort by metric value
        if self.mode == "min":
            best = min(self.best_checkpoints, key=lambda x: x['metric_value'])
        else:
            best = max(self.best_checkpoints, key=lambda x: x['metric_value'])
        
        return best['path']
    
    def verify_checkpoint_integrity(self, checkpoint_path: Union[str, Path]) -> bool:
        """Verify checkpoint file integrity.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            True if checkpoint is valid, False otherwise
        """
        try:
            checkpoint_path = Path(checkpoint_path)
            
            if not checkpoint_path.exists():
                logger.error(f"Checkpoint file does not exist: {checkpoint_path}")
                return False
            
            # Try to load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            
            # Check required keys
            required_keys = ["model_state_dict", "epoch", "global_step"]
            for key in required_keys:
                if key not in checkpoint:
                    logger.error(f"Checkpoint missing required key: {key}")
                    return False
            
            # Check model state dict is not empty
            if not checkpoint["model_state_dict"]:
                logger.error("Checkpoint has empty model_state_dict")
                return False
            
            # Check epoch and global_step are non-negative
            if checkpoint["epoch"] < 0 or checkpoint["global_step"] < 0:
                logger.error("Checkpoint has negative epoch or global_step")
                return False
            
            logger.info(f"Checkpoint integrity verified: {checkpoint_path.name}")
            return True
            
        except Exception as e:
            logger.error(f"Checkpoint integrity check failed: {e}")
            return False
    
    def _generate_checkpoint_filename(
        self,
        epoch: int,
        global_step: int,
        metrics: Dict[str, float],
        is_best: bool = False,
        is_emergency: bool = False
    ) -> str:
        """Generate checkpoint filename with epoch and metrics.
        
        Args:
            epoch: Training epoch
            global_step: Global training step
            metrics: Training metrics dictionary
            is_best: Whether this is the best checkpoint
            is_emergency: Whether this is an emergency checkpoint
            
        Returns:
            Generated filename string
        """
        if is_emergency:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            return f"emergency_checkpoint_epoch{epoch:03d}_step{global_step}_{timestamp}.pt"
        
        if is_best:
            prefix = "best_"
        else:
            prefix = ""
        
        # Format metrics for filename
        metric_parts = []
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                if key == self.monitor_metric:
                    # Primary metric gets special formatting
                    metric_parts.insert(0, f"{key}{value:.4f}")
                else:
                    # Secondary metrics
                    metric_parts.append(f"{key}{value:.4f}")
        
        metric_str = "_".join(metric_parts) if metric_parts else "no_metrics"
        
        return f"{prefix}checkpoint_epoch{epoch:03d}_step{global_step}_{metric_str}.pt"
    
    def _cleanup_best_checkpoints(self) -> None:
        """Remove old best checkpoints to keep only top-k."""
        if len(self.best_checkpoints) <= self.save_top_k:
            return
        
        # Sort by metric value
        if self.mode == "min":
            self.best_checkpoints.sort(key=lambda x: x['metric_value'])
        else:
            self.best_checkpoints.sort(key=lambda x: x['metric_value'], reverse=True)
        
        # Remove checkpoints beyond top-k
        to_remove = self.best_checkpoints[self.save_top_k:]
        self.best_checkpoints = self.best_checkpoints[:self.save_top_k]
        
        # Delete files
        for checkpoint_info in to_remove:
            try:
                checkpoint_path = Path(checkpoint_info['path'])
                if checkpoint_path.exists():
                    checkpoint_path.unlink()
                    logger.info(f"Removed old checkpoint: {checkpoint_path.name}")
            except Exception as e:
                logger.warning(f"Failed to remove checkpoint {checkpoint_info['path']}: {e}")
    
    def list_checkpoints(self) -> Dict[str, Any]:
        """List all available checkpoints with metadata.
        
        Returns:
            Dictionary containing checkpoint information
        """
        checkpoints = {
            'latest': self.find_latest_checkpoint(),
            'best': self.find_best_checkpoint(),
            'all_checkpoints': [],
            'emergency_checkpoints': []
        }
        
        if not self.checkpoint_dir.exists():
            return checkpoints
        
        # Find all checkpoint files
        checkpoint_files = list(self.checkpoint_dir.glob("*.pt"))
        checkpoint_files.extend(self.checkpoint_dir.glob("*.ckpt"))
        
        for checkpoint_path in checkpoint_files:
            try:
                # Load checkpoint metadata
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                
                checkpoint_info = {
                    'path': str(checkpoint_path),
                    'filename': checkpoint_path.name,
                    'size_mb': checkpoint_path.stat().st_size / 1024 / 1024,
                    'modified': datetime.fromtimestamp(checkpoint_path.stat().st_mtime).isoformat(),
                    'epoch': checkpoint.get('epoch', 0),
                    'global_step': checkpoint.get('global_step', 0),
                    'metrics': checkpoint.get('metrics', {}),
                    'is_emergency': checkpoint.get('metadata', {}).get('is_emergency', False)
                }
                
                if checkpoint_info['is_emergency']:
                    checkpoints['emergency_checkpoints'].append(checkpoint_info)
                else:
                    checkpoints['all_checkpoints'].append(checkpoint_info)
                    
            except Exception as e:
                logger.warning(f"Failed to read checkpoint metadata from {checkpoint_path}: {e}")
        
        # Sort by global step
        checkpoints['all_checkpoints'].sort(key=lambda x: x['global_step'])
        checkpoints['emergency_checkpoints'].sort(key=lambda x: x['global_step'])
        
        return checkpoints


def save_checkpoint(
    checkpoint_path: Union[str, Path],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any] = None,
    epoch: int = 0,
    global_step: int = 0,
    metrics: Optional[Dict[str, float]] = None,
    config: Optional[Dict[str, Any]] = None
) -> None:
    """Standalone function to save a checkpoint.
    
    This is a convenience function that creates a temporary CheckpointManager
    and saves a single checkpoint. For more advanced checkpoint management,
    use the CheckpointManager class directly.
    
    Args:
        checkpoint_path: Path where to save the checkpoint
        model: Model to save
        optimizer: Optimizer to save
        scheduler: Optional learning rate scheduler
        epoch: Current training epoch
        global_step: Global training step
        metrics: Dictionary of training metrics
        config: Training configuration dictionary
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_dir = checkpoint_path.parent
    filename = checkpoint_path.name
    
    # Create temporary checkpoint manager
    manager = CheckpointManager(checkpoint_dir)
    
    # Save checkpoint
    manager.save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=epoch,
        global_step=global_step,
        metrics=metrics,
        config=config,
        filename=filename
    )


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    strict: bool = True,
    map_location: Optional[Union[str, torch.device]] = None
) -> Dict[str, Any]:
    """Standalone function to load a checkpoint.
    
    This is a convenience function that creates a temporary CheckpointManager
    and loads a single checkpoint. For more advanced checkpoint management,
    use the CheckpointManager class directly.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        strict: Whether to strictly enforce state dict key matching
        map_location: Device to map tensors to
        
    Returns:
        Dictionary containing checkpoint metadata and loading info
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_dir = checkpoint_path.parent
    
    # Create temporary checkpoint manager
    manager = CheckpointManager(checkpoint_dir)
    
    # Load checkpoint
    return manager.load_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        strict=strict,
        map_location=map_location
    )