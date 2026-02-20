"""
LLM Training Configuration Module
"""

import yaml
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Union
from pathlib import Path

from purrsight.config import CHECKPOINTS_DIR, MLFLOW_TRACKING_URI

@dataclass
class TrainingConfig:
    """
    Configuration for Phase 2: Multimodal Instruction Tuning
    
    This dataclass contains all configuration parameters needed for Phase 2 training,
    including model paths, architecture settings, training hyperparameters, and
    infrastructure settings.
    """
    
    # Model paths (required)
    phase1_checkpoint_path: str
    llm_model_name: str = "Qwen/Qwen2.5-0.5B"
    
    # Architecture
    aligner_dim: int = 512
    llm_hidden_dim: int = 896  # Qwen2.5-0.5B hidden size
    projector_hidden_dim: int = 2048
    projector_num_layers: int = 2
    
    # LoRA configuration
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    
    # Training hyperparameters
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    batch_size: int = 8
    num_epochs: int = 10
    warmup_steps: int = 500
    gradient_clip_val: float = 1.0
    
    # Data
    data_dir: str = "data/processed"
    max_text_length: int = 512
    num_workers: int = 4
    
    # Logging and checkpointing
    log_every_n_steps: int = 10
    save_every_n_epochs: int = 1
    checkpoint_dir: str = "checkpoints/phase2"
    mlflow_experiment_name: str = "purrsight-phase2"
    
    # Distributed training
    num_gpus: int = 1
    distributed_backend: str = "nccl"
    
    # Environment
    seed: int = 42
    device: str = "auto"
    output_dir: str = "outputs"
    mlflow_tracking_uri: Optional[str] = None
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        if self.mlflow_tracking_uri is None:
            self.mlflow_tracking_uri = MLFLOW_TRACKING_URI
            
        # Ensure output dir exists
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)


def validate_config(config: Union[TrainingConfig, Dict[str, Any]]) -> None:
    """
    Validate training configuration for required fields and value constraints.
    
    Args:
        config: TrainingConfig instance or dictionary to validate
        
    Raises:
        ValueError: If configuration is invalid with descriptive error message
        FileNotFoundError: If required file paths don't exist
    """
    if isinstance(config, dict):
        config_dict = config
    else:
        config_dict = config.to_dict()
    
    errors = []
    
    # Check required fields
    required_fields = [
        "phase1_checkpoint_path",
        "llm_model_name", 
        "data_dir"
    ]
    
    for field in required_fields:
        if field not in config_dict or config_dict[field] is None:
            errors.append(f"Missing required field: {field}")
    
    # Validate value constraints
    if "learning_rate" in config_dict and config_dict["learning_rate"] is not None:
        try:
            lr = float(config_dict["learning_rate"])
            if lr <= 0:
                errors.append(f"learning_rate must be positive, got {lr}")
        except (ValueError, TypeError):
            errors.append(f"learning_rate must be a number, got {config_dict['learning_rate']}")
    
    if "batch_size" in config_dict and config_dict["batch_size"] is not None:
        try:
            bs = int(config_dict["batch_size"])
            if bs <= 0:
                errors.append(f"batch_size must be positive, got {bs}")
        except (ValueError, TypeError):
            errors.append(f"batch_size must be an integer, got {config_dict['batch_size']}")
    
    if "num_epochs" in config_dict and config_dict["num_epochs"] is not None:
        try:
            epochs = int(config_dict["num_epochs"])
            if epochs <= 0:
                errors.append(f"num_epochs must be positive, got {epochs}")
        except (ValueError, TypeError):
            errors.append(f"num_epochs must be an integer, got {config_dict['num_epochs']}")
    
    if "weight_decay" in config_dict and config_dict["weight_decay"] is not None:
        try:
            wd = float(config_dict["weight_decay"])
            if wd < 0:
                errors.append(f"weight_decay must be non-negative, got {wd}")
        except (ValueError, TypeError):
            errors.append(f"weight_decay must be a number, got {config_dict['weight_decay']}")
    
    if "lora_r" in config_dict and config_dict["lora_r"] is not None:
        try:
            r = int(config_dict["lora_r"])
            if r <= 0:
                errors.append(f"lora_r must be positive, got {r}")
        except (ValueError, TypeError):
            errors.append(f"lora_r must be an integer, got {config_dict['lora_r']}")
    
    if "lora_alpha" in config_dict and config_dict["lora_alpha"] is not None:
        try:
            alpha = float(config_dict["lora_alpha"])
            if alpha <= 0:
                errors.append(f"lora_alpha must be positive, got {alpha}")
        except (ValueError, TypeError):
            errors.append(f"lora_alpha must be a number, got {config_dict['lora_alpha']}")
    
    if "lora_dropout" in config_dict and config_dict["lora_dropout"] is not None:
        try:
            dropout = float(config_dict["lora_dropout"])
            if not (0 <= dropout <= 1):
                errors.append(f"lora_dropout must be between 0 and 1, got {dropout}")
        except (ValueError, TypeError):
            errors.append(f"lora_dropout must be a number, got {config_dict['lora_dropout']}")
    
    if "gradient_clip_val" in config_dict and config_dict["gradient_clip_val"] is not None:
        try:
            clip_val = float(config_dict["gradient_clip_val"])
            if clip_val <= 0:
                errors.append(f"gradient_clip_val must be positive, got {clip_val}")
        except (ValueError, TypeError):
            errors.append(f"gradient_clip_val must be a number, got {config_dict['gradient_clip_val']}")
    
    if "num_workers" in config_dict and config_dict["num_workers"] is not None:
        try:
            workers = int(config_dict["num_workers"])
            if workers < 0:
                errors.append(f"num_workers must be non-negative, got {workers}")
        except (ValueError, TypeError):
            errors.append(f"num_workers must be an integer, got {config_dict['num_workers']}")
    
    if "max_text_length" in config_dict and config_dict["max_text_length"] is not None:
        try:
            max_len = int(config_dict["max_text_length"])
            if max_len <= 0:
                errors.append(f"max_text_length must be positive, got {max_len}")
        except (ValueError, TypeError):
            errors.append(f"max_text_length must be an integer, got {config_dict['max_text_length']}")
    
    if "warmup_steps" in config_dict and config_dict["warmup_steps"] is not None:
        try:
            warmup = int(config_dict["warmup_steps"])
            if warmup < 0:
                errors.append(f"warmup_steps must be non-negative, got {warmup}")
        except (ValueError, TypeError):
            errors.append(f"warmup_steps must be an integer, got {config_dict['warmup_steps']}")
    
    if "log_every_n_steps" in config_dict and config_dict["log_every_n_steps"] is not None:
        try:
            log_steps = int(config_dict["log_every_n_steps"])
            if log_steps <= 0:
                errors.append(f"log_every_n_steps must be positive, got {log_steps}")
        except (ValueError, TypeError):
            errors.append(f"log_every_n_steps must be an integer, got {config_dict['log_every_n_steps']}")
    
    if "save_every_n_epochs" in config_dict and config_dict["save_every_n_epochs"] is not None:
        try:
            save_epochs = int(config_dict["save_every_n_epochs"])
            if save_epochs <= 0:
                errors.append(f"save_every_n_epochs must be positive, got {save_epochs}")
        except (ValueError, TypeError):
            errors.append(f"save_every_n_epochs must be an integer, got {config_dict['save_every_n_epochs']}")
    
    if "num_gpus" in config_dict and config_dict["num_gpus"] is not None:
        try:
            gpus = int(config_dict["num_gpus"])
            if gpus <= 0:
                errors.append(f"num_gpus must be positive, got {gpus}")
        except (ValueError, TypeError):
            errors.append(f"num_gpus must be an integer, got {config_dict['num_gpus']}")
    
    # Validate LoRA target modules
    if "lora_target_modules" in config_dict:
        modules = config_dict["lora_target_modules"]
        if not isinstance(modules, list):
            errors.append(f"lora_target_modules must be a list, got {type(modules)}")
        elif len(modules) == 0:
            errors.append("lora_target_modules cannot be empty")
        elif not all(isinstance(m, str) for m in modules):
            errors.append("All lora_target_modules must be strings")
    
    # Validate file paths exist (if they're provided and not None)
    if "phase1_checkpoint_path" in config_dict and config_dict["phase1_checkpoint_path"]:
        checkpoint_path = Path(config_dict["phase1_checkpoint_path"])
        if not checkpoint_path.exists():
            errors.append(f"Phase 1 checkpoint file not found: {checkpoint_path}")
    
    if "data_dir" in config_dict and config_dict["data_dir"]:
        data_path = Path(config_dict["data_dir"])
        if not data_path.exists():
            errors.append(f"Data directory not found: {data_path}")
    
    # Raise error if any validation failures
    if errors:
        error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in errors)
        raise ValueError(error_msg)


def load_config(config_path: Union[str, Path]) -> TrainingConfig:
    """
    Load and validate training configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Validated TrainingConfig instance
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
        ValueError: If configuration validation fails
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Failed to parse YAML configuration: {e}")
    
    # Handle nested phase2 configuration if present (like in existing config file)
    if "phase2" in config_dict:
        phase2_config = config_dict["phase2"]
        
        # Map existing config structure to TrainingConfig fields
        mapped_config = {}
        
        # Direct mappings
        field_mappings = {
            "llm_model_path": "llm_model_name",
            "adapter_path": "phase1_checkpoint_path",
            "data_path": "data_dir",
            "max_length": "max_text_length",
            "batch_size": "batch_size",
            "epochs": "num_epochs",
            "learning_rate": "learning_rate",
            "weight_decay": "weight_decay",
            "warmup_ratio": "warmup_steps",  # Will need conversion
            "num_workers": "num_workers",
            "experiment_name": "mlflow_experiment_name"
        }
        
        for old_key, new_key in field_mappings.items():
            if old_key in phase2_config:
                mapped_config[new_key] = phase2_config[old_key]
        
        # Handle nested configurations
        if "lora" in phase2_config:
            lora_config = phase2_config["lora"]
            if "r" in lora_config:
                mapped_config["lora_r"] = lora_config["r"]
            if "lora_alpha" in lora_config:
                mapped_config["lora_alpha"] = lora_config["lora_alpha"]
            if "lora_dropout" in lora_config:
                mapped_config["lora_dropout"] = lora_config["lora_dropout"]
            if "target_modules" in lora_config:
                mapped_config["lora_target_modules"] = lora_config["target_modules"]
        
        if "projector" in phase2_config:
            projector_config = phase2_config["projector"]
            if "hidden_dim" in projector_config:
                mapped_config["projector_hidden_dim"] = projector_config["hidden_dim"]
        
        # Handle common settings if present
        if "common" in config_dict:
            common_config = config_dict["common"]
            if "seed" in common_config:
                mapped_config["seed"] = common_config["seed"]
            if "output_dir" in common_config:
                mapped_config["output_dir"] = common_config["output_dir"]
            if "log_every" in common_config:
                mapped_config["log_every_n_steps"] = common_config["log_every"]
            if "save_every" in common_config:
                mapped_config["save_every_n_epochs"] = common_config["save_every"]
        
        config_dict = mapped_config
    
    # Apply default values for missing optional fields
    defaults = {
        "llm_model_name": "Qwen/Qwen2.5-0.5B",
        "aligner_dim": 512,
        "llm_hidden_dim": 896,
        "projector_hidden_dim": 2048,
        "projector_num_layers": 2,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "lora_target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
        "learning_rate": 2e-4,
        "weight_decay": 0.01,
        "batch_size": 8,
        "num_epochs": 10,
        "warmup_steps": 500,
        "gradient_clip_val": 1.0,
        "data_dir": "data/processed",
        "max_text_length": 512,
        "num_workers": 4,
        "log_every_n_steps": 10,
        "save_every_n_epochs": 1,
        "checkpoint_dir": "checkpoints/phase2",
        "mlflow_experiment_name": "purrsight-phase2",
        "num_gpus": 1,
        "distributed_backend": "nccl",
        "seed": 42,
        "device": "auto",
        "output_dir": "outputs"
    }
    
    # Apply defaults for missing fields
    for key, default_value in defaults.items():
        if key not in config_dict:
            config_dict[key] = default_value
    
    # Convert warmup_ratio to warmup_steps if needed
    if "warmup_ratio" in config_dict and "warmup_steps" not in config_dict:
        # Estimate total steps for warmup calculation
        # This is approximate - in practice would be calculated from dataset size
        estimated_steps_per_epoch = 1000  # Placeholder
        total_steps = config_dict.get("num_epochs", 10) * estimated_steps_per_epoch
        config_dict["warmup_steps"] = int(config_dict["warmup_ratio"] * total_steps)
        del config_dict["warmup_ratio"]
    
    # Validate the configuration
    validate_config(config_dict)
    
    # Create TrainingConfig instance
    try:
        config = TrainingConfig(**config_dict)
    except TypeError as e:
        raise ValueError(f"Failed to create TrainingConfig: {e}")
    
    return config


# Legacy configuration class for backward compatibility
@dataclass
class LLMConfig:
    """
    Legacy configuration for Phase 2: Multimodal Instruction Tuning
    
    This class is maintained for backward compatibility with existing code.
    New code should use TrainingConfig instead.
    """
    
    # Model
    llm_model_path: str
    data_path: str  # Move data_path here to avoid non-default argument error
    adapter_path: Optional[str] = None
    
    # Data
    max_length: int = 2048
    
    # Training
    batch_size: int = 4
    epochs: int = 3
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    gradient_accumulation_steps: int = 4
    num_workers: int = 4
    
    # LoRA
    lora: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "v_proj"]
    })
    
    # Projector
    projector: Dict[str, Any] = field(default_factory=lambda: {
        "hidden_dim": 2048,
        "num_tokens": 4
    })
    
    # Freezing
    freeze_encoders: bool = True
    freeze_projector: bool = False
    freeze_llm: bool = False
    
    # Environment
    seed: int = 42
    device: str = "auto"
    output_dir: str = "outputs"
    experiment_name: str = "instruction_tuning_phase2"
    mlflow_tracking_uri: Optional[str] = None
    
    # Logging
    log_every: int = 50
    save_every: int = 1
    
    def __post_init__(self):
        if self.mlflow_tracking_uri is None:
            self.mlflow_tracking_uri = MLFLOW_TRACKING_URI
            
        # Ensure output dir exists
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


def create_default_config() -> TrainingConfig:
    """
    Create a TrainingConfig with all default values.
    
    Returns:
        TrainingConfig instance with default values
    """
    return TrainingConfig(
        phase1_checkpoint_path="checkpoints/alignment/phase1_final.pt",
        data_dir="data/processed"
    )


def save_config(config: TrainingConfig, config_path: Union[str, Path]) -> None:
    """
    Save training configuration to YAML file.
    
    Args:
        config: TrainingConfig instance to save
        config_path: Path where to save the configuration
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, indent=2)