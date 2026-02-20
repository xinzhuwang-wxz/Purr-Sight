# Phase 2 Training Script

This document describes the main training script for Phase 2 of the Purr-Sight multi-modal LLM training pipeline.

## Overview

The `train_phase2.py` script integrates all Phase 2 components to create a complete training pipeline:

- **Phase 1 Checkpoint Loading**: Loads pre-trained aligner weights and freezes parameters
- **LoRA Application**: Applies Low-Rank Adaptation for parameter-efficient LLM fine-tuning
- **Projector Training**: Trains projector modules to transform aligned features to LLM input space
- **Multi-Modal Integration**: Processes image, audio, and text inputs through the complete pipeline
- **PyTorch Lightning**: Uses Lightning framework for robust training with callbacks and logging
- **MLflow Logging**: Tracks experiments, metrics, and hyperparameters
- **Error Handling**: Comprehensive error handling with emergency checkpoint saving

## Usage

### Basic Training

```bash
python train_phase2.py --config config/phase2_example.yaml
```

### Resume Training

```bash
# Resume from latest checkpoint
python train_phase2.py --config config/phase2_example.yaml --resume

# Resume from specific checkpoint
python train_phase2.py --config config/phase2_example.yaml --resume --checkpoint path/to/checkpoint.ckpt
```

### Override Configuration

```bash
# Override specific parameters
python train_phase2.py --config config/phase2_example.yaml \
    --batch-size 16 \
    --learning-rate 1e-4 \
    --num-epochs 20 \
    --num-gpus 2
```

## Configuration

The training script uses YAML configuration files. See `config/phase2_example.yaml` for a complete example.

### Key Configuration Sections

#### Model Configuration
```yaml
phase2:
  llm_model_path: "models/Qwen2.5-0.5B-Instruct"  # Path to LLM model
  adapter_path: "checkpoints/alignment/phase1_final.pt"  # Phase 1 checkpoint
```

#### Training Hyperparameters
```yaml
phase2:
  batch_size: 8
  epochs: 10
  learning_rate: 2e-4
  weight_decay: 0.01
  warmup_ratio: 0.1
```

#### LoRA Configuration
```yaml
phase2:
  lora:
    enabled: true
    r: 16                    # LoRA rank
    lora_alpha: 32          # LoRA scaling factor
    lora_dropout: 0.1       # LoRA dropout
    target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
```

#### Data Configuration
```yaml
phase2:
  data_path: "data/processed"
  max_length: 512
  num_workers: 4
```

## Requirements

### Phase 1 Checkpoint
- Must have completed Phase 1 training
- Checkpoint file should contain aligner weights (encoders + alignment heads)
- Path should point to the actual checkpoint file, not directory

### Data Format
The script expects data in the following structure:
```
data/processed/
├── train.jsonl          # Training data index
├── val.jsonl            # Validation data index (optional)
├── images/              # Image files
├── audio/               # Audio files
└── text/                # Text files
```

Each line in the JSONL files should contain:
```json
{
  "sample_id": "sample_001",
  "image": "images/sample_001.jpg",
  "audio": "audio/sample_001.wav",
  "text": "text/sample_001.txt",
  "split": "train"
}
```

### Dependencies
- PyTorch Lightning
- Transformers (Hugging Face)
- PEFT (for LoRA)
- MLflow (optional, for logging)
- PyTorch with CUDA support (recommended)

## Architecture Integration

The training script integrates the following components:

1. **PurrSightMMLLM**: Complete multi-modal model
2. **CheckpointLoader**: Loads and verifies Phase 1 weights
3. **LoRAManager**: Applies LoRA configuration to LLM
4. **MultiModalDataset**: Handles multi-modal data loading
5. **MultiModalLLMModule**: Lightning module for training
6. **TrainingConfig**: Configuration management

## Training Flow

1. **Configuration Loading**: Load and validate YAML configuration
2. **Model Setup**: 
   - Initialize PurrSightMMLLM
   - Load Phase 1 checkpoint
   - Apply LoRA to LLM
   - Freeze appropriate parameters
3. **Data Setup**:
   - Create training and validation datasets
   - Initialize data loaders with proper collation
4. **Trainer Setup**:
   - Configure PyTorch Lightning trainer
   - Set up MLflow logging
   - Configure callbacks (checkpointing, LR monitoring)
5. **Training Execution**:
   - Run training loop with validation
   - Log metrics and save checkpoints
   - Handle interruptions gracefully

## Error Handling

The script includes comprehensive error handling:

- **Configuration Validation**: Validates all parameters before training
- **File Existence Checks**: Verifies checkpoint and data files exist
- **Graceful Interruption**: Saves emergency checkpoint on SIGINT/SIGTERM
- **Detailed Error Messages**: Provides actionable error information
- **Recovery Support**: Can resume from any saved checkpoint

## Logging and Monitoring

### MLflow Integration
- Experiment tracking with unique run IDs
- Hyperparameter logging
- Metric logging (loss, learning rate, gradient norms)
- Model artifact saving

### Console Logging
- Detailed progress information
- Parameter counts and trainability status
- Training metrics and validation results
- Error messages with full tracebacks

### Checkpointing
- Automatic checkpoint saving every N epochs
- Best model checkpointing based on validation loss
- Emergency checkpoint on interruption
- Checkpoint metadata (epoch, step, metrics)

## Performance Optimization

### Memory Optimization
- Mixed precision training (FP16)
- Gradient checkpointing for LLM
- Efficient data loading with proper batching
- LoRA for parameter-efficient fine-tuning

### Multi-GPU Support
- Distributed Data Parallel (DDP) support
- Automatic GPU detection and configuration
- Distributed sampling for data loading

### Monitoring
- GPU memory usage tracking
- Training speed metrics
- Gradient norm monitoring
- Learning rate scheduling

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Enable gradient checkpointing
   - Use smaller model or lower precision

2. **Phase 1 Checkpoint Not Found**
   - Verify checkpoint path in configuration
   - Ensure Phase 1 training completed successfully
   - Check file permissions

3. **Data Loading Errors**
   - Verify data directory structure
   - Check JSONL file format
   - Ensure all referenced files exist

4. **LoRA Configuration Errors**
   - Verify target modules exist in the model
   - Check LoRA parameters (rank, alpha, dropout)
   - Ensure PEFT library is installed

### Debug Mode
Enable debug logging by setting environment variable:
```bash
export PURRSIGHT_LOG_LEVEL=DEBUG
python train_phase2.py --config config/phase2_example.yaml
```

## Testing

Run the test script to verify installation:
```bash
python test_train_script.py
```

This will test:
- Configuration loading and validation
- Training manager initialization
- Command-line argument parsing
- Checkpoint finding functionality

## Example Training Session

```bash
# 1. Prepare configuration
cp config/phase2_example.yaml config/my_training.yaml
# Edit config/my_training.yaml with your paths

# 2. Start training
python train_phase2.py --config config/my_training.yaml

# 3. Monitor progress
# Check MLflow UI at http://localhost:5000
# Or monitor console output

# 4. Resume if interrupted
python train_phase2.py --config config/my_training.yaml --resume
```

## Output Structure

Training creates the following output structure:
```
outputs/
├── checkpoints/
│   ├── phase2-epoch=01-train_loss=2.3456.ckpt
│   ├── phase2-epoch=02-train_loss=2.1234.ckpt
│   └── last.ckpt
├── logs/
│   └── training.log
└── mlruns/
    └── experiment_id/
        └── run_id/
            ├── metrics/
            ├── params/
            └── artifacts/
```

This comprehensive training script provides a robust foundation for Phase 2 training with proper error handling, logging, and recovery mechanisms.