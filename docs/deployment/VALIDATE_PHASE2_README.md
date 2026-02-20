# Phase 2 Training Validation Script

## Overview

The `validate_phase2.py` script performs comprehensive validation of the Phase 2 training pipeline to ensure all components work correctly before running full training. It executes a short training session (1-2 epochs) with a small dataset to verify system integration.

## Features

### Validation Checks

The script performs the following validation checks:

1. **Configuration Validation** - Validates training configuration parameters and file paths
2. **Model Initialization** - Tests model creation and Phase 1 checkpoint loading
3. **Parameter Freezing** - Verifies aligner parameters are frozen and projectors are trainable
4. **Data Pipeline** - Tests data loading, preprocessing, and batch creation
5. **Forward Pass** - Validates multi-modal forward pass execution
6. **Training Step** - Tests loss computation, backpropagation, and parameter updates
7. **Checkpoint Saving** - Verifies checkpoint saving and loading functionality
8. **MLflow Logging** - Tests experiment tracking and metric logging
9. **Short Training Run** - Executes complete training loop for 1-2 epochs

### Validation Results

Each check provides:
- ✅ **PASS** or ❌ **FAIL** status
- Execution time
- Detailed error messages for failures
- Additional details for successful checks

## Usage

### Basic Usage

```bash
# Run validation with default settings
python validate_phase2.py

# Run quick validation (1 epoch, minimal logging)
python validate_phase2.py --quick

# Run with verbose logging
python validate_phase2.py --verbose

# Use custom configuration
python validate_phase2.py --config config/validation_config.yaml
```

### Command Line Options

- `--config, -c`: Path to training configuration YAML file
- `--quick, -q`: Run quick validation (1 epoch, minimal logging)
- `--verbose, -v`: Enable verbose logging

### Configuration

The script can use any training configuration file. For validation, it automatically:

- Reduces batch size to 2-4 for faster execution
- Limits training to 1-2 epochs
- Uses temporary directories for outputs
- Enables frequent logging for validation
- Uses minimal warmup steps

A sample validation configuration is provided in `config/validation_config.yaml`.

## Requirements

### Prerequisites

1. **Phase 1 Checkpoint**: A trained Phase 1 aligner checkpoint must be available
2. **Training Data**: Multi-modal training data in the expected format
3. **Dependencies**: All Phase 2 training dependencies must be installed

### Automatic Detection

The script automatically detects:
- Available Phase 1 checkpoints in `checkpoints/alignment/`
- Training data directories (`data/instruction`, `data/preprocessed`, etc.)
- GPU availability and configures accordingly

## Output

### Console Output

The script provides real-time feedback:

```
================================================================================
PHASE 2 TRAINING VALIDATION
================================================================================
Quick mode: False
Verbose mode: False
Config file: config/validation_config.yaml
--------------------------------------------------------------------------------
✅ PASS: Configuration Validation (0.05s)
✅ PASS: Model Initialization (2.34s)
✅ PASS: Parameter Freezing (0.12s)
✅ PASS: Data Pipeline (1.45s)
✅ PASS: Forward Pass (0.89s)
✅ PASS: Training Step (1.23s)
✅ PASS: Checkpoint Saving (0.67s)
✅ PASS: MLflow Logging (0.34s)
✅ PASS: Short Training Run (45.67s)

================================================================================
PHASE 2 TRAINING VALIDATION SUMMARY
================================================================================
✅ PASS: Configuration Validation (0.05s)
✅ PASS: Model Initialization (2.34s)
✅ PASS: Parameter Freezing (0.12s)
✅ PASS: Data Pipeline (1.45s)
✅ PASS: Forward Pass (0.89s)
✅ PASS: Training Step (1.23s)
✅ PASS: Checkpoint Saving (0.67s)
✅ PASS: MLflow Logging (0.34s)
✅ PASS: Short Training Run (45.67s)
--------------------------------------------------------------------------------
Results: 9/9 checks passed
Total time: 52.76s
🎉 ALL VALIDATION CHECKS PASSED!
Phase 2 training pipeline is ready for full training.
================================================================================
```

### Temporary Files

The script creates temporary directories for validation:
- Checkpoints: `/tmp/phase2_validation_*/checkpoints/`
- Outputs: `/tmp/phase2_validation_*/outputs/`

These are automatically cleaned up after validation.

## Troubleshooting

### Common Issues

1. **Phase 1 Checkpoint Not Found**
   - Ensure Phase 1 training has been completed
   - Check that checkpoint files exist in `checkpoints/alignment/`
   - Verify the checkpoint path in your configuration

2. **Data Directory Not Found**
   - Ensure training data has been prepared
   - Check that data files exist in the specified directory
   - Verify the data directory path in your configuration

3. **Model Download Issues**
   - Ensure internet connection for downloading Qwen2.5-0.5B
   - Consider using a local model path if available
   - Check Hugging Face authentication if needed

4. **GPU Memory Issues**
   - Use `--quick` mode to reduce memory usage
   - Reduce batch size in configuration
   - Use CPU-only mode if necessary

5. **MLflow Connection Issues**
   - MLflow logging is optional and will be skipped if unavailable
   - Check MLflow server configuration if using remote tracking
   - Ensure MLflow is installed if logging is required

### Exit Codes

- `0`: All validation checks passed
- `1`: One or more validation checks failed
- `130`: Validation interrupted by user (Ctrl+C)

## Integration

### CI/CD Pipeline

The validation script can be integrated into CI/CD pipelines:

```bash
# Run validation as part of testing
python validate_phase2.py --quick
if [ $? -eq 0 ]; then
    echo "Validation passed, proceeding with full training"
    python train_phase2.py --config config/train_config.yaml
else
    echo "Validation failed, aborting training"
    exit 1
fi
```

### Pre-Training Check

Always run validation before starting full training:

```bash
# Validate pipeline
python validate_phase2.py --config config/train_config.yaml

# If validation passes, run full training
python train_phase2.py --config config/train_config.yaml
```

## Performance

### Expected Execution Times

- **Quick Mode**: 30-60 seconds
- **Standard Mode**: 2-5 minutes
- **Verbose Mode**: 3-6 minutes

### Resource Usage

- **GPU Memory**: 2-4 GB (depending on model and batch size)
- **CPU Memory**: 1-2 GB
- **Disk Space**: 100-500 MB (temporary files)

## Validation Checklist

The script validates the following requirements from the Phase 2 specification:

- ✅ **Requirement 1**: Phase 1 Checkpoint Loading
- ✅ **Requirement 2**: Projector Initialization and Training
- ✅ **Requirement 3**: LoRA Configuration and LLM Fine-tuning
- ✅ **Requirement 4**: Local Training Validation
- ✅ **Requirement 5**: Data Pipeline Verification
- ✅ **Requirement 7**: MLflow Logging and Checkpoint Management
- ✅ **Requirement 8**: Configuration Validation
- ✅ **Requirement 9**: Forward Pass Verification
- ✅ **Requirement 10**: Error Handling and Recovery

This ensures that all critical components of the Phase 2 training pipeline are working correctly before committing to full-scale training.