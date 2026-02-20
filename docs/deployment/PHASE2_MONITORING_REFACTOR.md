# Phase 2 Monitoring Refactoring

## Overview

Phase 2 training monitoring has been refactored to match Phase 1's clean and organized structure, using only MLflow for logging with clear naming conventions.

## Changes Made

### 1. Checkpoint Organization (Phase 1 Style)

**Before:**
```
checkpoints/phase2/
├── last.ckpt
├── phase2-epoch=00-train_loss=2.9476.ckpt
├── phase2-epoch=01-train_loss=3.0279.ckpt
└── phase2-epoch=02-train_loss=2.9635.ckpt
```

**After:**
```
checkpoints/phase2/
└── ac7c134d5a5a44c4ac38fbc83f650af0_20260201_041227/  # Unique run ID + timestamp
    ├── lightning_checkpoints/
    │   ├── best-epoch=00-train_loss=2.9476.ckpt  # Only best checkpoint
    │   └── last.ckpt                              # Only last checkpoint
    ├── model.pt                                   # Deployment model (saved to MLflow)
    └── config.json                                # Training config (saved to MLflow)
```

### 2. MLflow Integration

**Improvements:**
- ✅ Unique run ID with timestamp format: `{uuid}_{YYYYMMDD_HHMMSS}`
- ✅ Clear run naming: `phase2_{run_id}`
- ✅ Automatic artifact saving after training
- ✅ Training curve visualization saved to MLflow
- ✅ Configuration saved as JSON artifact

**MLflow Structure:**
```
mlruns/
└── {experiment_id}/
    └── {run_id}/
        ├── artifacts/
        │   ├── model/
        │   │   └── model.pt          # Deployment model
        │   ├── config/
        │   │   └── config.json       # Training configuration
        │   └── plots/
        │       └── training_curve.png # Training visualization
        ├── metrics/
        │   ├── train_loss
        │   ├── train_loss_epoch
        │   ├── learning_rate
        │   └── epoch
        ├── params/
        │   └── {all config parameters}
        └── meta.yaml
```

### 3. Checkpoint Saving Strategy

**Before:**
- Saved top 3 checkpoints
- Saved all epoch checkpoints
- Cluttered checkpoint directory

**After:**
- Save only best checkpoint (top_k=1)
- Save only last checkpoint
- Clean, organized structure with run ID

### 4. Code Changes

#### `train/train_llm/train_phase2.py`

**Modified `setup_trainer()` method:**
```python
# Create unique run ID with timestamp (Phase 1 style)
import uuid
from datetime import datetime
run_id = f"{uuid.uuid4().hex}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# Create checkpoint directory with run ID
checkpoint_dir = Path(self.config.checkpoint_dir) / run_id
checkpoint_dir.mkdir(parents=True, exist_ok=True)

# Setup checkpoint callback (only last and best)
checkpoint_callback = ModelCheckpoint(
    dirpath=str(checkpoint_dir / "lightning_checkpoints"),
    filename="best-{epoch:02d}-{train_loss:.4f}",
    monitor=CHECKPOINT_MONITOR,
    mode=CHECKPOINT_MONITOR_MODE,
    save_top_k=1,  # Only save best
    save_last=True,  # Also save last
    every_n_epochs=self.config.save_every_n_epochs,
    verbose=True,
)
```

**Added `_save_artifacts_to_mlflow()` method:**
- Saves model weights (model.pt) for deployment
- Saves configuration (config.json)
- Creates and saves training curve visualization
- Follows Phase 1 artifact structure

## Benefits

### 1. Consistency
- Phase 1 and Phase 2 now use the same monitoring structure
- Easy to understand and navigate
- Predictable file locations

### 2. Disk Space Efficiency
- Only 2 checkpoints saved instead of 3+
- No redundant checkpoint files in root directory
- Organized by run ID for easy cleanup

### 3. Traceability
- Unique run ID links checkpoints to MLflow runs
- Timestamp in run ID for easy identification
- All artifacts grouped by run

### 4. MLflow Integration
- Complete training history in MLflow
- Visualizations automatically saved
- Easy comparison between runs
- Deployment-ready model artifacts

## Usage

### Starting Training

```bash
# Start Phase 2 training
bash sub/run_train.sh 2

# Training will create:
# - Checkpoint directory: checkpoints/phase2/{run_id}/
# - MLflow run: mlruns/{experiment_id}/{run_id}/
```

### Finding Checkpoints

```bash
# List all Phase 2 runs
ls -lt checkpoints/phase2/

# Find latest run
ls -lt checkpoints/phase2/ | head -n 2

# Access best checkpoint
checkpoints/phase2/{run_id}/lightning_checkpoints/best-*.ckpt

# Access last checkpoint
checkpoints/phase2/{run_id}/lightning_checkpoints/last.ckpt
```

### Viewing MLflow Logs

```bash
# Start MLflow UI
mlflow ui

# Open browser to http://localhost:5000
# Navigate to "phase2_training_with_pretrained_aligner" experiment
# View runs, metrics, and artifacts
```

## Migration Guide

### For Existing Checkpoints

Old checkpoints in `checkpoints/phase2/*.ckpt` can be safely deleted after verifying new training works:

```bash
# Remove old checkpoint files (after verification)
rm -f checkpoints/phase2/last.ckpt
rm -f checkpoints/phase2/phase2-epoch*.ckpt
```

### For Scripts Using Checkpoints

Update scripts to use the new structure:

**Before:**
```python
checkpoint_path = "checkpoints/phase2/last.ckpt"
```

**After:**
```python
# Find latest run
import os
from pathlib import Path

phase2_dir = Path("checkpoints/phase2")
runs = sorted([d for d in phase2_dir.iterdir() if d.is_dir()], 
              key=lambda x: x.stat().st_mtime, reverse=True)
latest_run = runs[0]
checkpoint_path = latest_run / "lightning_checkpoints" / "last.ckpt"
```

## Comparison with Phase 1

| Feature | Phase 1 | Phase 2 (After Refactor) |
|---------|---------|--------------------------|
| Run ID Format | `{uuid}_{timestamp}` | `{uuid}_{timestamp}` ✅ |
| Checkpoint Dir | `checkpoints/alignment/{run_id}/` | `checkpoints/phase2/{run_id}/` ✅ |
| Checkpoints Saved | best + last | best + last ✅ |
| MLflow Logging | Yes | Yes ✅ |
| Artifact Saving | Yes (aligner.pt, config.json, plots) | Yes (model.pt, config.json, plots) ✅ |
| Training Curves | Yes | Yes ✅ |

## Future Improvements

1. **Automatic Cleanup**: Add script to clean up old runs (keep last N)
2. **Checkpoint Comparison**: Tool to compare metrics across runs
3. **Model Registry**: Register best models to MLflow Model Registry
4. **Distributed Training**: Ensure structure works with multi-GPU training

## References

- Phase 1 Implementation: `train/train_alignment/train.py`
- Phase 2 Implementation: `train/train_llm/train_phase2.py`
- MLflow Documentation: https://mlflow.org/docs/latest/index.html
- PyTorch Lightning Checkpointing: https://lightning.ai/docs/pytorch/stable/common/checkpointing.html
