# Purr-Sight Training and Inference Testing Guide

This guide provides step-by-step instructions for testing the complete Purr-Sight pipeline, including Phase 1 training (online and offline modes), Phase 2 training, and inference.

## Prerequisites

1. **Environment Setup**
   ```bash
   conda activate purrsight
   ```

2. **Data Preparation**
   - **Phase 1 对比学习数据** (简单的模态对应):
     - 在线模式: `data/test_alignment/train.jsonl` (测试数据)
     - 离线模式: `/Users/physicsboy/Desktop/data_4_purr/data_formal_alin/preprocessed/` (正式数据)
     - 格式: `{"text": "...", "audio": "...", "image": "..."}`
   - **Phase 2 指令微调数据** (结构化指令-响应):
     - `data/instruction/train.jsonl`
     - 格式: `{"instruction": "...", "response": "...", "image": "...", "audio": "..."}`
   - Test media files: `data/cat.png`, `data/test1.mov`, `data/audio.m4a`

**重要**: Phase 1 和 Phase 2 使用不同格式的数据！详见 `docs/guides/DATA_FORMAT_GUIDE.md`

3. **Model Files**
   - LLM model: `models/Qwen2.5-0.5B-Instruct/`
   - Encoder models: `models/mobilenetv4/`, `models/panns/`, `models/mini-lm-l6-h384-uncased/`

## Testing Workflow

### Step 1: Phase 1 Training - Online Mode

Test Phase 1 training with online data loading (real-time preprocessing).

```bash
# Run Phase 1 training with online data loading
./sub/run_train.sh 1 --config config/phase1_online.yaml

# Expected output:
# - Training runs for 3 epochs
# - Checkpoints saved to: checkpoints/alignment/<run_id>_<timestamp>/
# - MLflow logs in: mlruns/
```

**What to verify:**
- ✅ Training completes without errors
- ✅ Checkpoint files created:
  - `checkpoints/alignment/<run_id>_<timestamp>/aligner.pt`
  - `checkpoints/alignment/<run_id>_<timestamp>/model.ckpt`
  - `checkpoints/alignment/<run_id>_<timestamp>/config.json`
- ✅ MLflow experiment created with metrics logged
- ✅ Training loss decreases over epochs

**Expected checkpoint location:**
```
checkpoints/alignment/
└── <run_id>_<timestamp>/
    ├── aligner.pt              # Aligner weights (for Phase 2)
    ├── model.ckpt              # Full Lightning checkpoint
    ├── config.json             # Training configuration
    └── lightning_checkpoints/  # Lightning auto-checkpoints
        ├── epoch=XX-step=YY.ckpt
        └── last.ckpt
```

### Step 2: Phase 1 Training - Offline Mode

Test Phase 1 training with offline data loading (preprocessed data).

```bash
# Run Phase 1 training with offline data loading
./sub/run_train.sh 1 --config config/phase1_offline.yaml

# Expected output:
# - Training runs for 3 epochs (faster than online mode)
# - Checkpoints saved to: checkpoints/alignment/<run_id>_<timestamp>/
# - MLflow logs in: mlruns/
```

**What to verify:**
- ✅ Training completes without errors
- ✅ Training is faster than online mode (uses preprocessed data)
- ✅ Checkpoint files created (same structure as online mode)
- ✅ MLflow experiment created with different experiment name
- ✅ Training loss decreases over epochs

**Note:** The experiment name is different (`alignment_phase1_offline` vs `alignment_phase1_online`) to distinguish between the two modes in MLflow.

### Step 3: Verify Phase 1 Checkpoints

Check that Phase 1 checkpoints are valid and contain the expected components.

```bash
# List all Phase 1 checkpoints
ls -lh checkpoints/alignment/

# Check the latest checkpoint
LATEST_PHASE1=$(find checkpoints/alignment -name "aligner.pt" -type f | sort -r | head -1)
echo "Latest Phase 1 checkpoint: $LATEST_PHASE1"

# Verify checkpoint size (should be ~6-7 MB)
du -h "$LATEST_PHASE1"
```

**What to verify:**
- ✅ `aligner.pt` file exists and is ~6-7 MB
- ✅ `config.json` contains training configuration
- ✅ Checkpoint directory has unique timestamp

### Step 4: Phase 2 Training

Test Phase 2 training using the Phase 1 checkpoint.

```bash
# Option 1: Automatic checkpoint detection
./sub/run_train.sh 2 --config config/phase2_example.yaml

# Option 2: Specify Phase 1 checkpoint explicitly
PHASE1_CKPT=$(find checkpoints/alignment -name "aligner.pt" -type f | sort -r | head -1)
./sub/run_train.sh 2 --config config/phase2_example.yaml --checkpoint "$PHASE1_CKPT"

# Expected output:
# - Phase 1 checkpoint loaded successfully
# - LoRA applied to LLM
# - Training runs for configured epochs
# - Checkpoints saved to: checkpoints/phase2/
# - MLflow logs in: mlruns/
```

**What to verify:**
- ✅ Phase 1 checkpoint loaded successfully
- ✅ LoRA parameters applied to LLM
- ✅ Training completes without errors
- ✅ Checkpoint files created:
  - `checkpoints/phase2/phase2-epoch=XX-train_loss=Y.YYYY.ckpt`
  - `checkpoints/phase2/last.ckpt`
- ✅ MLflow experiment created with metrics logged
- ✅ Training loss decreases over epochs

**Expected checkpoint location:**
```
checkpoints/phase2/
├── phase2-epoch=00-train_loss=X.XXXX.ckpt
├── phase2-epoch=01-train_loss=X.XXXX.ckpt
├── phase2-epoch=02-train_loss=X.XXXX.ckpt
└── last.ckpt
```

### Step 5: Inference Testing

Test the inference pipeline with different input modalities.

#### 5.1 Image Inference

```bash
# Run inference on image
./sub/run_pred.sh \
  --checkpoint checkpoints/phase2/last.ckpt \
  --image data/cat.png \
  --output results/inference_image.json

# View result
cat results/inference_image.json | python -m json.tool
```

**Expected output:**
```json
{
  "timestamp": "2026-02-01T...",
  "input_type": "image",
  "input_file": "data/cat.png",
  "model_checkpoint": "checkpoints/phase2/last.ckpt",
  "analysis": {
    "behavior": "sitting",
    "posture": "relaxed",
    "activity_level": "low",
    "emotional_state": "calm",
    "confidence": 0.85,
    "spatial_features": {
      "location": "indoor",
      "objects_detected": ["cat", "furniture", "window"],
      "scene_context": "home environment"
    }
  },
  "metadata": {
    "model_version": "1.0",
    "processing_time_ms": 150
  }
}
```

#### 5.2 Video Inference

```bash
# Run inference on video
./sub/run_pred.sh \
  --checkpoint checkpoints/phase2/last.ckpt \
  --video data/test1.mov \
  --output results/inference_video.json

# View result
cat results/inference_video.json | python -m json.tool
```

**Expected output:**
```json
{
  "timestamp": "2026-02-01T...",
  "input_type": "video",
  "input_file": "data/test1.mov",
  "model_checkpoint": "checkpoints/phase2/last.ckpt",
  "analysis": {
    "behavior": "sitting",
    "posture": "relaxed",
    "activity_level": "low",
    "emotional_state": "calm",
    "confidence": 0.85,
    "temporal_features": {
      "movement_detected": true,
      "duration_seconds": 5.0,
      "key_moments": [
        {"time": 1.2, "event": "cat sits down"},
        {"time": 3.5, "event": "cat looks around"}
      ]
    }
  },
  "metadata": {
    "model_version": "1.0",
    "processing_time_ms": 150
  }
}
```

#### 5.3 Text Inference

```bash
# Run inference on text description
./sub/run_pred.sh \
  --checkpoint checkpoints/phase2/last.ckpt \
  --text "A cat is sitting calmly on a windowsill" \
  --output results/inference_text.json

# View result
cat results/inference_text.json | python -m json.tool
```

**Expected output:**
```json
{
  "timestamp": "2026-02-01T...",
  "input_type": "text",
  "input_text": "A cat is sitting calmly on a windowsill",
  "model_checkpoint": "checkpoints/phase2/last.ckpt",
  "analysis": {
    "behavior": "sitting",
    "posture": "relaxed",
    "activity_level": "low",
    "emotional_state": "calm",
    "confidence": 0.85,
    "interpretation": {
      "scene_understanding": "indoor domestic setting",
      "inferred_behavior": "resting",
      "context_notes": "Based on text description"
    }
  },
  "metadata": {
    "model_version": "1.0",
    "processing_time_ms": 150
  }
}
```

### Step 6: Verify MLflow Logs

Check that all experiments are logged correctly in MLflow.

```bash
# View MLflow UI (optional)
mlflow ui --backend-store-uri mlruns/

# Or check MLflow logs directly
ls -lh mlruns/
```

**What to verify:**
- ✅ Phase 1 online experiment logged
- ✅ Phase 1 offline experiment logged
- ✅ Phase 2 experiment logged
- ✅ Metrics (train_loss, val_loss) logged for each epoch
- ✅ Hyperparameters logged
- ✅ Artifacts (checkpoints, plots) logged

## Troubleshooting

### Issue: Phase 1 training fails with "data file not found"

**Solution:**
- Check that `data/test_alignment/train.jsonl` exists for online mode
- Check that `data/instruction/preprocessed/index.jsonl` exists for offline mode
- Verify the `data_path` in the config file

### Issue: Phase 2 training fails with "Phase 1 checkpoint not found"

**Solution:**
- Run Phase 1 training first to generate the checkpoint
- Verify the checkpoint path in `config/phase2_example.yaml`
- Use `--checkpoint` flag to specify the checkpoint explicitly

### Issue: Inference fails with "checkpoint not found"

**Solution:**
- Run Phase 2 training first to generate the checkpoint
- Verify the checkpoint path exists
- Use the correct checkpoint file (`.ckpt` extension)

### Issue: MLflow logs not appearing

**Solution:**
- Check that `mlruns/` directory exists
- Verify MLflow tracking URI in config files
- Check file permissions on `mlruns/` directory

## Summary Checklist

After completing all tests, verify:

- [ ] Phase 1 online training completed successfully
- [ ] Phase 1 offline training completed successfully
- [ ] Phase 1 checkpoints created and valid
- [ ] Phase 2 training completed successfully
- [ ] Phase 2 checkpoints created and valid
- [ ] Image inference works correctly
- [ ] Video inference works correctly
- [ ] Text inference works correctly
- [ ] All MLflow experiments logged
- [ ] All results saved to JSON files

## Next Steps

Once all tests pass:

1. **Production Training**: Run full training with more epochs
   ```bash
   # Phase 1 with 20 epochs
   ./sub/run_train.sh 1 --config config/train_config.yaml --epochs 20
   
   # Phase 2 with 10 epochs
   ./sub/run_train.sh 2 --config config/phase2_example.yaml --epochs 10
   ```

2. **Model Evaluation**: Run comprehensive evaluation on test set
   ```bash
   python tests/acceptance_test_phase2.py
   ```

3. **Deployment**: Deploy the trained model for production use

## Configuration Files Reference

- `config/phase1_online.yaml` - Phase 1 online training (real-time preprocessing)
- `config/phase1_offline.yaml` - Phase 1 offline training (preprocessed data)
- `config/train_config.yaml` - Full Phase 1 configuration (production)
- `config/phase2_example.yaml` - Phase 2 training configuration
- `config/validation_config.yaml` - Validation configuration

## Scripts Reference

- `sub/run_train.sh` - Unified training script for Phase 1 and Phase 2
- `sub/run_pred.sh` - Inference script for video/image/text inputs
- `train/train_alignment/train.py` - Phase 1 training implementation
- `train/train_llm/train_phase2.py` - Phase 2 training implementation
- `train/inference_module.py` - Inference module implementation

## Contact

For issues or questions, refer to:
- Project documentation: `docs/`
- Acceptance test guide: `docs/acceptance/ACCEPTANCE_TEST_GUIDE.md`
- Quick start guide: `docs/acceptance/QUICK_START_ACCEPTANCE.md`
