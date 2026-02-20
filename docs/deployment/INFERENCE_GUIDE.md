# Phase 2 Inference Guide

## Overview

This guide explains how to run end-to-end inference with the trained Phase 2 model for cat behavior analysis.

## Quick Start

### Using the Inference Script

```bash
# Image inference
./sub/run_pred.sh --checkpoint checkpoints/phase2/94525c6650a3407985928d7c2f83f9eb_20260201_044652/model.pt \
                  --image data/cat.png

# Text inference
./sub/run_pred.sh --checkpoint checkpoints/phase2/94525c6650a3407985928d7c2f83f9eb_20260201_044652/model.pt \
                  --text "A cat is sitting calmly on a windowsill"

# Video inference (requires ffmpeg and opencv-python)
./sub/run_pred.sh --checkpoint checkpoints/phase2/94525c6650a3407985928d7c2f83f9eb_20260201_044652/model.pt \
                  --video data/test1.mov
```

### Using Python Directly

```python
from train.inference_module import PurrSightInference

# Initialize inference pipeline
inference = PurrSightInference(
    checkpoint_path="checkpoints/phase2/94525c6650a3407985928d7c2f83f9eb_20260201_044652/model.pt",
    device=None  # Auto-detect (cuda/mps/cpu)
)

# Run inference on image
result = inference.infer_from_image("data/cat.png")

# Run inference on text
result = inference.infer_from_text("A cat is sitting calmly")

# Save result
inference.save_result(result, "results/output.json")
```

## Inference Module Features

### Supported Input Types

1. **Image Input** (`.jpg`, `.png`, etc.)
   - Single image analysis
   - Extracts visual features (ears, tail, posture)
   - Provides behavioral classification

2. **Text Input** (text description)
   - Scene description analysis
   - Behavioral inference from text
   - Useful for testing or when images unavailable

3. **Video Input** (`.mp4`, `.mov`, etc.) - Requires additional dependencies
   - Extracts multiple frames
   - Optional audio extraction
   - Temporal behavior analysis

### Output Format

The inference module returns a structured JSON with:

```json
{
  "timestamp": "2026-02-01T05:01:35.500094",
  "input_type": "image",
  "model_checkpoint": "checkpoints/phase2/.../model.pt",
  "generated_text": "Model's generated response...",
  "metadata": {
    "model_version": "2.0",
    "phase": "phase2"
  },
  "input_file": "data/cat.png",
  "analysis": {
    "raw_response": "...",
    "note": "Parsing status"
  }
}
```

## Model Architecture

The inference pipeline uses:

1. **Image Encoder**: MobileNetV4-small (frozen)
2. **Audio Encoder**: PANNs CNN14 (frozen)
3. **Aligner**: Contrastive aligner from Phase 1 (frozen)
4. **Projector**: Multimodal projector (trained)
5. **LLM**: Qwen2.5-0.5B with LoRA (trained)

## Preprocessing

### Image Preprocessing
- Resize to 256x256
- Center crop to 224x224
- Normalize with ImageNet stats
- Output: (3, 224, 224) tensor

### Audio Preprocessing
- Resample to 16kHz
- Convert to mel spectrogram (64 mel bins)
- Trim silence
- Pad/crop to 256 frames
- Output: (64, 256) tensor

### Text Preprocessing
- Tokenize with LLM tokenizer
- Pad to max_length
- Output: (seq_len,) token IDs

## Testing

Run the test script to verify inference works:

```bash
python test_inference.py
```

This will test:
- Image inference with `data/cat.png`
- Text inference with sample description
- Save results to `results/test_inference_result.json`

## Performance Notes

### Device Support
- **CUDA**: Best performance for GPU inference
- **MPS**: Good performance on Apple Silicon Macs
- **CPU**: Slower but works everywhere

### Inference Speed (approximate)
- Image inference: ~2-5 seconds on CPU, <1 second on GPU
- Text inference: ~1-3 seconds on CPU, <0.5 seconds on GPU
- Video inference: Depends on video length and frame sampling

### Memory Requirements
- Minimum: 4GB RAM
- Recommended: 8GB+ RAM
- GPU: 4GB+ VRAM for faster inference

## Troubleshooting

### Common Issues

1. **ImportError: cannot import name 'ImagePreprocessor'**
   - Fixed: Use `_ImageProcessor`, `_AudioProcessor`, `_TextProcessor`

2. **ValueError: 不支持的图像类型**
   - Fixed: Pass PIL Image object, not string path

3. **ValueError: 所有模态都缺失**
   - Fixed: Provide dummy tensors for missing modalities

4. **Model not generating structured JSON**
   - Expected: Model needs more training data and epochs
   - Current output is free-form text
   - Future: Fine-tune with more structured examples

### Video Inference Requirements

For video inference, install additional dependencies:

```bash
# Install OpenCV for frame extraction
pip install opencv-python

# Install ffmpeg for audio extraction (macOS)
brew install ffmpeg

# Or on Linux
sudo apt-get install ffmpeg
```

## Next Steps

1. **Improve Training Data**
   - Add more Phase 2 training samples
   - Include diverse cat behaviors
   - Ensure JSON format in responses

2. **Fine-tune Generation**
   - Adjust temperature and top_p
   - Add system prompts for structured output
   - Use constrained decoding for JSON

3. **Optimize Performance**
   - Quantize model for faster inference
   - Batch multiple inputs
   - Cache model weights

4. **Add Features**
   - Real-time video streaming
   - Batch processing
   - API server deployment

## Related Documentation

- [Phase 2 Training Guide](PHASE2_DEPLOYMENT_GUIDE.md)
- [Phase 2 Cluster Training](PHASE2_CLUSTER_TRAINING.md)
- [Phase 2 Monitoring](PHASE2_MONITORING_REFACTOR.md)
- [Checkpoint-MLflow Mapping](PHASE2_CHECKPOINT_MLFLOW_MAPPING.md)
