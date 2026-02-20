#!/bin/bash
# Purr-Sight Inference Launcher
# Usage: ./sub/run_pred.sh --checkpoint <path> [--input <path>] [--type video|image|text]

set -e  # Exit on error

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root
cd "$PROJECT_ROOT"

# Function to show usage
usage() {
    echo "=========================================="
    echo "   Purr-Sight Inference Launcher"
    echo "=========================================="
    echo ""
    echo "Usage: $0 --checkpoint PATH [options]"
    echo ""
    echo "Required:"
    echo "  --checkpoint PATH   Path to trained model checkpoint"
    echo ""
    echo "Input (choose one):"
    echo "  --video PATH        Path to video file"
    echo "  --image PATH        Path to image file"
    echo "  --text TEXT         Text description"
    echo ""
    echo "Options:"
    echo "  --output PATH       Output JSON file path (default: results/inference_<timestamp>.json)"
    echo "  --device DEVICE     Device to use: cpu, cuda, mps (default: auto)"
    echo ""
    echo "Examples:"
    echo "  # Image inference"
    echo "  $0 --checkpoint checkpoints/phase2/best.pt --image data/cat.png"
    echo ""
    echo "  # Video inference"
    echo "  $0 --checkpoint checkpoints/phase2/best.pt --video data/test1.mov"
    echo ""
    echo "  # Text inference"
    echo "  $0 --checkpoint checkpoints/phase2/best.pt --text \"A cat is sitting calmly\""
    echo ""
    echo "  # With custom output"
    echo "  $0 --checkpoint checkpoints/phase2/best.pt --image data/cat.png --output my_result.json"
    echo ""
    exit 1
}

# Parse arguments
CHECKPOINT=""
INPUT_TYPE=""
INPUT_PATH=""
INPUT_TEXT=""
OUTPUT=""
DEVICE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --video)
            INPUT_TYPE="video"
            INPUT_PATH="$2"
            shift 2
            ;;
        --image)
            INPUT_TYPE="image"
            INPUT_PATH="$2"
            shift 2
            ;;
        --text)
            INPUT_TYPE="text"
            INPUT_TEXT="$2"
            shift 2
            ;;
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "❌ Unknown option: $1"
            usage
            ;;
    esac
done

# Validate checkpoint
if [ -z "$CHECKPOINT" ]; then
    echo "❌ Error: --checkpoint is required"
    usage
fi

if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ Error: Checkpoint file not found: $CHECKPOINT"
    exit 1
fi

# Validate input
if [ -z "$INPUT_TYPE" ]; then
    echo "❌ Error: Must specify one of --video, --image, or --text"
    usage
fi

# Validate input file exists (for video/image)
if [[ "$INPUT_TYPE" == "video" || "$INPUT_TYPE" == "image" ]]; then
    if [ ! -f "$INPUT_PATH" ]; then
        echo "❌ Error: Input file not found: $INPUT_PATH"
        exit 1
    fi
fi

# Set default output path
if [ -z "$OUTPUT" ]; then
    TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
    OUTPUT="results/inference_${INPUT_TYPE}_${TIMESTAMP}.json"
fi

# Create results directory
mkdir -p "$(dirname "$OUTPUT")"

echo "=========================================="
echo "   Purr-Sight Inference"
echo "=========================================="
echo "Checkpoint:   $CHECKPOINT"
echo "Input Type:   $INPUT_TYPE"
if [ "$INPUT_TYPE" == "text" ]; then
    echo "Input Text:   $INPUT_TEXT"
else
    echo "Input File:   $INPUT_PATH"
fi
echo "Output:       $OUTPUT"
if [ -n "$DEVICE" ]; then
    echo "Device:       $DEVICE"
fi
echo "Date:         $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
echo ""

# Build command
CMD="python train/inference_module.py --checkpoint \"$CHECKPOINT\" --output \"$OUTPUT\""

if [ -n "$DEVICE" ]; then
    CMD="$CMD --device \"$DEVICE\""
fi

case $INPUT_TYPE in
    video)
        CMD="$CMD --video \"$INPUT_PATH\""
        ;;
    image)
        CMD="$CMD --image \"$INPUT_PATH\""
        ;;
    text)
        CMD="$CMD --text \"$INPUT_TEXT\""
        ;;
esac

# Run inference
echo "🚀 Running inference..."
echo ""

eval $CMD

# Check if inference succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ Inference Completed Successfully"
    echo "=========================================="
    echo ""
    echo "Result saved to: $OUTPUT"
    echo ""
    echo "View result:"
    echo "  cat $OUTPUT | python -m json.tool"
    echo ""
else
    echo ""
    echo "❌ Inference Failed"
    exit 1
fi
