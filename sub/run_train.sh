#!/bin/bash
# Purr-Sight Training Launcher
# Usage: ./sub/run_train.sh [1|2] [config_path]

set -e  # Exit on error

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root
cd "$PROJECT_ROOT"

# Function to show usage
usage() {
    echo "=========================================="
    echo "   Purr-Sight Training Launcher"
    echo "=========================================="
    echo ""
    echo "Usage: $0 [phase] [options]"
    echo ""
    echo "Arguments:"
    echo "  phase: 1 or 2 (Required)"
    echo "    1 - Phase 1: Alignment Training"
    echo "    2 - Phase 2: LLM Fine-tuning with LoRA"
    echo ""
    echo "Options:"
    echo "  --config PATH       Config file path"
    echo "  --epochs N          Number of epochs"
    echo "  --batch-size N      Batch size"
    echo "  --checkpoint PATH   Phase 1 checkpoint (Phase 2 only)"
    echo ""
    echo "Examples:"
    echo "  $0 1                                    # Phase 1 with default config"
    echo "  $0 1 --epochs 20                        # Phase 1 with 20 epochs"
    echo "  $0 2 --checkpoint checkpoints/phase1.pt # Phase 2 with specific checkpoint"
    echo ""
    exit 1
}

# Check arguments
if [ -z "$1" ]; then
    usage
fi

PHASE=$1
shift  # Remove phase from arguments

# Parse optional arguments
CONFIG=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        *)
            # Keep other arguments for passing to training script
            break
            ;;
    esac
done

# Validate Phase
if [[ "$PHASE" != "1" && "$PHASE" != "2" ]]; then
    echo "❌ Error: Phase must be 1 or 2."
    usage
fi

echo "=========================================="
echo "   Purr-Sight Training - Phase $PHASE"
echo "=========================================="
echo "Project Root: $PROJECT_ROOT"
echo "Date:         $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
echo ""

# Run Phase 1 or Phase 2
if [ "$PHASE" == "1" ]; then
    echo "🚀 Starting Phase 1: Alignment Training"
    echo ""
    
    # Default config for Phase 1
    if [ -z "$CONFIG" ]; then
        CONFIG="config/phase1_online.yaml"
    fi
    
    # Check if config exists
    if [ ! -f "$CONFIG" ]; then
        echo "❌ Error: Config file '$CONFIG' not found."
        exit 1
    fi
    
    echo "Config: $CONFIG"
    echo ""
    
    # Set PYTHONPATH to include project root
    export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
    
    # Run Phase 1 training
    python train/train_alignment/train.py --config "$CONFIG" "$@"
    
    # Check if training succeeded
    if [ $? -eq 0 ]; then
        echo ""
        echo "=========================================="
        echo "✅ Phase 1 Training Completed Successfully"
        echo "=========================================="
        echo ""
        echo "Checkpoints saved in: checkpoints/alignment/"
        echo "MLflow logs in: mlruns/"
        echo ""
        echo "Next step: Run Phase 2 training"
        echo "  ./sub/run_train.sh 2 --checkpoint <path_to_aligner.pt>"
        echo ""
    else
        echo ""
        echo "❌ Phase 1 Training Failed"
        exit 1
    fi
    
elif [ "$PHASE" == "2" ]; then
    echo "🚀 Starting Phase 2: LLM Fine-tuning"
    echo ""
    
    # Default config for Phase 2
    if [ -z "$CONFIG" ]; then
        CONFIG="config/phase2_example.yaml"
    fi
    
    # Check if config exists
    if [ ! -f "$CONFIG" ]; then
        echo "❌ Error: Config file '$CONFIG' not found."
        exit 1
    fi
    
    echo "Config: $CONFIG"
    echo ""
    
    # Set PYTHONPATH to include project root
    export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
    
    # Note: Phase 1 checkpoint is passed via config file (adapter_path)
    # Do NOT pass it as --checkpoint to the training script
    # Run Phase 2 training
    python -m train.train_llm.train_phase2 --config "$CONFIG" "$@"
    
    # Check if training succeeded
    if [ $? -eq 0 ]; then
        echo ""
        echo "=========================================="
        echo "✅ Phase 2 Training Completed Successfully"
        echo "=========================================="
        echo ""
        echo "Checkpoints saved in: checkpoints/phase2/"
        echo "MLflow logs in: mlruns/"
        echo ""
        echo "Next step: Run inference"
        echo "  ./sub/run_pred.sh --checkpoint <path_to_checkpoint>"
        echo ""
    else
        echo ""
        echo "❌ Phase 2 Training Failed"
        exit 1
    fi
fi
