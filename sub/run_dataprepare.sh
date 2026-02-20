#!/bin/bash
# Purr-Sight Data Preparation Script
# Usage: ./sub/run_dataprepare.sh [1|2]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

usage() {
    echo "=========================================="
    echo "   Purr-Sight Data Preparation"
    echo "=========================================="
    echo ""
    echo "Usage: $0 [phase]"
    echo ""
    echo "Arguments:"
    echo "  phase: 1 or 2 (Required)"
    echo "    1 - Prepare Phase 1 data (alignment/contrastive learning)"
    echo "    2 - Prepare Phase 2 data (instruction tuning)"
    echo ""
    echo "Examples:"
    echo "  $0 1    # Prepare Phase 1 data"
    echo "  $0 2    # Prepare Phase 2 data"
    echo ""
    exit 1
}

if [ -z "$1" ]; then
    usage
fi

PHASE=$1

if [[ "$PHASE" != "1" && "$PHASE" != "2" ]]; then
    echo "❌ Error: Phase must be 1 or 2."
    usage
fi

echo "=========================================="
echo "   Data Preparation - Phase $PHASE"
echo "=========================================="
echo "Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
echo ""

if [ "$PHASE" == "1" ]; then
    echo "🔧 Preparing Phase 1 Data (Contrastive Learning)"
    echo ""
    
    # Create directories
    mkdir -p data/phase1/online
    mkdir -p data/phase1/offline
    
    echo "📝 Creating Phase 1 online data (simple modality pairs)..."
    
    # Create simple alignment data using existing media files
    cat > data/phase1/online/train.jsonl << 'EOF'
{"text": "a cat in a photo", "image": "data/cat.png"}
{"text": "a human speaking", "audio": "data/audio.m4a"}
{"text": "a video of movement", "image": "data/cat.png", "audio": "data/audio.m4a"}
{"text": "animal behavior observation", "image": "data/cat.png"}
{"text": "audio recording of sounds", "audio": "data/audio.m4a"}
{"text": "cat sitting calmly", "image": "data/cat.png"}
{"text": "voice and speech", "audio": "data/audio.m4a"}
{"text": "feline in frame", "image": "data/cat.png"}
EOF
    
    echo "✅ Created: data/phase1/online/train.jsonl (8 samples)"
    
    echo ""
    echo "📝 Setting up Phase 1 offline data..."
    
    # Check if external preprocessed data exists
    EXTERNAL_DATA="/Users/physicsboy/Desktop/data_4_purr/data_formal_alin"
    if [ -d "$EXTERNAL_DATA/preprocessed" ] && [ -f "$EXTERNAL_DATA/preprocessed/index.jsonl" ]; then
        echo "✅ Found external preprocessed data"
        echo "   Linking to local offline directory..."
        
        # Create symbolic link to external preprocessed data
        ln -sf "$EXTERNAL_DATA/preprocessed" data/phase1/offline/preprocessed
        
        echo "✅ Linked: data/phase1/offline/preprocessed -> $EXTERNAL_DATA/preprocessed"
        echo "   (Using large-scale preprocessed data without copying)"
    else
        echo "⚠️  External preprocessed data not found at: $EXTERNAL_DATA/preprocessed"
        echo "   Will use online data only"
    fi
    
    echo ""
    echo "=========================================="
    echo "✅ Phase 1 Data Preparation Complete"
    echo "=========================================="
    echo ""
    echo "Available datasets:"
    echo "  Online mode:  data/phase1/online/train.jsonl (8 samples)"
    if [ -L "data/phase1/offline/preprocessed" ]; then
        echo "  Offline mode: data/phase1/offline/preprocessed/ (linked to external data)"
    fi
    echo ""
    echo "Next steps:"
    echo "  # Train with online data"
    echo "  ./sub/run_train.sh 1 --config config/phase1_online.yaml"
    echo ""
    if [ -L "data/phase1/offline/preprocessed" ]; then
        echo "  # Train with offline data (faster, large-scale)"
        echo "  ./sub/run_train.sh 1 --config config/phase1_offline.yaml"
        echo ""
    fi
    
elif [ "$PHASE" == "2" ]; then
    echo "🔧 Preparing Phase 2 Data (Instruction Tuning)"
    echo ""
    
    # Create directory
    mkdir -p data/phase2
    
    echo "📝 Creating Phase 2 instruction data..."
    
    # Create instruction tuning data
    cat > data/phase2/train.jsonl << 'EOF'
{"instruction": "Analyze the animal in this image. Describe its posture and state.", "response": "{\"animal\": \"cat\", \"posture\": \"sitting\", \"state\": \"calm\", \"confidence\": 0.9}", "image": "data/cat.png"}
{"instruction": "What sounds can you hear in this audio? Provide analysis.", "response": "{\"sound_type\": \"human_voice\", \"activity\": \"speaking\", \"environment\": \"indoor\", \"confidence\": 0.85}", "audio": "data/audio.m4a"}
{"instruction": "Analyze this multimodal input. Image shows an animal, audio contains sounds. Provide comprehensive analysis in JSON format.", "response": "{\"visual\": {\"animal\": \"cat\", \"posture\": \"sitting\"}, \"audio\": {\"sound\": \"ambient\"}, \"overall_assessment\": \"calm_environment\", \"confidence\": 0.88}", "image": "data/cat.png", "audio": "data/audio.m4a"}
EOF
    
    echo "✅ Created: data/phase2/train.jsonl (3 samples)"
    
    echo ""
    echo "=========================================="
    echo "✅ Phase 2 Data Preparation Complete"
    echo "=========================================="
    echo ""
    echo "Available dataset:"
    echo "  data/phase2/train.jsonl (3 samples)"
    echo ""
    echo "Next steps:"
    echo "  # First, train Phase 1 to get aligner checkpoint"
    echo "  ./sub/run_train.sh 1 --config config/phase1_online.yaml"
    echo ""
    echo "  # Then train Phase 2"
    echo "  ./sub/run_train.sh 2 --config config/phase2_example.yaml"
    echo ""
fi
