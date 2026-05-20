#!/bin/bash

###############################################################################
# LSTM-AE Training Script - For GCP Temporal Defense
###############################################################################

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Activate conda environment (using the original guest environment)
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate guest 2>/dev/null || echo "Warning: Could not activate conda environment 'guest'"

# Set Python path
export PYTHONPATH="${PROJ_ROOT}/coperception:$PYTHONPATH"
export PYTHONPATH="${PROJ_ROOT}:$PYTHONPATH"

# Change to working directory
cd "${PROJ_ROOT}/coperception/tools/det/"

# Color output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}   LSTM-AE Training for GCP${NC}"
echo -e "${GREEN}================================================${NC}"

# Default parameters
DATA_PATH="${PROJ_ROOT}/coperception/logs/scene_0_ego_1.npz"
SAVE_PATH="${PROJ_ROOT}/coperception/logs/model/"
EPOCHS=100
BATCH_SIZE=32
SEQ_LENGTH=5
LEARNING_RATE=0.001

# Show usage
function show_usage() {
    echo ""
    echo -e "${YELLOW}Usage:${NC}"
    echo "  $0 [options]"
    echo ""
    echo -e "${YELLOW}Options:${NC}"
    echo "  --data_path PATH      BEV flow data path (default: $DATA_PATH)"
    echo "  --save_path PATH      Model save path (default: $SAVE_PATH)"
    echo "  --epochs N            Number of epochs (default: $EPOCHS)"
    echo "  --batch_size N        Batch size (default: $BATCH_SIZE)"
    echo "  --seq_length N        Sequence length (default: $SEQ_LENGTH)"
    echo "  --lr FLOAT            Learning rate (default: $LEARNING_RATE)"
    echo ""
    echo -e "${YELLOW}Example:${NC}"
    echo "  $0 --epochs 150 --batch_size 64"
    echo ""
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data_path)
            DATA_PATH="$2"
            shift 2
            ;;
        --save_path)
            SAVE_PATH="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --seq_length)
            SEQ_LENGTH="$2"
            shift 2
            ;;
        --lr)
            LEARNING_RATE="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

echo -e "${GREEN}[INFO] Training Configuration:${NC}"
echo "  Data Path: $DATA_PATH"
echo "  Save Path: $SAVE_PATH"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Sequence Length: $SEQ_LENGTH"
echo "  Learning Rate: $LEARNING_RATE"
echo ""

# Create save directory
mkdir -p "$SAVE_PATH"

# Run training
echo -e "${GREEN}[INFO] Starting LSTM-AE training...${NC}"

python train_lstm_ae.py \
    --data_path "$DATA_PATH" \
    --save_path "$SAVE_PATH" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --seq_length $SEQ_LENGTH \
    --lr $LEARNING_RATE

echo -e "${GREEN}[INFO] Training completed!${NC}"
echo -e "${GREEN}[INFO] Model saved to: $SAVE_PATH${NC}"
