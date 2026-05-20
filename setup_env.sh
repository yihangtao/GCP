#!/bin/bash

###############################################################################
# GCP Environment Setup Script
# Run this script before first use to install dependencies
###############################################################################

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}   GCP Environment Setup${NC}"
echo -e "${GREEN}================================================${NC}"

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo -e "${RED}[ERROR] Conda is not installed!${NC}"
    echo "Please install Anaconda or Miniconda first."
    exit 1
fi

# Enter coperception directory
cd /data2/user2/yihang/GCP/coperception

# Check if environment.yml exists
if [ ! -f "environment.yml" ]; then
    echo -e "${RED}[ERROR] environment.yml not found!${NC}"
    exit 1
fi

# Create or update conda environment
echo -e "${GREEN}[INFO] Setting up conda environment 'guest'...${NC}"
if conda env list | grep -q "^guest "; then
    echo -e "${YELLOW}[INFO] Environment 'guest' already exists. Updating...${NC}"
    conda env update -f environment.yml -n guest
else
    echo -e "${GREEN}[INFO] Creating new environment 'guest'...${NC}"
    conda env create -f environment.yml
fi

# Activate environment
echo -e "${GREEN}[INFO] Activating environment...${NC}"
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh
conda activate guest

# Install coperception package
echo -e "${GREEN}[INFO] Installing coperception package...${NC}"
pip install -e .

# Install additional dependencies
echo -e "${GREEN}[INFO] Installing additional dependencies...${NC}"
pip install shapely filterpy

# Verify installation
echo ""
echo -e "${GREEN}[INFO] Verifying installation...${NC}"
python -c "from coperception.datasets import V2XSimDet; print('✓ coperception.datasets imported successfully')" || echo -e "${RED}✗ Failed to import coperception.datasets${NC}"
python -c "from coperception.models.det.LSTMAutoencoder import LSTMAE; print('✓ LSTM-AE imported successfully')" || echo -e "${RED}✗ Failed to import LSTM-AE${NC}"
python -c "import torch; print(f'✓ PyTorch {torch.__version__} with CUDA {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')" || echo -e "${RED}✗ PyTorch not available${NC}"

echo ""
echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}   Setup Complete!${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "  1. Make sure you have the V2X-Sim dataset"
echo "  2. Train LSTM-AE: ./scripts/train_lstm_ae.sh"
echo "  3. Run GCP: ./scripts/run_gcp.sh"
echo ""
echo -e "${YELLOW}To activate the environment manually:${NC}"
echo "  conda activate guest"
echo ""
