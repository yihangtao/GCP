#!/bin/bash

###############################################################################
# GCP (Guarded Collaborative Perception) - Experiment Runner
# Spatial-Temporal Defense Framework for Robust Collaborative Perception
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
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}   GCP Defense System - Running Script${NC}"
echo -e "${GREEN}================================================${NC}"

# Default parameters
DATA_PATH="${PROJ_ROOT}/dataset/V2X-Sim/test"
MODEL_PATH="${PROJ_ROOT}/coperception/ckpt/det/meanfusion/epoch_49.pth"
LSTM_MODEL="${PROJ_ROOT}/coperception/logs/model/best_model.pth"
LOG_PATH="${PROJ_ROOT}/coperception/logs"

# Default experiment parameters
SCENE_ID="8"  # Scene ID, can be changed via --scene_id
EGO_AGENT=1
NUM_ATTACKERS=1
HISTORY_LENGTH=5
RECON_THRESHOLD=0.8
BOX_MATCH_THRESH=0.3
SAMPLE_START=""  # Start frame (inclusive), empty means from beginning
SAMPLE_END=""    # End frame (inclusive), empty means to end
SAMPLE_ID=""     # Legacy parameter for backward compatibility

# Gated Late Fusion parameters
GATED_FUSION_IOU=0.3  # IoU threshold for box matching
GATED_FUSION_VOTES=2  # Minimum votes required to accept a box

# Attack parameters
ADV_METHOD="pgd"
EPS=0.5
ADV_ITER=15
ATTACK_MODE="Random"
ATTACK_RATIO=0.6
BAC_EPS=0.3
SIMPLE_BAC_EPS=0.3

# Show usage
function show_usage() {
    echo ""
    echo -e "${YELLOW}Usage:${NC}"
    echo "  $0 [mode] [options]"
    echo "  $0 [options]               # default: mode=gcp, adv_method=pgd"
    echo ""
    echo -e "${YELLOW}Modes:${NC}"
    echo "  upperbound       - Clean collaboration (no attackers)"
    echo "  lowerbound       - Ego-only perception"
    echo "  no_defense       - CP with attackers, no defense"
    echo "  robosac          - ROBOSAC defense baseline"
    echo "  made             - MADE defense baseline"
    echo "  gcp              - GCP defense (our method)"
    echo "  gated_late_fusion - Gated Late Fusion baseline (consistency voting)"
    echo ""
    echo -e "${YELLOW}Attack Options:${NC}"
    echo "  --adv_method METHOD      - pgd/bim/cw/simple_bac/bac (default: pgd)"
    echo "  --eps EPS                - Attack epsilon (default: 0.5)"
    echo "  --adv_iter N             - Attack iterations (default: 15)"
    echo "  --num_attackers N        - Number of attackers (default: 1)"
    echo "  --attack_mode MODE       - Random/Poisson/SI (default: Random)"
    echo "  --attack_ratio RATIO     - Attack frame ratio (default: 0.6)"
    echo "  --bac_eps EPS            - BAC attack threshold (default: 0.3)"
    echo "  --simple_bac_eps EPS     - Simple BAC threshold (default: 0.3)"
    echo ""
    echo -e "${YELLOW}Defense Options:${NC}"
    echo "  --history_length N       - Temporal sequence length (default: 5)"
    echo "  --recon_threshold T      - LSTM-AE threshold (default: 0.8)"
    echo "  --box_match_thresh T     - IoU threshold (default: 0.3)"
    echo "  --step_budget N          - ROBOSAC step budget (default: 3)"
    echo "  --gated_fusion_iou T     - Gated fusion IoU threshold (default: 0.3)"
    echo "  --gated_fusion_votes N   - Gated fusion min votes (default: 2)"
    echo ""
    echo -e "${YELLOW}Sample Control:${NC}"
    echo "  --scene_id ID            - Scene to test (default: 8)"
    echo "  --sample_start N         - Start frame ID (inclusive, default: 0)"
    echo "  --sample_end N           - End frame ID (inclusive, default: last frame)"
    echo "  --ego_agent ID           - Ego agent ID (default: 1)"
    echo ""
    echo -e "${YELLOW}Examples:${NC}"
    echo "  $0"
    echo "  $0 gcp"
    echo "  $0 no_defense --adv_method bac --eps 0.5 --num_attackers 2"
    echo "  $0 gcp --adv_method pgd --attack_mode Poisson --attack_ratio 0.6"
    echo "  $0 robosac --step_budget 5 --adv_method pgd"
    echo "  $0 gated_late_fusion --gated_fusion_iou 0.3 --gated_fusion_votes 2"
    echo ""
    echo -e "${YELLOW}Frame Range Examples:${NC}"
    echo "  $0 no_defense --scene_id 8 --sample_start 0 --sample_end 0    # Only frame 0"
    echo "  $0 gcp --scene_id 8 --sample_start 0 --sample_end 10          # Frames 0-10"
    echo "  $0 no_defense --sample_start 50 --sample_end 60               # Frames 50-60"
    echo "  $0 gcp --scene_id 96                                          # All frames (~100)"
    echo ""
    echo -e "${YELLOW}Note on Sample Control:${NC}"
    echo "  - No range: Tests ALL frames in scene (~100 frames)"
    echo "  - --sample_start N --sample_end N: Single frame N"
    echo "  - --sample_start N --sample_end M: Frame range [N, M] inclusive"
    echo "  - --sample_start N (no end): From frame N to end"
    echo "  - Scene 8, 96, 97 have 6 agents"
    echo ""
}

# Check if LSTM-AE model exists
function check_lstm_model() {
    if [ ! -f "$LSTM_MODEL" ]; then
        echo -e "${RED}[ERROR] LSTM-AE model not found: $LSTM_MODEL${NC}"
        echo -e "${YELLOW}[INFO] Please train LSTM-AE first:${NC}"
        echo "  cd ${PROJ_ROOT}"
        echo "  ./scripts/train_lstm_ae.sh --data_path /path/to/bev_flow_data.npz"
        exit 1
    fi
}

# Run GCP Defense
function run_gcp() {
    echo -e "${GREEN}[INFO] Running GCP Defense...${NC}"
    check_lstm_model
    
    local CMD="python gcp.py \
        --log \
        --gcp gcp \
        --data \"$DATA_PATH\" \
        --resume \"$MODEL_PATH\" \
        --logpath \"$LOG_PATH\" \
        --scene_id $SCENE_ID \
        --ego_agent $EGO_AGENT \
        --number_of_attackers $NUM_ATTACKERS \
        --adv_method $ADV_METHOD \
        --eps $EPS \
        --adv_iter $ADV_ITER \
        --attack_mode $ATTACK_MODE \
        --attack_ratio $ATTACK_RATIO \
        --bac_eps $BAC_EPS \
        --simple_bac_eps $SIMPLE_BAC_EPS \
        --history_length $HISTORY_LENGTH \
        --reconstruction_threshold $RECON_THRESHOLD \
        --box_matching_thresh $BOX_MATCH_THRESH"
    
    # Add sample range if specified
    if [ -n "$SAMPLE_START" ]; then
        CMD="$CMD --sample_start $SAMPLE_START"
    fi
    
    if [ -n "$SAMPLE_END" ]; then
        CMD="$CMD --sample_end $SAMPLE_END"
    fi
    
    # Legacy support for --sample_id
    if [ -n "$SAMPLE_ID" ]; then
        CMD="$CMD --sample_id $SAMPLE_ID"
    fi
    
    # Add extra arguments
    if [ -n "$EXTRA_ARGS" ]; then
        CMD="$CMD $EXTRA_ARGS"
    fi
    
    eval $CMD
}

# Run Upperbound (Clean environment, all agents collaborate)
function run_upperbound() {
    echo -e "${GREEN}[INFO] Running Upperbound (Clean Collaboration)...${NC}"
    
    local CMD="python gcp.py \
        --log \
        --gcp upperbound \
        --data \"$DATA_PATH\" \
        --resume \"$MODEL_PATH\" \
        --logpath \"$LOG_PATH\" \
        --scene_id $SCENE_ID \
        --ego_agent $EGO_AGENT"
    
    if [ -n "$SAMPLE_START" ]; then
        CMD="$CMD --sample_start $SAMPLE_START"
    fi
    
    if [ -n "$SAMPLE_END" ]; then
        CMD="$CMD --sample_end $SAMPLE_END"
    fi
    
    if [ -n "$SAMPLE_ID" ]; then
        CMD="$CMD --sample_id $SAMPLE_ID"
    fi
    
    if [ -n "$EXTRA_ARGS" ]; then
        CMD="$CMD $EXTRA_ARGS"
    fi
    
    eval $CMD
}

# Run Lowerbound (Ego-only)
function run_lowerbound() {
    echo -e "${GREEN}[INFO] Running Lowerbound (Ego-only)...${NC}"
    
    local CMD="python gcp.py \
        --log \
        --gcp lowerbound \
        --data \"$DATA_PATH\" \
        --resume \"$MODEL_PATH\" \
        --logpath \"$LOG_PATH\" \
        --scene_id $SCENE_ID \
        --ego_agent $EGO_AGENT"
    
    if [ -n "$SAMPLE_ID" ]; then
        CMD="$CMD --sample_id $SAMPLE_ID"
    fi
    
    if [ -n "$EXTRA_ARGS" ]; then
        CMD="$CMD $EXTRA_ARGS"
    fi
    
    eval $CMD
}

# Run No Defense (with attackers, no defense)
function run_no_defense() {
    echo -e "${GREEN}[INFO] Running No Defense (with attackers)...${NC}"
    
    local CMD="python gcp.py \
        --log \
        --gcp no_defense \
        --data \"$DATA_PATH\" \
        --resume \"$MODEL_PATH\" \
        --logpath \"$LOG_PATH\" \
        --scene_id $SCENE_ID \
        --ego_agent $EGO_AGENT \
        --number_of_attackers $NUM_ATTACKERS \
        --adv_method $ADV_METHOD \
        --eps $EPS \
        --adv_iter $ADV_ITER \
        --attack_mode $ATTACK_MODE \
        --attack_ratio $ATTACK_RATIO \
        --bac_eps $BAC_EPS \
        --simple_bac_eps $SIMPLE_BAC_EPS"
    
    if [ -n "$SAMPLE_START" ]; then
        CMD="$CMD --sample_start $SAMPLE_START"
    fi
    
    if [ -n "$SAMPLE_END" ]; then
        CMD="$CMD --sample_end $SAMPLE_END"
    fi
    
    if [ -n "$SAMPLE_ID" ]; then
        CMD="$CMD --sample_id $SAMPLE_ID"
    fi
    
    if [ -n "$EXTRA_ARGS" ]; then
        CMD="$CMD $EXTRA_ARGS"
    fi
    
    eval $CMD
}

# Run ROBOSAC
function run_robosac() {
    echo -e "${GREEN}[INFO] Running ROBOSAC Defense...${NC}"
    
    # Use default step budget if not set
    local STEP_BUDGET=${STEP_BUDGET:-3}
    
    local CMD="python gcp.py \
        --log \
        --gcp robosac_mAP \
        --data \"$DATA_PATH\" \
        --resume \"$MODEL_PATH\" \
        --logpath \"$LOG_PATH\" \
        --scene_id $SCENE_ID \
        --ego_agent $EGO_AGENT \
        --number_of_attackers $NUM_ATTACKERS \
        --adv_method $ADV_METHOD \
        --eps $EPS \
        --adv_iter $ADV_ITER \
        --attack_mode $ATTACK_MODE \
        --attack_ratio $ATTACK_RATIO \
        --bac_eps $BAC_EPS \
        --simple_bac_eps $SIMPLE_BAC_EPS \
        --step_budget $STEP_BUDGET \
        --box_matching_thresh $BOX_MATCH_THRESH"
    
    if [ -n "$SAMPLE_START" ]; then
        CMD="$CMD --sample_start $SAMPLE_START"
    fi
    
    if [ -n "$SAMPLE_END" ]; then
        CMD="$CMD --sample_end $SAMPLE_END"
    fi
    
    if [ -n "$SAMPLE_ID" ]; then
        CMD="$CMD --sample_id $SAMPLE_ID"
    fi
    
    if [ -n "$EXTRA_ARGS" ]; then
        CMD="$CMD $EXTRA_ARGS"
    fi
    
    eval $CMD
}

# Run MADE
function run_made() {
    echo -e "${GREEN}[INFO] Running MADE Defense...${NC}"
    
    local CMD="python gcp.py \
        --log \
        --gcp made \
        --data \"$DATA_PATH\" \
        --resume \"$MODEL_PATH\" \
        --logpath \"$LOG_PATH\" \
        --scene_id $SCENE_ID \
        --ego_agent $EGO_AGENT \
        --number_of_attackers $NUM_ATTACKERS \
        --adv_method $ADV_METHOD \
        --eps $EPS \
        --adv_iter $ADV_ITER \
        --attack_mode $ATTACK_MODE \
        --attack_ratio $ATTACK_RATIO \
        --bac_eps $BAC_EPS \
        --simple_bac_eps $SIMPLE_BAC_EPS \
        --box_matching_thresh $BOX_MATCH_THRESH"
    
    if [ -n "$SAMPLE_START" ]; then
        CMD="$CMD --sample_start $SAMPLE_START"
    fi
    
    if [ -n "$SAMPLE_END" ]; then
        CMD="$CMD --sample_end $SAMPLE_END"
    fi
    
    if [ -n "$SAMPLE_ID" ]; then
        CMD="$CMD --sample_id $SAMPLE_ID"
    fi
    
    if [ -n "$EXTRA_ARGS" ]; then
        CMD="$CMD $EXTRA_ARGS"
    fi
    
    eval $CMD
}

function run_gated_late_fusion() {
    echo -e "${GREEN}[INFO] Running Gated Late Fusion Baseline...${NC}"
    
    local CMD="python gcp.py \
        --log \
        --gcp gated_late_fusion \
        --data \"$DATA_PATH\" \
        --resume \"$MODEL_PATH\" \
        --logpath \"$LOG_PATH\" \
        --scene_id $SCENE_ID \
        --ego_agent $EGO_AGENT \
        --number_of_attackers $NUM_ATTACKERS \
        --adv_method $ADV_METHOD \
        --eps $EPS \
        --adv_iter $ADV_ITER \
        --attack_mode $ATTACK_MODE \
        --attack_ratio $ATTACK_RATIO \
        --gated_fusion_iou $GATED_FUSION_IOU \
        --gated_fusion_votes $GATED_FUSION_VOTES"
    
    if [ -n "$SAMPLE_START" ]; then
        CMD="$CMD --sample_start $SAMPLE_START"
    fi
    
    if [ -n "$SAMPLE_END" ]; then
        CMD="$CMD --sample_end $SAMPLE_END"
    fi
    
    if [ -n "$SAMPLE_ID" ]; then
        CMD="$CMD --sample_id $SAMPLE_ID"
    fi
    
    if [ -n "$EXTRA_ARGS" ]; then
        CMD="$CMD $EXTRA_ARGS"
    fi
    
    eval $CMD
}

# Parse command line arguments
function parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --adv_method)
                ADV_METHOD="$2"
                shift 2
                ;;
            --eps)
                EPS="$2"
                shift 2
                ;;
            --adv_iter)
                ADV_ITER="$2"
                shift 2
                ;;
            --num_attackers)
                NUM_ATTACKERS="$2"
                shift 2
                ;;
            --attack_mode)
                ATTACK_MODE="$2"
                shift 2
                ;;
            --attack_ratio)
                ATTACK_RATIO="$2"
                shift 2
                ;;
            --bac_eps)
                BAC_EPS="$2"
                shift 2
                ;;
            --simple_bac_eps)
                SIMPLE_BAC_EPS="$2"
                shift 2
                ;;
            --history_length)
                HISTORY_LENGTH="$2"
                shift 2
                ;;
            --recon_threshold)
                RECON_THRESHOLD="$2"
                shift 2
                ;;
            --box_match_thresh)
                BOX_MATCH_THRESH="$2"
                shift 2
                ;;
            --gated_fusion_iou)
                GATED_FUSION_IOU="$2"
                shift 2
                ;;
            --gated_fusion_votes)
                GATED_FUSION_VOTES="$2"
                shift 2
                ;;
            --step_budget)
                STEP_BUDGET="$2"
                shift 2
                ;;
            --scene_id)
                SCENE_ID="$2"
                shift 2
                ;;
            --sample_id)
                SAMPLE_ID="$2"
                shift 2
                ;;
            --sample_start)
                SAMPLE_START="$2"
                shift 2
                ;;
            --sample_end)
                SAMPLE_END="$2"
                shift 2
                ;;
            --ego_agent)
                EGO_AGENT="$2"
                shift 2
                ;;
            *)
                # Unknown option, keep for passing to python script
                EXTRA_ARGS="$EXTRA_ARGS $1"
                shift
                ;;
        esac
    done
}

# Main function
function main() {
    local FIRST_ARG="${1:-}"
    case "$FIRST_ARG" in
        gcp|upperbound|lowerbound|no_defense|robosac|made|gated_late_fusion)
            MODE="$FIRST_ARG"
            shift
            ;;
        -h|--help|help)
            show_usage
            exit 0
            ;;
        *)
            MODE="gcp"
            ;;
    esac
    
    # Parse remaining arguments
    parse_args "$@"
    
    # Show configuration
    echo -e "${GREEN}[CONFIG] Mode: $MODE${NC}"
    echo -e "${GREEN}[CONFIG] Attack: $ADV_METHOD, eps=$EPS, iter=$ADV_ITER${NC}"
    echo -e "${GREEN}[CONFIG] Attack Pattern: $ATTACK_MODE (ratio=$ATTACK_RATIO)${NC}"
    
    # Display sample range
    if [ -n "$SAMPLE_START" ] && [ -n "$SAMPLE_END" ]; then
        echo -e "${GREEN}[CONFIG] Scene: $SCENE_ID, Frames: [$SAMPLE_START, $SAMPLE_END], Ego: $EGO_AGENT${NC}"
    elif [ -n "$SAMPLE_START" ]; then
        echo -e "${GREEN}[CONFIG] Scene: $SCENE_ID, Frames: [$SAMPLE_START, end], Ego: $EGO_AGENT${NC}"
    elif [ -n "$SAMPLE_ID" ]; then
        echo -e "${GREEN}[CONFIG] Scene: $SCENE_ID, Frames: [$SAMPLE_ID, end], Ego: $EGO_AGENT${NC}"
    else
        echo -e "${GREEN}[CONFIG] Scene: $SCENE_ID, Frames: [all], Ego: $EGO_AGENT${NC}"
    fi
    echo ""
    
    case "$MODE" in
        gcp)
            run_gcp
            ;;
        upperbound)
            run_upperbound
            ;;
        lowerbound)
            run_lowerbound
            ;;
        no_defense)
            run_no_defense
            ;;
        robosac)
            run_robosac
            ;;
        made)
            run_made
            ;;
        gated_late_fusion)
            run_gated_late_fusion
            ;;
        -h|--help|help)
            show_usage
            ;;
        *)
            echo -e "${RED}[ERROR] Unknown mode: $MODE${NC}"
            show_usage
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
