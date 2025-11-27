#!/bin/bash
################################################################################
# Phase 6 Training Launcher
################################################################################
#
# 一鍵啟動 Phase 6 多階段訓練
#
# 使用方法:
#   bash RUN_PHASE6.sh
#
# 或者給予執行權限後直接運行:
#   chmod +x RUN_PHASE6.sh
#   ./RUN_PHASE6.sh
#
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Header
echo -e "${BLUE}================================================================================================${NC}"
echo -e "${BLUE}                    Phase 6 Training: Hard Example Mining + Custom Loss${NC}"
echo -e "${BLUE}================================================================================================${NC}"
echo ""

# Check GPU
echo -e "${YELLOW}[1/6] Checking GPU availability...${NC}"
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}ERROR: nvidia-smi not found. CUDA GPU required.${NC}"
    exit 1
fi

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
GPU_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)

if [ "$GPU_MEM" -lt 8000 ]; then
    echo -e "${RED}WARNING: GPU memory < 8GB. Training may fail with OOM.${NC}"
    echo -e "${YELLOW}Consider reducing batch size in configs/phase6_config.yaml${NC}"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi
echo -e "${GREEN}✓ GPU available${NC}"
echo ""

# Check Phase 5 weights
echo -e "${YELLOW}[2/6] Checking Phase 5 weights...${NC}"
PHASE5_WEIGHTS="harmony_omr_v2_phase5/fermata_barline_enhanced/weights/best.pt"

if [ ! -f "$PHASE5_WEIGHTS" ]; then
    echo -e "${RED}ERROR: Phase 5 weights not found at: $PHASE5_WEIGHTS${NC}"

    # Try alternatives
    ALT1="harmony_omr_v2_phase4/enhanced_rare_class/weights/best.pt"
    ALT2="harmony_omr_v2_phase3/external_data_training/weights/best.pt"

    if [ -f "$ALT1" ]; then
        echo -e "${YELLOW}Found Phase 4 weights: $ALT1${NC}"
        echo -e "${YELLOW}Will use this as starting point.${NC}"
    elif [ -f "$ALT2" ]; then
        echo -e "${YELLOW}Found Phase 3 weights: $ALT2${NC}"
        echo -e "${YELLOW}Will use this as starting point.${NC}"
    else
        echo -e "${RED}No suitable base weights found!${NC}"
        exit 1
    fi
fi
echo -e "${GREEN}✓ Base weights found${NC}"
echo ""

# Check dataset
echo -e "${YELLOW}[3/6] Checking dataset...${NC}"
DATASET="datasets/yolo_harmony_v2_phase5/harmony_phase5.yaml"

if [ ! -f "$DATASET" ]; then
    echo -e "${RED}ERROR: Dataset config not found: $DATASET${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Dataset config found${NC}"
echo ""

# Check Python environment
echo -e "${YELLOW}[4/6] Checking Python environment...${NC}"
if ! python -c "import ultralytics" 2>/dev/null; then
    echo -e "${RED}ERROR: ultralytics not installed${NC}"
    echo -e "${YELLOW}Install with: pip install ultralytics${NC}"
    exit 1
fi

if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo -e "${RED}ERROR: PyTorch CUDA not available${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python environment ready${NC}"
echo ""

# Test components
echo -e "${YELLOW}[5/6] Testing Phase 6 components...${NC}"
if python custom_training/test_components.py all > /dev/null 2>&1; then
    echo -e "${GREEN}✓ All components passed tests${NC}"
else
    echo -e "${RED}ERROR: Component tests failed${NC}"
    echo -e "${YELLOW}Run manually to see details: python custom_training/test_components.py${NC}"
    exit 1
fi
echo ""

# Confirm start
echo -e "${BLUE}================================================================================================${NC}"
echo -e "${BLUE}                                Ready to Start Phase 6 Training${NC}"
echo -e "${BLUE}================================================================================================${NC}"
echo ""
echo -e "${YELLOW}Training Plan:${NC}"
echo -e "  Stage 1: Full dataset + Weighted loss (150 epochs, ~4-6 hours)"
echo -e "  HEM:     Hard example mining (~30-60 minutes)"
echo -e "  Stage 2: Hard example fine-tuning (50 epochs, ~1-2 hours)"
echo -e "  ${BLUE}Total estimated time: 6-9 hours${NC}"
echo ""
echo -e "${YELLOW}Expected Improvements:${NC}"
echo -e "  barline (23):        mAP50 0.201 → 0.50-0.60  (+150-200%)"
echo -e "  barline_double (24): mAP50 0.140 → 0.40-0.50  (+185-260%)"
echo -e "  Overall mAP50:       0.580 → 0.65-0.68        (+12-17%)"
echo ""
echo -e "${YELLOW}Output Directory:${NC} harmony_omr_v2_phase6/"
echo ""

# Final confirmation
read -p "Start training now? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Training cancelled.${NC}"
    exit 0
fi

# Start training
echo ""
echo -e "${BLUE}================================================================================================${NC}"
echo -e "${BLUE}[6/6] Starting Phase 6 Training...${NC}"
echo -e "${BLUE}================================================================================================${NC}"
echo ""
echo -e "${GREEN}Training started at: $(date)${NC}"
echo ""
echo -e "${YELLOW}Tip: Monitor training in another terminal with:${NC}"
echo -e "  tail -f harmony_omr_v2_phase6/stage1_weighted_loss/train.log"
echo -e "  watch -n 1 nvidia-smi"
echo ""
echo -e "${BLUE}================================================================================================${NC}"
echo ""

# Run training
python custom_training/train_phase6.py

# Check result
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${BLUE}================================================================================================${NC}"
    echo -e "${GREEN}                           Phase 6 Training Completed Successfully!${NC}"
    echo -e "${BLUE}================================================================================================${NC}"
    echo ""
    echo -e "${GREEN}Finished at: $(date)${NC}"
    echo ""
    echo -e "${YELLOW}Results saved to:${NC}"
    echo -e "  harmony_omr_v2_phase6/stage1_weighted_loss/weights/best.pt  (Stage 1)"
    echo -e "  harmony_omr_v2_phase6/stage2_hard_examples/weights/best.pt  (Final model)"
    echo -e "  harmony_omr_v2_phase6/phase6_summary.json                   (Summary)"
    echo ""
    echo -e "${YELLOW}Next steps:${NC}"
    echo -e "  1. Check final metrics in phase6_summary.json"
    echo -e "  2. Compare with Phase 5 results"
    echo -e "  3. If targets met: proceed to TFLite export"
    echo -e "  4. If targets not met: check analysis and plan Phase 6.1"
    echo ""
else
    echo ""
    echo -e "${RED}================================================================================================${NC}"
    echo -e "${RED}                               Training Failed or Interrupted${NC}"
    echo -e "${RED}================================================================================================${NC}"
    echo ""
    echo -e "${YELLOW}Check logs for errors:${NC}"
    echo -e "  harmony_omr_v2_phase6/stage1_weighted_loss/train.log"
    echo ""
    echo -e "${YELLOW}Common issues:${NC}"
    echo -e "  - OOM: Reduce batch size in configs/phase6_config.yaml"
    echo -e "  - Interrupted: Can resume by commenting out completed stages in train_phase6.py"
    echo ""
    exit 1
fi

echo -e "${BLUE}================================================================================================${NC}"
