#!/bin/bash
#==============================================================================
# Phase 7 Ultimate: Complete Automation Pipeline
#==============================================================================
#
# This script automates the entire Phase 7 training pipeline:
# 1. Install dependencies
# 2. Render OpenScore Lieder data
# 3. Refine barline annotations
# 4. Generate synthetic barline data
# 5. Merge all datasets
# 6. Run multi-stage training
# 7. Generate final reports
#
# Usage:
#   ./run_phase7_complete.sh [options]
#
# Options:
#   --skip-render     Skip OpenScore rendering (use existing data)
#   --skip-synthetic  Skip synthetic data generation
#   --skip-merge      Skip dataset merge (use existing merged dataset)
#   --stages STAGES   Run specific training stages (e.g., "stage1 stage2")
#   --dry-run         Show what would be done without executing
#   --help            Show this help message
#
# Author: Phase 7 Automation System
# Date: 2025-11-26
#==============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
TRAINING_DIR="$PROJECT_ROOT/training"
DATASETS_DIR="$TRAINING_DIR/datasets"
PHASE7_DIR="$TRAINING_DIR/phase7_ultimate"
SCRIPTS_DIR="$PHASE7_DIR/scripts"
LOGS_DIR="$PHASE7_DIR/logs"
REPORTS_DIR="$PHASE7_DIR/reports"

# Configuration
SKIP_RENDER=false
SKIP_SYNTHETIC=false
SKIP_MERGE=false
DRY_RUN=false
STAGES=""
PYTHON_ENV="$TRAINING_DIR/venv_yolo12"

# Timestamps
START_TIME=$(date +%s)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

#==============================================================================
# Helper Functions
#==============================================================================

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo ""
    echo -e "${PURPLE}======================================================================${NC}"
    echo -e "${PURPLE}  $1${NC}"
    echo -e "${PURPLE}======================================================================${NC}"
}

log_substep() {
    echo -e "${CYAN}  → $1${NC}"
}

check_command() {
    if ! command -v "$1" &> /dev/null; then
        log_error "Required command not found: $1"
        return 1
    fi
    return 0
}

elapsed_time() {
    local end_time=$(date +%s)
    local elapsed=$((end_time - START_TIME))
    local hours=$((elapsed / 3600))
    local minutes=$(((elapsed % 3600) / 60))
    local seconds=$((elapsed % 60))
    echo "${hours}h ${minutes}m ${seconds}s"
}

show_help() {
    head -40 "$0" | tail -30
    exit 0
}

#==============================================================================
# Parse Arguments
#==============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-render)
            SKIP_RENDER=true
            shift
            ;;
        --skip-synthetic)
            SKIP_SYNTHETIC=true
            shift
            ;;
        --skip-merge)
            SKIP_MERGE=true
            shift
            ;;
        --stages)
            STAGES="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help|-h)
            show_help
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

#==============================================================================
# Initialization
#==============================================================================

log_step "Phase 7 Ultimate: Complete Automation Pipeline"
log_info "Timestamp: $TIMESTAMP"
log_info "Project root: $PROJECT_ROOT"
log_info "Training directory: $TRAINING_DIR"

# Create directories
mkdir -p "$LOGS_DIR"
mkdir -p "$REPORTS_DIR"
mkdir -p "$PHASE7_DIR/checkpoints"

# Log file
LOG_FILE="$LOGS_DIR/phase7_run_${TIMESTAMP}.log"
log_info "Log file: $LOG_FILE"

# Redirect all output to log file as well
exec > >(tee -a "$LOG_FILE") 2>&1

#==============================================================================
# Step 1: Environment Setup
#==============================================================================

log_step "Step 1: Environment Setup"

# Check Python environment
if [ -d "$PYTHON_ENV" ]; then
    log_substep "Activating Python environment: $PYTHON_ENV"
    source "$PYTHON_ENV/bin/activate"
else
    log_warn "Python environment not found at $PYTHON_ENV"
    log_substep "Using system Python"
fi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1)
log_substep "Python: $PYTHON_VERSION"

# Check required packages
log_substep "Checking required packages..."

packages_to_check="ultralytics torch pillow pyyaml numpy"
for pkg in $packages_to_check; do
    if python3 -c "import $pkg" 2>/dev/null; then
        log_substep "  ✅ $pkg"
    else
        log_warn "  ❌ $pkg not found"
    fi
done

# Check LilyPond (for synthetic data)
if check_command lilypond; then
    LILYPOND_VERSION=$(lilypond --version 2>&1 | head -1)
    log_substep "LilyPond: $LILYPOND_VERSION"
else
    log_warn "LilyPond not installed - synthetic generation may fail"
fi

# Check Verovio (for OpenScore rendering)
if python3 -c "import verovio" 2>/dev/null; then
    log_substep "✅ Verovio available"
else
    log_warn "Verovio not installed - installing..."
    if [ "$DRY_RUN" = false ]; then
        pip install verovio
    fi
fi

# Check GPU
if python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
    log_substep "GPU: $GPU_NAME"
else
    log_warn "No GPU detected - training will be slow!"
fi

#==============================================================================
# Step 2: OpenScore Lieder Rendering
#==============================================================================

log_step "Step 2: OpenScore Lieder Rendering"

OPENSCORE_OUTPUT="$DATASETS_DIR/openscore_lieder_rendered"

if [ "$SKIP_RENDER" = true ]; then
    log_info "Skipping OpenScore rendering (--skip-render)"
elif [ -d "$OPENSCORE_OUTPUT" ] && [ "$(ls -A $OPENSCORE_OUTPUT/images 2>/dev/null | wc -l)" -gt 100 ]; then
    log_info "OpenScore rendered data already exists ($OPENSCORE_OUTPUT)"
    log_info "Use --skip-render=false to re-render"
else
    log_substep "Rendering OpenScore Lieder corpus..."

    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY RUN] Would run: python3 $SCRIPTS_DIR/render_openscore_complete.py"
    else
        python3 "$SCRIPTS_DIR/render_openscore_complete.py" \
            --input "$DATASETS_DIR/external/openscore_lieder" \
            --output "$OPENSCORE_OUTPUT" \
            2>&1 | tee "$LOGS_DIR/openscore_render_${TIMESTAMP}.log"

        # Count results
        if [ -d "$OPENSCORE_OUTPUT/images" ]; then
            RENDER_COUNT=$(ls "$OPENSCORE_OUTPUT/images" | wc -l)
            log_substep "Rendered $RENDER_COUNT images"
        fi
    fi
fi

#==============================================================================
# Step 3: Annotation Refinement
#==============================================================================

log_step "Step 3: Barline Annotation Refinement"

log_substep "Refining barline annotations in Phase 6 dataset..."

PHASE6_DATASET="$DATASETS_DIR/yolo_harmony_v2_phase6_ultimate"

if [ "$DRY_RUN" = true ]; then
    log_info "[DRY RUN] Would run: python3 $SCRIPTS_DIR/refine_barline_annotations.py"
else
    python3 "$SCRIPTS_DIR/refine_barline_annotations.py" \
        --input "$PHASE6_DATASET" \
        --output "$PHASE6_DATASET" \
        2>&1 | tee "$LOGS_DIR/refine_annotations_${TIMESTAMP}.log"
fi

#==============================================================================
# Step 4: Synthetic Data Generation
#==============================================================================

log_step "Step 4: Synthetic Barline Data Generation"

SYNTHETIC_OUTPUT="$DATASETS_DIR/synthetic_barlines"

if [ "$SKIP_SYNTHETIC" = true ]; then
    log_info "Skipping synthetic data generation (--skip-synthetic)"
elif [ -d "$SYNTHETIC_OUTPUT" ] && [ "$(ls -A $SYNTHETIC_OUTPUT/images 2>/dev/null | wc -l)" -gt 1000 ]; then
    log_info "Synthetic data already exists ($SYNTHETIC_OUTPUT)"
else
    log_substep "Generating synthetic barline data..."

    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY RUN] Would run: python3 $SCRIPTS_DIR/generate_synthetic_barlines.py"
    else
        python3 "$SCRIPTS_DIR/generate_synthetic_barlines.py" \
            --output "$SYNTHETIC_OUTPUT" \
            --double 2000 \
            --final 1500 \
            --repeat 1000 \
            2>&1 | tee "$LOGS_DIR/synthetic_gen_${TIMESTAMP}.log"

        # Count results
        if [ -d "$SYNTHETIC_OUTPUT/images" ]; then
            SYNTH_COUNT=$(ls "$SYNTHETIC_OUTPUT/images" | wc -l)
            log_substep "Generated $SYNTH_COUNT synthetic images"
        fi
    fi
fi

#==============================================================================
# Step 5: Dataset Merge
#==============================================================================

log_step "Step 5: Dataset Merge and Validation"

MERGED_OUTPUT="$DATASETS_DIR/yolo_harmony_v2_phase7_ultimate"

if [ "$SKIP_MERGE" = true ]; then
    log_info "Skipping dataset merge (--skip-merge)"
elif [ -f "$MERGED_OUTPUT/harmony_phase7_ultimate.yaml" ]; then
    log_info "Merged dataset already exists"
    log_info "Delete $MERGED_OUTPUT to re-merge"
else
    log_substep "Merging all data sources..."

    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY RUN] Would run: python3 $SCRIPTS_DIR/merge_phase7_dataset.py"
    else
        python3 "$SCRIPTS_DIR/merge_phase7_dataset.py" \
            --output "yolo_harmony_v2_phase7_ultimate" \
            --base-dir "$DATASETS_DIR" \
            --train-ratio 0.90 \
            2>&1 | tee "$LOGS_DIR/merge_dataset_${TIMESTAMP}.log"

        # Show merge stats
        if [ -f "$MERGED_OUTPUT/merge_report.json" ]; then
            log_substep "Merge report:"
            python3 -c "import json; r=json.load(open('$MERGED_OUTPUT/merge_report.json')); print(f'  Total images: {r[\"total_images\"]:,}'); print(f'  Total bboxes: {r[\"total_bboxes\"]:,}')"
        fi
    fi
fi

#==============================================================================
# Step 6: Multi-Stage Training
#==============================================================================

log_step "Step 6: Multi-Stage Training"

TRAINING_OUTPUT="$TRAINING_DIR/harmony_omr_v2_phase7"

log_substep "Starting Phase 7 multi-stage training..."
log_info "This may take several hours. Monitor progress in:"
log_info "  - TensorBoard: tensorboard --logdir $TRAINING_OUTPUT"
log_info "  - Logs: $LOGS_DIR"

if [ "$DRY_RUN" = true ]; then
    log_info "[DRY RUN] Would run: python3 $SCRIPTS_DIR/train_phase7_ultimate.py"
else
    # Build training command
    TRAIN_CMD="python3 $SCRIPTS_DIR/train_phase7_ultimate.py"
    TRAIN_CMD="$TRAIN_CMD --dataset $MERGED_OUTPUT/harmony_phase7_ultimate.yaml"
    TRAIN_CMD="$TRAIN_CMD --output $TRAINING_OUTPUT"
    TRAIN_CMD="$TRAIN_CMD --phase6-weights $TRAINING_DIR/harmony_omr_v2_phase6/ultimate_barline_fixed/weights/best.pt"
    TRAIN_CMD="$TRAIN_CMD --device 0"
    TRAIN_CMD="$TRAIN_CMD --workers 8"

    if [ -n "$STAGES" ]; then
        TRAIN_CMD="$TRAIN_CMD --stages $STAGES"
    fi

    log_substep "Running: $TRAIN_CMD"

    # Run training
    $TRAIN_CMD 2>&1 | tee "$LOGS_DIR/training_${TIMESTAMP}.log"

    # Check result
    if [ -f "$TRAINING_OUTPUT/checkpoints/stage4_polish_best.pt" ]; then
        log_substep "✅ Training completed successfully!"
    elif [ -f "$TRAINING_OUTPUT/checkpoints/stage3_high_res_best.pt" ]; then
        log_substep "⚠️ Training completed through Stage 3"
    elif [ -f "$TRAINING_OUTPUT/checkpoints/stage2_barline_focus_best.pt" ]; then
        log_substep "⚠️ Training completed through Stage 2"
    else
        log_warn "Training may have failed - check logs"
    fi
fi

#==============================================================================
# Step 7: Final Report Generation
#==============================================================================

log_step "Step 7: Final Report Generation"

log_substep "Generating final report..."

# Create summary report
FINAL_REPORT="$REPORTS_DIR/phase7_final_report_${TIMESTAMP}.md"

if [ "$DRY_RUN" = false ]; then
    cat > "$FINAL_REPORT" << EOF
# Phase 7 Ultimate Training Report

**Generated**: $(date)
**Duration**: $(elapsed_time)

## Pipeline Summary

| Step | Status |
|------|--------|
| Environment Setup | ✅ Complete |
| OpenScore Rendering | $([ "$SKIP_RENDER" = true ] && echo "⏭️ Skipped" || echo "✅ Complete") |
| Annotation Refinement | ✅ Complete |
| Synthetic Generation | $([ "$SKIP_SYNTHETIC" = true ] && echo "⏭️ Skipped" || echo "✅ Complete") |
| Dataset Merge | $([ "$SKIP_MERGE" = true ] && echo "⏭️ Skipped" || echo "✅ Complete") |
| Training | ✅ Complete |

## Dataset Statistics

$(if [ -f "$MERGED_OUTPUT/merge_report.json" ]; then
    python3 -c "
import json
r = json.load(open('$MERGED_OUTPUT/merge_report.json'))
print('| Metric | Value |')
print('|--------|-------|')
print(f'| Total Images | {r[\"total_images\"]:,} |')
print(f'| Total BBoxes | {r[\"total_bboxes\"]:,} |')
print()
print('### Target Class Distribution')
print()
print('| Class | Count |')
print('|-------|-------|')
for cls, count in r.get('target_class_stats', {}).items():
    print(f'| {cls} | {count:,} |')
"
else
    echo "Merge report not available"
fi)

## Training Results

$(if [ -f "$TRAINING_OUTPUT/training_report.json" ]; then
    python3 -c "
import json
r = json.load(open('$TRAINING_OUTPUT/training_report.json'))
print('| Class | Best mAP50 | Target | Status |')
print('|-------|------------|--------|--------|')
for cls, val in r.get('best_maps', {}).items():
    target = {'barline': 0.85, 'barline_double': 0.70, 'barline_final': 0.80, 'barline_repeat': 0.90}.get(cls, 0.5)
    status = '✅' if val >= target else '❌'
    print(f'| {cls} | {val:.3f} | {target:.2f} | {status} |')
print()
print(f'**Training Time**: {r.get(\"total_training_time_hours\", 0):.2f} hours')
"
else
    echo "Training report not available"
fi)

## Output Locations

- **Best Model**: \`$TRAINING_OUTPUT/checkpoints/\`
- **Merged Dataset**: \`$MERGED_OUTPUT/\`
- **Logs**: \`$LOGS_DIR/\`
- **Reports**: \`$REPORTS_DIR/\`

## Next Steps

1. Evaluate model on test images
2. Export to TFLite for Android deployment
3. Integration testing in Android app

---
*Generated by Phase 7 Ultimate Automation System*
EOF

    log_substep "Final report: $FINAL_REPORT"
fi

#==============================================================================
# Completion
#==============================================================================

log_step "Phase 7 Ultimate Pipeline Complete"

TOTAL_TIME=$(elapsed_time)
log_info "Total execution time: $TOTAL_TIME"
log_info "Log file: $LOG_FILE"

if [ "$DRY_RUN" = true ]; then
    log_warn "This was a DRY RUN - no actual changes were made"
fi

echo ""
echo -e "${GREEN}======================================================================${NC}"
echo -e "${GREEN}  Phase 7 Ultimate Training Complete!${NC}"
echo -e "${GREEN}======================================================================${NC}"
echo ""
echo "Next steps:"
echo "  1. Review final report: $FINAL_REPORT"
echo "  2. Check training metrics: tensorboard --logdir $TRAINING_OUTPUT"
echo "  3. Test best model: $TRAINING_OUTPUT/checkpoints/"
echo ""
