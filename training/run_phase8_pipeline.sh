#!/bin/bash
#===============================================================================
# Phase 8 Data Pipeline - Full Automation Script
# Runs all data generation tasks in parallel and merges results
#===============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASETS_DIR="${SCRIPT_DIR}/datasets"
LOG_DIR="${SCRIPT_DIR}/logs/phase8_pipeline"
VENV_PATH="${SCRIPT_DIR}/venv_yolo12"

# Output directories
DEEPSCORES_OUTPUT="${DATASETS_DIR}/yolo_deepscores_dynamics"
SYNTHETIC_OUTPUT="${DATASETS_DIR}/yolo_synthetic_phase8"
OPENSCORE_LIEDER_OUTPUT="${DATASETS_DIR}/yolo_openscore_lieder"
OPENSCORE_QUARTETS_OUTPUT="${DATASETS_DIR}/yolo_openscore_quartets"
PHASE8_OUTPUT="${DATASETS_DIR}/yolo_harmony_v2_phase8"

# Create directories
mkdir -p "${LOG_DIR}"
mkdir -p "${DEEPSCORES_OUTPUT}"
mkdir -p "${SYNTHETIC_OUTPUT}"
mkdir -p "${OPENSCORE_LIEDER_OUTPUT}"
mkdir -p "${OPENSCORE_QUARTETS_OUTPUT}"
mkdir -p "${PHASE8_OUTPUT}"

# Timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

log_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARN:${NC} $1"
}

log_task() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] TASK:${NC} $1"
}

#===============================================================================
# TASK 1: DeepScores V2 Dynamics Conversion
#===============================================================================
run_deepscores_conversion() {
    log_task "Starting DeepScores V2 Dynamics Conversion..."

    cd "${SCRIPT_DIR}"
    source "${VENV_PATH}/bin/activate"

    python convert_deepscores_dynamics_to_yolo.py \
        2>&1 | tee "${LOG_DIR}/deepscores_${TIMESTAMP}.log"

    if [ -d "${DEEPSCORES_OUTPUT}/images" ]; then
        local count=$(find "${DEEPSCORES_OUTPUT}/images" -name "*.png" | wc -l)
        log "DeepScores conversion complete: ${count} images"
    else
        log_error "DeepScores conversion failed"
        return 1
    fi
}

#===============================================================================
# TASK 2: LilyPond Synthetic Data Generation
#===============================================================================
run_synthetic_generation() {
    log_task "Starting LilyPond Synthetic Data Generation..."

    cd "${SCRIPT_DIR}"
    source "${VENV_PATH}/bin/activate"

    # Install scipy if needed for connected component analysis
    pip install scipy --quiet 2>/dev/null || true

    # Generate double_flat samples (class 17)
    log "Generating double_flat samples..."
    python lilypond_synthetic_generator.py \
        --class 17 \
        --count 3000 \
        --output "${SYNTHETIC_OUTPUT}" \
        2>&1 | tee "${LOG_DIR}/synthetic_doubleflat_${TIMESTAMP}.log"

    # Generate dynamic_loud samples (class 31)
    log "Generating dynamic_loud samples..."
    python lilypond_synthetic_generator.py \
        --class 31 \
        --count 3000 \
        --output "${SYNTHETIC_OUTPUT}" \
        2>&1 | tee "${LOG_DIR}/synthetic_dynamicloud_${TIMESTAMP}.log"

    if [ -d "${SYNTHETIC_OUTPUT}/images" ]; then
        local count=$(find "${SYNTHETIC_OUTPUT}/images" -name "*.png" 2>/dev/null | wc -l)
        log "Synthetic generation complete: ${count} images"
    else
        log_warn "Synthetic generation may have issues"
    fi
}

#===============================================================================
# TASK 3: OpenScore Lieder Rendering
#===============================================================================
run_openscore_lieder() {
    log_task "Starting OpenScore Lieder Rendering..."

    cd "${SCRIPT_DIR}"
    source "${VENV_PATH}/bin/activate"

    local lieder_path="${DATASETS_DIR}/external/omr_downloads/OpenScoreLieder/Lieder-main/scores"

    if [ ! -d "${lieder_path}" ]; then
        log_error "OpenScore Lieder not found at ${lieder_path}"
        return 1
    fi

    python render_openscore_with_musescore.py \
        --input "${lieder_path}" \
        --output "${OPENSCORE_LIEDER_OUTPUT}" \
        --workers 6 \
        2>&1 | tee "${LOG_DIR}/openscore_lieder_${TIMESTAMP}.log"

    if [ -d "${OPENSCORE_LIEDER_OUTPUT}/images" ]; then
        local count=$(find "${OPENSCORE_LIEDER_OUTPUT}/images" -name "*.png" 2>/dev/null | wc -l)
        log "OpenScore Lieder complete: ${count} images"
    else
        log_warn "OpenScore Lieder rendering may have issues"
    fi
}

#===============================================================================
# TASK 4: OpenScore Quartets Rendering
#===============================================================================
run_openscore_quartets() {
    log_task "Starting OpenScore Quartets Rendering..."

    cd "${SCRIPT_DIR}"
    source "${VENV_PATH}/bin/activate"

    local quartets_path="${DATASETS_DIR}/external/omr_downloads/OpenScoreStringQuartets"

    if [ ! -d "${quartets_path}" ]; then
        log_error "OpenScore Quartets not found at ${quartets_path}"
        return 1
    fi

    python render_openscore_with_musescore.py \
        --input "${quartets_path}" \
        --output "${OPENSCORE_QUARTETS_OUTPUT}" \
        --workers 4 \
        2>&1 | tee "${LOG_DIR}/openscore_quartets_${TIMESTAMP}.log"

    if [ -d "${OPENSCORE_QUARTETS_OUTPUT}/images" ]; then
        local count=$(find "${OPENSCORE_QUARTETS_OUTPUT}/images" -name "*.png" 2>/dev/null | wc -l)
        log "OpenScore Quartets complete: ${count} images"
    else
        log_warn "OpenScore Quartets rendering may have issues"
    fi
}

#===============================================================================
# TASK 5: Merge All Datasets
#===============================================================================
merge_datasets() {
    log_task "Merging all datasets into Phase 8..."

    cd "${SCRIPT_DIR}"
    source "${VENV_PATH}/bin/activate"

    # Create Phase 8 structure
    mkdir -p "${PHASE8_OUTPUT}/images/train"
    mkdir -p "${PHASE8_OUTPUT}/images/val"
    mkdir -p "${PHASE8_OUTPUT}/labels/train"
    mkdir -p "${PHASE8_OUTPUT}/labels/val"

    local total_train=0
    local total_val=0

    # Copy from Phase 7 (base dataset)
    log "Copying Phase 7 base dataset..."
    local phase7_dir="${DATASETS_DIR}/yolo_harmony_v2_phase7_ultimate"
    if [ -d "${phase7_dir}/train/images" ]; then
        cp -r "${phase7_dir}/train/images/"* "${PHASE8_OUTPUT}/images/train/" 2>/dev/null || true
        cp -r "${phase7_dir}/train/labels/"* "${PHASE8_OUTPUT}/labels/train/" 2>/dev/null || true
        cp -r "${phase7_dir}/val/images/"* "${PHASE8_OUTPUT}/images/val/" 2>/dev/null || true
        cp -r "${phase7_dir}/val/labels/"* "${PHASE8_OUTPUT}/labels/val/" 2>/dev/null || true
    fi

    # Merge DeepScores dynamics
    log "Merging DeepScores dynamics..."
    if [ -d "${DEEPSCORES_OUTPUT}/images/train" ]; then
        cp -r "${DEEPSCORES_OUTPUT}/images/train/"* "${PHASE8_OUTPUT}/images/train/" 2>/dev/null || true
        cp -r "${DEEPSCORES_OUTPUT}/labels/train/"* "${PHASE8_OUTPUT}/labels/train/" 2>/dev/null || true
    fi
    if [ -d "${DEEPSCORES_OUTPUT}/images/val" ]; then
        cp -r "${DEEPSCORES_OUTPUT}/images/val/"* "${PHASE8_OUTPUT}/images/val/" 2>/dev/null || true
        cp -r "${DEEPSCORES_OUTPUT}/labels/val/"* "${PHASE8_OUTPUT}/labels/val/" 2>/dev/null || true
    fi

    # Merge synthetic data (90% train, 10% val)
    log "Merging synthetic data..."
    if [ -d "${SYNTHETIC_OUTPUT}/images" ]; then
        local synth_files=($(find "${SYNTHETIC_OUTPUT}/images" -name "*.png" 2>/dev/null))
        local synth_count=${#synth_files[@]}
        local val_count=$((synth_count / 10))

        for i in "${!synth_files[@]}"; do
            local img="${synth_files[$i]}"
            local basename=$(basename "$img" .png)
            local label="${SYNTHETIC_OUTPUT}/labels/${basename}.txt"

            if [ $i -lt $val_count ]; then
                cp "$img" "${PHASE8_OUTPUT}/images/val/" 2>/dev/null || true
                [ -f "$label" ] && cp "$label" "${PHASE8_OUTPUT}/labels/val/" 2>/dev/null || true
            else
                cp "$img" "${PHASE8_OUTPUT}/images/train/" 2>/dev/null || true
                [ -f "$label" ] && cp "$label" "${PHASE8_OUTPUT}/labels/train/" 2>/dev/null || true
            fi
        done
    fi

    # Merge OpenScore Lieder (90% train, 10% val)
    log "Merging OpenScore Lieder..."
    if [ -d "${OPENSCORE_LIEDER_OUTPUT}/images/train" ]; then
        local lieder_files=($(find "${OPENSCORE_LIEDER_OUTPUT}/images/train" -name "*.png" 2>/dev/null))
        local lieder_count=${#lieder_files[@]}
        local val_count=$((lieder_count / 10))

        for i in "${!lieder_files[@]}"; do
            local img="${lieder_files[$i]}"
            local basename=$(basename "$img" .png)
            local label="${OPENSCORE_LIEDER_OUTPUT}/labels/train/${basename}.txt"

            if [ $i -lt $val_count ]; then
                cp "$img" "${PHASE8_OUTPUT}/images/val/" 2>/dev/null || true
                [ -f "$label" ] && cp "$label" "${PHASE8_OUTPUT}/labels/val/" 2>/dev/null || true
            else
                cp "$img" "${PHASE8_OUTPUT}/images/train/" 2>/dev/null || true
                [ -f "$label" ] && cp "$label" "${PHASE8_OUTPUT}/labels/train/" 2>/dev/null || true
            fi
        done
    fi

    # Merge OpenScore Quartets
    log "Merging OpenScore Quartets..."
    if [ -d "${OPENSCORE_QUARTETS_OUTPUT}/images/train" ]; then
        cp -r "${OPENSCORE_QUARTETS_OUTPUT}/images/train/"* "${PHASE8_OUTPUT}/images/train/" 2>/dev/null || true
        cp -r "${OPENSCORE_QUARTETS_OUTPUT}/labels/train/"* "${PHASE8_OUTPUT}/labels/train/" 2>/dev/null || true
    fi

    # Count final results
    total_train=$(find "${PHASE8_OUTPUT}/images/train" -name "*.png" 2>/dev/null | wc -l)
    total_val=$(find "${PHASE8_OUTPUT}/images/val" -name "*.png" 2>/dev/null | wc -l)

    log "Phase 8 dataset created:"
    log "  - Training images: ${total_train}"
    log "  - Validation images: ${total_val}"
    log "  - Total: $((total_train + total_val))"

    # Create YAML config
    cat > "${PHASE8_OUTPUT}/harmony_phase8.yaml" << EOF
# Phase 8 Harmony OMR Dataset
# Generated: $(date)
path: ${PHASE8_OUTPUT}
train: images/train
val: images/val

nc: 33
names:
  0: notehead_filled
  1: notehead_hollow
  2: stem
  3: beam
  4: flag_8th
  5: flag_16th
  6: flag_32nd
  7: augmentation_dot
  8: tie
  9: clef_treble
  10: clef_bass
  11: clef_alto
  12: clef_tenor
  13: accidental_sharp
  14: accidental_flat
  15: accidental_natural
  16: accidental_double_sharp
  17: accidental_double_flat
  18: rest_whole
  19: rest_half
  20: rest_quarter
  21: rest_8th
  22: rest_16th
  23: barline
  24: barline_double
  25: barline_final
  26: barline_repeat
  27: time_signature
  28: key_signature
  29: fermata
  30: dynamic_soft
  31: dynamic_loud
  32: ledger_line

# Dataset composition:
# - Phase 7 base: ~25,000 images
# - DeepScores V2 dynamics: ~855 images (8,882 dynamic annotations)
# - LilyPond synthetic: ~6,000 images (double_flat, dynamic_loud)
# - OpenScore Lieder: ~4,000+ images (barlines, fermatas)
# - OpenScore Quartets: ~1,500+ images (barlines)
EOF

    log "YAML config created: ${PHASE8_OUTPUT}/harmony_phase8.yaml"
}

#===============================================================================
# MAIN EXECUTION
#===============================================================================
main() {
    log "=========================================="
    log "Phase 8 Data Pipeline Starting"
    log "=========================================="
    log "Timestamp: ${TIMESTAMP}"
    log "Log directory: ${LOG_DIR}"
    log ""

    # Check prerequisites
    if [ ! -d "${VENV_PATH}" ]; then
        log_error "Python venv not found at ${VENV_PATH}"
        exit 1
    fi

    # Activate venv and install dependencies
    source "${VENV_PATH}/bin/activate"
    pip install scipy pillow --quiet 2>/dev/null || true

    # Run tasks in parallel where possible
    log "Starting parallel tasks..."

    # Task 1 & 2 can run in parallel (different data sources)
    run_deepscores_conversion &
    DEEPSCORES_PID=$!

    run_synthetic_generation &
    SYNTHETIC_PID=$!

    # Task 3 & 4 need MuseScore (run sequentially to avoid resource contention)
    # But we can start them after a delay
    sleep 5
    run_openscore_lieder &
    LIEDER_PID=$!

    sleep 10
    run_openscore_quartets &
    QUARTETS_PID=$!

    # Wait for all tasks
    log "Waiting for all tasks to complete..."

    wait $DEEPSCORES_PID
    log "DeepScores conversion finished (exit code: $?)"

    wait $SYNTHETIC_PID
    log "Synthetic generation finished (exit code: $?)"

    wait $LIEDER_PID
    log "OpenScore Lieder finished (exit code: $?)"

    wait $QUARTETS_PID
    log "OpenScore Quartets finished (exit code: $?)"

    # Merge all datasets
    merge_datasets

    log "=========================================="
    log "Phase 8 Data Pipeline Complete!"
    log "=========================================="
    log ""
    log "Output: ${PHASE8_OUTPUT}"
    log "Logs: ${LOG_DIR}"
    log ""
    log "To start training, run:"
    log "  python yolo12_train_phase8.py"
}

# Run main function
main "$@"
