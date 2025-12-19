#!/bin/bash
#===============================================================================
# Phase 9 完整執行管道 (修正版)
#===============================================================================
#
# 修正說明:
#   原計劃: 清理 tiny bbox → 訓練
#   修正後: 合併未使用數據 → 訓練
#
# 原因:
#   - Tiny bbox 與 mAP 相關係數僅 -0.143 (非常弱)
#   - barline_double 問題不在 tiny bbox (只有 0.2% tiny)
#   - 發現 8,726 張未使用圖片可大幅改善瓶頸類別
#
# 功能：
#   1. 合併未使用數據集
#   2. 驗證合併結果
#   3. 執行 Phase 9 訓練
#   4. 生成最終報告
#
# 執行方式：
#   chmod +x run_phase9_pipeline.sh
#   ./run_phase9_pipeline.sh
#
# 或背景執行（推薦）：
#   nohup ./run_phase9_pipeline.sh > phase9_pipeline.log 2>&1 &
#
#===============================================================================

set -e  # 遇到錯誤立即停止

# 顏色定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 配置
TRAINING_DIR="/home/thc1006/dev/music-app/training"
LOG_DIR="${TRAINING_DIR}/logs/phase9"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG="${LOG_DIR}/phase9_pipeline_${TIMESTAMP}.log"

# 創建日誌目錄
mkdir -p "${LOG_DIR}"

#-------------------------------------------------------------------------------
# 輔助函數
#-------------------------------------------------------------------------------

log() {
    local level=$1
    local message=$2
    local timestamp=$(date "+%Y-%m-%d %H:%M:%S")

    case $level in
        "INFO")  color=$GREEN ;;
        "WARN")  color=$YELLOW ;;
        "ERROR") color=$RED ;;
        "STEP")  color=$BLUE ;;
        *)       color=$NC ;;
    esac

    echo -e "${color}[${timestamp}] [${level}] ${message}${NC}"
    echo "[${timestamp}] [${level}] ${message}" >> "${MAIN_LOG}"
}

check_gpu() {
    log "INFO" "檢查 GPU 狀態..."

    if command -v nvidia-smi &> /dev/null; then
        gpu_info=$(nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null)
        if [ -n "$gpu_info" ]; then
            log "INFO" "GPU 資訊: $gpu_info"

            # 檢查 GPU 是否被佔用
            gpu_util=$(echo "$gpu_info" | cut -d',' -f4 | tr -d ' ')
            if [ "$gpu_util" -gt 50 ]; then
                log "WARN" "GPU 使用率 ${gpu_util}%，可能有其他程序在運行"
                read -p "是否繼續？(y/n): " confirm
                if [ "$confirm" != "y" ]; then
                    log "ERROR" "用戶取消執行"
                    exit 1
                fi
            fi
            return 0
        fi
    fi

    log "ERROR" "無法檢測到 GPU"
    return 1
}

send_notification() {
    local title=$1
    local message=$2

    # 如果有 notify-send（Linux 桌面通知）
    if command -v notify-send &> /dev/null; then
        notify-send "$title" "$message" 2>/dev/null || true
    fi

    # 終端響鈴
    echo -e "\a"
}

#-------------------------------------------------------------------------------
# Step 1: 合併未使用數據集 (修正: 不再清理 tiny bbox)
#-------------------------------------------------------------------------------

step1_merge_datasets() {
    log "STEP" "=========================================="
    log "STEP" "Step 1: 合併未使用數據集"
    log "STEP" "=========================================="

    cd "${TRAINING_DIR}"

    # 檢查合併腳本
    if [ ! -f "scripts/merge_phase9_datasets.py" ]; then
        log "ERROR" "合併腳本不存在: scripts/merge_phase9_datasets.py"
        exit 1
    fi

    log "INFO" "開始執行合併..."
    local merge_log="${LOG_DIR}/step1_merge_${TIMESTAMP}.log"

    # 執行合併
    python3 scripts/merge_phase9_datasets.py 2>&1 | tee "${merge_log}"

    # 驗證結果
    local output_dir="datasets/yolo_harmony_v2_phase9_merged"
    if [ -d "$output_dir" ]; then
        local train_count=$(find "${output_dir}/train/labels" -name "*.txt" 2>/dev/null | wc -l)
        local val_count=$(find "${output_dir}/val/labels" -name "*.txt" 2>/dev/null | wc -l)

        log "INFO" "合併完成！"
        log "INFO" "  訓練集: ${train_count} 個文件"
        log "INFO" "  驗證集: ${val_count} 個文件"

        # 檢查合併報告
        if [ -f "${output_dir}/merge_report.json" ]; then
            local total=$(python3 -c "import json; print(json.load(open('${output_dir}/merge_report.json'))['total_annotations'])")
            log "INFO" "  總標註數: ${total}"
        fi
    else
        log "ERROR" "合併輸出目錄不存在: ${output_dir}"
        exit 1
    fi

    log "INFO" "Step 1 完成 ✓"
    echo ""
}

#-------------------------------------------------------------------------------
# Step 2: 驗證合併後數據集
#-------------------------------------------------------------------------------

step2_validate_dataset() {
    log "STEP" "=========================================="
    log "STEP" "Step 2: 驗證合併後數據集"
    log "STEP" "=========================================="

    cd "${TRAINING_DIR}"

    local dataset_dir="datasets/yolo_harmony_v2_phase9_merged"
    local yaml_file="${dataset_dir}/harmony_phase9_merged.yaml"

    # 檢查 YAML 配置
    if [ ! -f "$yaml_file" ]; then
        log "ERROR" "數據集配置不存在: ${yaml_file}"
        exit 1
    fi
    log "INFO" "數據集配置: ${yaml_file}"

    # 驗證圖片和標註對應
    log "INFO" "驗證圖片與標註對應..."
    python3 << 'EOF'
import os
from pathlib import Path

dataset = Path("datasets/yolo_harmony_v2_phase9_merged")
errors = []

for split in ['train', 'val']:
    images = set(p.stem for p in (dataset / split / 'images').glob('*.png'))
    labels = set(p.stem for p in (dataset / split / 'labels').glob('*.txt'))

    missing_labels = images - labels
    missing_images = labels - images

    if missing_labels:
        errors.append(f"{split}: {len(missing_labels)} 圖片缺少標註")
    if missing_images:
        errors.append(f"{split}: {len(missing_images)} 標註缺少圖片")

    print(f"  {split}: {len(images)} 圖片, {len(labels)} 標註")

if errors:
    print("警告:")
    for e in errors:
        print(f"  - {e}")
else:
    print("  ✓ 所有圖片和標註都對應")
EOF

    # 統計各類別數量
    log "INFO" "統計各類別標註數量..."
    python3 << 'EOF'
import os
from collections import defaultdict
from pathlib import Path

CLASS_NAMES = {
    0: 'notehead_filled', 1: 'notehead_hollow', 2: 'stem', 3: 'beam',
    4: 'flag_8th', 5: 'flag_16th', 6: 'flag_32nd', 7: 'augmentation_dot',
    8: 'tie', 9: 'clef_treble', 10: 'clef_bass', 11: 'clef_alto',
    12: 'clef_tenor', 13: 'accidental_sharp', 14: 'accidental_flat',
    15: 'accidental_natural', 16: 'accidental_double_sharp', 17: 'accidental_double_flat',
    18: 'rest_whole', 19: 'rest_half', 20: 'rest_quarter', 21: 'rest_8th',
    22: 'rest_16th', 23: 'barline', 24: 'barline_double', 25: 'barline_final',
    26: 'barline_repeat', 27: 'time_signature', 28: 'key_signature',
    29: 'fermata', 30: 'dynamic_soft', 31: 'dynamic_loud', 32: 'ledger_line'
}

counts = defaultdict(int)
dataset = Path("datasets/yolo_harmony_v2_phase9_merged")

for split in ['train', 'val']:
    for label_file in (dataset / split / 'labels').glob('*.txt'):
        with open(label_file) as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    counts[int(parts[0])] += 1

total = sum(counts.values())
print(f"  總標註數: {total:,}")

# 找出樣本最少的 5 個類別
sorted_counts = sorted(counts.items(), key=lambda x: x[1])
print("  樣本最少的類別:")
for cid, cnt in sorted_counts[:5]:
    print(f"    {CLASS_NAMES[cid]}: {cnt:,}")
EOF

    log "INFO" "Step 2 完成 ✓"
    echo ""
}

#-------------------------------------------------------------------------------
# Step 3: 執行 Phase 9 訓練
#-------------------------------------------------------------------------------

step3_train() {
    log "STEP" "=========================================="
    log "STEP" "Step 3: 執行 Phase 9 訓練"
    log "STEP" "=========================================="

    cd "${TRAINING_DIR}"

    # 再次檢查 GPU
    check_gpu

    # 檢查訓練腳本
    if [ ! -f "yolo12_train_phase9.py" ]; then
        log "ERROR" "訓練腳本不存在: yolo12_train_phase9.py"
        exit 1
    fi

    log "INFO" "開始 Phase 9 訓練..."
    log "INFO" "預估時間: 6-9 小時"
    log "INFO" "訓練日誌: ${LOG_DIR}/step3_train_${TIMESTAMP}.log"

    local train_log="${LOG_DIR}/step3_train_${TIMESTAMP}.log"
    local start_time=$(date +%s)

    # 執行訓練（使用 --auto 參數自動執行）
    python3 yolo12_train_phase9.py --auto 2>&1 | tee "${train_log}"

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local hours=$((duration / 3600))
    local minutes=$(((duration % 3600) / 60))

    log "INFO" "訓練完成！"
    log "INFO" "總耗時: ${hours} 小時 ${minutes} 分鐘"

    # 發送通知
    send_notification "Phase 9 訓練完成" "總耗時: ${hours}h ${minutes}m"

    log "INFO" "Step 3 完成 ✓"
    echo ""
}

#-------------------------------------------------------------------------------
# Step 4: 評估與生成報告
#-------------------------------------------------------------------------------

step4_evaluate() {
    log "STEP" "=========================================="
    log "STEP" "Step 4: 評估與生成報告"
    log "STEP" "=========================================="

    cd "${TRAINING_DIR}"

    local results_dir="harmony_omr_v2_phase9/merged_data_training"

    if [ ! -d "$results_dir" ]; then
        log "ERROR" "訓練結果目錄不存在: ${results_dir}"
        exit 1
    fi

    log "INFO" "分析訓練結果..."

    # 提取最終指標
    python3 << EOF
import csv
from pathlib import Path

results_file = Path("${results_dir}/results.csv")
if results_file.exists():
    with open(results_file) as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

        if rows:
            last_row = rows[-1]
            epoch = last_row[0]

            # 找到指標欄位
            try:
                map50_idx = header.index('metrics/mAP50(B)')
                map5095_idx = header.index('metrics/mAP50-95(B)')

                map50 = float(last_row[map50_idx])
                map5095 = float(last_row[map5095_idx])

                print(f"最終結果 (Epoch {epoch}):")
                print(f"  mAP50:    {map50:.4f}")
                print(f"  mAP50-95: {map5095:.4f}")

                # 與 Phase 8 比較
                phase8_map50 = 0.644
                phase8_map5095 = 0.581

                print(f"\n與 Phase 8 比較:")
                print(f"  mAP50:    {map50:.4f} vs {phase8_map50:.4f} ({(map50-phase8_map50)*100:+.1f}%)")
                print(f"  mAP50-95: {map5095:.4f} vs {phase8_map5095:.4f} ({(map5095-phase8_map5095)*100:+.1f}%)")

                # 判斷是否達標
                print("\n目標達成:")
                print(f"  mAP50 >= 0.68: {'✓' if map50 >= 0.68 else '✗'}")
                print(f"  mAP50 >= 0.70: {'✓' if map50 >= 0.70 else '✗'}")
            except (ValueError, IndexError) as e:
                print(f"無法解析結果: {e}")
else:
    print("找不到結果文件")
EOF

    # 檢查最佳權重
    local best_weights="${results_dir}/weights/best.pt"
    if [ -f "$best_weights" ]; then
        local size=$(du -h "$best_weights" | cut -f1)
        log "INFO" "最佳權重: ${best_weights} (${size})"
    fi

    log "INFO" "Step 4 完成 ✓"
    echo ""
}

#-------------------------------------------------------------------------------
# 主程序
#-------------------------------------------------------------------------------

main() {
    echo ""
    log "STEP" "=============================================="
    log "STEP" "Phase 9 完整執行管道"
    log "STEP" "=============================================="
    log "INFO" "開始時間: $(date '+%Y-%m-%d %H:%M:%S')"
    log "INFO" "日誌文件: ${MAIN_LOG}"
    echo ""

    # 檢查 GPU
    check_gpu
    echo ""

    # 執行步驟
    step1_merge_datasets
    step2_validate_dataset
    step3_train
    step4_evaluate

    # 完成
    log "STEP" "=============================================="
    log "STEP" "Phase 9 執行完成！"
    log "STEP" "=============================================="
    log "INFO" "結束時間: $(date '+%Y-%m-%d %H:%M:%S')"
    log "INFO" "完整日誌: ${MAIN_LOG}"
    echo ""

    # 發送最終通知
    send_notification "Phase 9 Pipeline 完成" "請查看結果"
}

# 執行主程序
main "$@"
