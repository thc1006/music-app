#!/bin/bash
#===============================================================================
# Phase 9 訓練進度監控腳本
#===============================================================================
#
# 功能：
#   - 監控訓練進度
#   - 顯示當前 GPU 狀態
#   - 顯示最新的訓練指標
#
# 執行方式：
#   ./check_phase9_progress.sh
#   watch -n 30 ./check_phase9_progress.sh  # 每 30 秒自動刷新
#
#===============================================================================

TRAINING_DIR="/home/thc1006/dev/music-app/training"
PHASE9_DIR="${TRAINING_DIR}/harmony_omr_v2_phase9/merged_data_training"

# 顏色
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}        Phase 9 訓練進度監控${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "時間: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

#-------------------------------------------------------------------------------
# GPU 狀態
#-------------------------------------------------------------------------------
echo -e "${YELLOW}📊 GPU 狀態${NC}"
echo "────────────────────────────────────────────────────────────"

if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null | \
    while IFS=, read -r name util mem_used mem_total temp power; do
        echo "GPU:     $name"
        echo "使用率:  ${util}%"
        echo "記憶體:  ${mem_used} / ${mem_total} MB"
        echo "溫度:    ${temp}°C"
        echo "功耗:    ${power}W"
    done
else
    echo "nvidia-smi 不可用"
fi
echo ""

#-------------------------------------------------------------------------------
# 訓練進度
#-------------------------------------------------------------------------------
echo -e "${YELLOW}📈 訓練進度${NC}"
echo "────────────────────────────────────────────────────────────"

RESULTS_CSV="${PHASE9_DIR}/results.csv"

if [ -f "$RESULTS_CSV" ]; then
    # 獲取總行數（減去標題）
    total_lines=$(wc -l < "$RESULTS_CSV")
    current_epoch=$((total_lines - 1))
    target_epochs=100

    if [ $current_epoch -gt 0 ]; then
        progress=$((current_epoch * 100 / target_epochs))

        # 進度條
        bar_width=40
        filled=$((progress * bar_width / 100))
        empty=$((bar_width - filled))

        printf "進度:    ["
        printf "%${filled}s" | tr ' ' '█'
        printf "%${empty}s" | tr ' ' '░'
        printf "] %d%% (%d/%d)\n" "$progress" "$current_epoch" "$target_epochs"

        # 獲取最新指標
        last_line=$(tail -1 "$RESULTS_CSV")

        # 解析 CSV
        IFS=',' read -ra values <<< "$last_line"

        # 假設欄位順序（根據 YOLO 輸出）
        # epoch,time,train/box_loss,train/cls_loss,train/dfl_loss,
        # metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B),...

        if [ ${#values[@]} -ge 9 ]; then
            box_loss=${values[2]}
            cls_loss=${values[3]}
            dfl_loss=${values[4]}
            precision=${values[5]}
            recall=${values[6]}
            map50=${values[7]}
            map5095=${values[8]}

            echo ""
            echo "當前指標 (Epoch $current_epoch):"
            echo "  mAP50:      $map50"
            echo "  mAP50-95:   $map5095"
            echo "  Precision:  $precision"
            echo "  Recall:     $recall"
            echo ""
            echo "損失值:"
            echo "  box_loss:   $box_loss"
            echo "  cls_loss:   $cls_loss"
            echo "  dfl_loss:   $dfl_loss"
        fi

        # 估計剩餘時間
        if [ ${#values[@]} -ge 2 ]; then
            elapsed_time=${values[1]}
            if [ -n "$elapsed_time" ] && [ "$elapsed_time" != "time" ]; then
                avg_time_per_epoch=$(echo "scale=2; $elapsed_time / $current_epoch" | bc 2>/dev/null)
                remaining_epochs=$((target_epochs - current_epoch))
                remaining_time=$(echo "scale=0; $avg_time_per_epoch * $remaining_epochs / 3600" | bc 2>/dev/null)

                if [ -n "$remaining_time" ]; then
                    echo ""
                    echo "預估剩餘時間: ~${remaining_time} 小時"
                fi
            fi
        fi
    else
        echo "訓練尚未開始或剛開始"
    fi
else
    echo "訓練尚未開始（找不到 results.csv）"

    # 檢查合併數據集是否存在
    MERGED_DIR="${TRAINING_DIR}/datasets/yolo_harmony_v2_phase9_merged"
    if [ -d "$MERGED_DIR" ]; then
        echo ""
        echo "✓ 合併後數據集已準備就緒"
    else
        echo ""
        echo "⚠ 合併後數據集尚未生成"
        echo "  請先執行: python scripts/merge_phase9_datasets.py"
    fi
fi
echo ""

#-------------------------------------------------------------------------------
# 最近日誌
#-------------------------------------------------------------------------------
echo -e "${YELLOW}📝 最近日誌${NC}"
echo "────────────────────────────────────────────────────────────"

# 找到最新的訓練日誌
LOG_DIR="${TRAINING_DIR}/logs/phase9"
if [ -d "$LOG_DIR" ]; then
    latest_log=$(ls -t "$LOG_DIR"/step3_train_*.log 2>/dev/null | head -1)
    if [ -n "$latest_log" ]; then
        echo "日誌文件: $latest_log"
        echo ""
        tail -5 "$latest_log" 2>/dev/null | head -5
    else
        echo "暫無訓練日誌"
    fi
else
    echo "日誌目錄不存在"
fi
echo ""

#-------------------------------------------------------------------------------
# Phase 8 比較
#-------------------------------------------------------------------------------
echo -e "${YELLOW}📊 與 Phase 8 比較${NC}"
echo "────────────────────────────────────────────────────────────"
echo "Phase 8 最終結果:"
echo "  mAP50:    0.6444"
echo "  mAP50-95: 0.5809"
echo ""

if [ -f "$RESULTS_CSV" ] && [ $current_epoch -gt 0 ]; then
    # 計算改進
    if [ -n "$map50" ]; then
        improvement=$(echo "scale=4; ($map50 - 0.6444) * 100" | bc 2>/dev/null)
        if [ -n "$improvement" ]; then
            echo "當前改進: ${improvement}%"
        fi
    fi
fi

echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
