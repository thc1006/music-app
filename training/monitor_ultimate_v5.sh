#!/bin/bash
#
# YOLO12 Ultimate v5 訓練監控腳本
# =================================
#
# 使用方式:
#   ./monitor_ultimate_v5.sh        # 持續監控
#   ./monitor_ultimate_v5.sh status # 查看當前狀態
#   ./monitor_ultimate_v5.sh stop   # 停止訓練
#

cd /home/thc1006/dev/music-app/training

case "$1" in
    status)
        echo "=========================================="
        echo "YOLO12 Ultimate v5 訓練狀態"
        echo "=========================================="

        # 檢查進程
        if pgrep -f "yolo12_ultimate_v5" > /dev/null; then
            PID=$(cat logs/ultimate_v5_training.pid 2>/dev/null)
            echo "✅ 訓練正在運行 (PID: $PID)"
        else
            echo "❌ 訓練未運行"
        fi
        echo ""

        # GPU 狀態
        echo "GPU 狀態:"
        nvidia-smi --query-gpu=name,temperature.gpu,memory.used,memory.total,utilization.gpu --format=csv,noheader
        echo ""

        # 最新日誌
        echo "最新進度 (最後 5 行):"
        tail -5 logs/ultimate_v5_training.log 2>/dev/null | grep -E "Epoch|mAP|loss"
        echo ""

        # 檢查結果
        if [ -f "harmony_omr_v2_ultimate_v5/highres_1280/results.csv" ]; then
            echo "訓練結果 (最後 5 epochs):"
            tail -6 harmony_omr_v2_ultimate_v5/highres_1280/results.csv | head -5 | \
                awk -F',' 'NR>1 {printf "Epoch %d: mAP50=%.4f, mAP50-95=%.4f\n", $1, $8, $9}'
        fi
        ;;

    stop)
        echo "正在停止訓練..."
        PID=$(cat logs/ultimate_v5_training.pid 2>/dev/null)
        if [ -n "$PID" ]; then
            kill $PID 2>/dev/null
            echo "✅ 已發送停止信號 (PID: $PID)"
        else
            pkill -f "yolo12_ultimate_v5"
            echo "✅ 已停止所有相關進程"
        fi
        ;;

    *)
        echo "=========================================="
        echo "YOLO12 Ultimate v5 訓練監控"
        echo "=========================================="
        echo "按 Ctrl+C 退出監控"
        echo ""
        tail -f logs/ultimate_v5_training.log
        ;;
esac
