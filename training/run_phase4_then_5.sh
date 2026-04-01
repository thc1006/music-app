#!/bin/bash
# 自動執行 Phase 4 -> Phase 5
# Phase 4 已在運行中，此腳本監控並在完成後啟動 Phase 5

LOG_DIR="logs"
# 只取主進程 PID (第一個)
PHASE4_PID=$(pgrep -of "exp4_focal_loss.py" 2>/dev/null | head -1)

echo "========================================"
echo "Phase 4 -> Phase 5 自動執行監控"
echo "========================================"
echo "開始時間: $(date)"
echo ""

if [ -z "$PHASE4_PID" ]; then
    echo "Phase 4 未運行，直接啟動 Phase 5..."
else
    echo "Phase 4 運行中 (PID: $PHASE4_PID)"
    echo "等待 Phase 4 完成..."
    echo ""

    # 等待 Phase 4 完成
    while kill -0 $PHASE4_PID 2>/dev/null; do
        # 每 60 秒報告一次狀態
        PROGRESS=$(tail -1 $LOG_DIR/exp4_finetune.log 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' | grep -oE '[0-9]+/[0-9]+.*[0-9]+%' | tail -1)
        echo "[$(date '+%H:%M:%S')] Phase 4: $PROGRESS"
        sleep 60
    done

    echo ""
    echo "========================================"
    echo "Phase 4 完成! $(date)"
    echo "========================================"

    # 檢查 Phase 4 結果
    if [ -f "harmony_omr_v2_experiments/exp4_finetune/weights/best.pt" ]; then
        echo "Phase 4 模型已儲存: harmony_omr_v2_experiments/exp4_finetune/weights/best.pt"
    fi
fi

echo ""
echo "========================================"
echo "啟動 Phase 5: DINOv3 蒸餾"
echo "========================================"
echo ""

# 清理 GPU 記憶體
python -c "import torch; torch.cuda.empty_cache(); print('GPU 記憶體已清理')"

# 啟動 Phase 5
cd /home/thc1006/dev/music-app/training
source venv_yolo12/bin/activate
python exp5_dinov3_distill.py 2>&1 | tee $LOG_DIR/exp5_dinov3.log

echo ""
echo "========================================"
echo "Phase 5 完成! $(date)"
echo "========================================"
